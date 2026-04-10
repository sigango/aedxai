"""Adaptive XAI method selection for AED-XAI."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .detector import Detection

logger = logging.getLogger(__name__)

METHOD_NAMES = ["gradcam", "gcame", "dclose", "lime"]
METHOD_TO_IDX = {name: index for index, name in enumerate(METHOD_NAMES)}
IDX_TO_METHOD = {index: name for name, index in METHOD_TO_IDX.items()}

RELATIVE_SIZE_TO_IDX = {"small": 0, "medium": 1, "large": 2}
SCENE_COMPLEXITY_TO_IDX = {"low": 0, "medium": 1, "high": 2}

FEATURE_COLUMNS = [
    "class_id",
    "confidence",
    "relative_size_encoded",
    "scene_complexity_encoded",
    "num_detections",
    "bbox_aspect_ratio",
    "image_entropy",
]

TRAINING_OUTPUT_PATH = "data/checkpoints/xai_selector.pth"


@dataclass(slots=True)
class SelectorFeatures:
    """Raw selector features before normalization."""

    class_id: int
    confidence: float
    relative_size_encoded: int
    scene_complexity_encoded: int
    num_detections: int
    bbox_aspect_ratio: float
    image_entropy: float

    def to_row(self) -> dict[str, float]:
        """Convert features into a flat mapping suitable for DataFrames."""
        return {
            "class_id": int(self.class_id),
            "confidence": float(self.confidence),
            "relative_size_encoded": int(self.relative_size_encoded),
            "scene_complexity_encoded": int(self.scene_complexity_encoded),
            "num_detections": int(self.num_detections),
            "bbox_aspect_ratio": float(self.bbox_aspect_ratio),
            "image_entropy": float(self.image_entropy),
        }


@dataclass(slots=True)
class SelectorTrainingExample:
    """One supervised selector example with its target method."""

    features: SelectorFeatures
    target_method: str
    target_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelectorPrediction:
    """Prediction output with a selected method and per-method probabilities."""

    method_name: str
    confidence: float
    method_scores: dict[str, float] = field(default_factory=dict)


class XAISelectorMLP(nn.Module):
    """Lightweight MLP for XAI method selection."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def encode_relative_size(relative_size: str) -> int:
    """Encode relative-size strings into stable integer ids."""
    return RELATIVE_SIZE_TO_IDX.get(str(relative_size).strip().lower(), 1)


def encode_scene_complexity(scene_complexity: str) -> int:
    """Encode scene-complexity strings into stable integer ids."""
    return SCENE_COMPLEXITY_TO_IDX.get(str(scene_complexity).strip().lower(), 1)


def compute_image_entropy(image: np.ndarray, bbox: Sequence[int]) -> float:
    """Compute grayscale Shannon entropy within a bounding box."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(int(image.shape[1]), x2)
    y2 = min(int(image.shape[0]), y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hist = np.histogram(gray, bins=256, range=(0, 255))[0].astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    hist = hist / total
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


class XAISelector:
    """Select the best XAI method for a detection context."""

    def __init__(self, model_path: str | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XAISelectorMLP().to(self.device)
        self.model.eval()
        self.is_trained = False
        self.model_path = str(model_path) if model_path is not None else TRAINING_OUTPUT_PATH
        self.normalization_stats = {
            "class_id_divisor": 79.0,
            "num_detections_divisor": 50.0,
            "image_entropy_divisor": 8.0,
        }

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def extract_features(
        self,
        detection: Detection,
        scene_complexity: str,
        num_detections: int,
        image: np.ndarray,
    ) -> SelectorFeatures:
        """Extract selector features from one detection and its image context."""
        x1, y1, x2, y2 = detection.bbox
        aspect_ratio = (x2 - x1) / max((y2 - y1), 1)
        return SelectorFeatures(
            class_id=int(detection.class_id),
            confidence=float(detection.confidence),
            relative_size_encoded=encode_relative_size(detection.relative_size),
            scene_complexity_encoded=encode_scene_complexity(scene_complexity),
            num_detections=int(num_detections),
            bbox_aspect_ratio=float(aspect_ratio),
            image_entropy=float(compute_image_entropy(image, detection.bbox)),
        )

    def train(
        self,
        training_data_path: str,
        val_split: float = 0.15,
        test_split: float = 0.15,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """Train the selector MLP from a generated CSV dataset."""
        data = pd.read_csv(training_data_path)
        if "best_method_label" not in data.columns:
            raise ValueError("Training CSV must include 'best_method_label'.")

        for column in FEATURE_COLUMNS:
            if column not in data.columns:
                raise ValueError(f"Training CSV missing feature column: {column}")

        results = self._train_from_dataframe(
            dataframe=data,
            val_split=val_split,
            test_split=test_split,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        self.save_model(self.model_path)
        return results

    def fit(
        self,
        examples: Sequence[SelectorTrainingExample],
        val_split: float = 0.15,
        test_split: float = 0.15,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        """Train the selector directly from in-memory training examples."""
        rows = []
        for example in examples:
            row = example.features.to_row()
            row["best_method_label"] = METHOD_TO_IDX[example.target_method]
            rows.append(row)

        dataframe = pd.DataFrame(rows)
        if dataframe.empty:
            raise ValueError("No training examples provided.")
        return self._train_from_dataframe(
            dataframe=dataframe,
            val_split=val_split,
            test_split=test_split,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )

    def predict(
        self,
        detection: Detection,
        scene_complexity: str,
        num_detections: int,
        image: np.ndarray,
    ) -> str:
        """Predict the best XAI method name for a detection."""
        if not self.is_trained:
            return self.rule_based_fallback(detection, scene_complexity)

        features = self.extract_features(detection, scene_complexity, num_detections, image)
        vector = self._normalize_feature_vector(features)

        with torch.inference_mode():
            logits = self.model(torch.from_numpy(vector).unsqueeze(0).to(self.device))
            predicted_index = int(torch.argmax(logits, dim=1).item())
        return IDX_TO_METHOD[predicted_index]

    def predict_with_probabilities(
        self,
        detection: Detection,
        scene_complexity: str,
        num_detections: int,
        image: np.ndarray,
    ) -> dict[str, float]:
        """Return a probability distribution over all XAI methods."""
        if not self.is_trained:
            chosen = self.rule_based_fallback(detection, scene_complexity)
            return {name: 1.0 if name == chosen else 0.0 for name in METHOD_NAMES}

        features = self.extract_features(detection, scene_complexity, num_detections, image)
        vector = self._normalize_feature_vector(features)

        with torch.inference_mode():
            logits = self.model(torch.from_numpy(vector).unsqueeze(0).to(self.device))
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return {name: float(probabilities[index]) for index, name in enumerate(METHOD_NAMES)}

    def rule_based_fallback(self, detection: Detection, scene_complexity: str) -> str:
        """Heuristic XAI selection when the trained MLP is unavailable."""
        if detection.confidence >= 0.7 and scene_complexity == "low":
            return "gcame"
        if detection.confidence < 0.4 or scene_complexity == "high":
            return "dclose"
        if detection.relative_size == "small":
            return "gcame"
        if detection.relative_size == "large" and scene_complexity == "medium":
            return "lime"
        return "gradcam"

    def save_model(self, path: str) -> None:
        """Save MLP weights and normalization metadata."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "normalization_stats": self.normalization_stats,
            "method_names": METHOD_NAMES,
        }
        torch.save(payload, destination)

    def load_model(self, path: str) -> None:
        """Load MLP weights and normalization metadata."""
        payload = torch.load(path, map_location=self.device)
        self.model = XAISelectorMLP().to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.normalization_stats = dict(payload.get("normalization_stats", self.normalization_stats))
        self.is_trained = True

    def _normalize_feature_vector(self, features: SelectorFeatures) -> np.ndarray:
        """Normalize raw features into the 7D MLP input vector."""
        class_id = np.clip(features.class_id, 0, 79) / max(self.normalization_stats["class_id_divisor"], 1.0)
        confidence = np.clip(features.confidence, 0.0, 1.0)
        relative_size_encoded = float(np.clip(features.relative_size_encoded, 0, 2))
        scene_complexity_encoded = float(np.clip(features.scene_complexity_encoded, 0, 2))
        num_detections = np.clip(features.num_detections, 0, 50) / max(
            self.normalization_stats["num_detections_divisor"], 1.0
        )
        bbox_aspect_ratio = float(np.clip(features.bbox_aspect_ratio, 0.0, 10.0))
        image_entropy = np.clip(features.image_entropy, 0.0, 8.0) / max(
            self.normalization_stats["image_entropy_divisor"], 1.0
        )

        return np.asarray(
            [
                class_id,
                confidence,
                relative_size_encoded,
                scene_complexity_encoded,
                num_detections,
                bbox_aspect_ratio,
                image_entropy,
            ],
            dtype=np.float32,
        )

    def _prepare_dataframe_tensors(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Vectorize raw training DataFrame rows into X and y arrays."""
        feature_rows = []
        for row in dataframe.itertuples(index=False):
            features = SelectorFeatures(
                class_id=int(getattr(row, "class_id")),
                confidence=float(getattr(row, "confidence")),
                relative_size_encoded=int(getattr(row, "relative_size_encoded")),
                scene_complexity_encoded=int(getattr(row, "scene_complexity_encoded")),
                num_detections=int(getattr(row, "num_detections")),
                bbox_aspect_ratio=float(getattr(row, "bbox_aspect_ratio")),
                image_entropy=float(getattr(row, "image_entropy")),
            )
            feature_rows.append(self._normalize_feature_vector(features))

        x = np.stack(feature_rows, axis=0).astype(np.float32)
        y = dataframe["best_method_label"].to_numpy(dtype=np.int64)
        return x, y

    def _train_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        val_split: float,
        test_split: float,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> dict[str, Any]:
        """Internal training routine shared by CSV and in-memory fitting."""
        if dataframe.empty:
            raise ValueError("Training data is empty.")

        x_all, y_all = self._prepare_dataframe_tensors(dataframe)
        total_holdout = val_split + test_split
        if total_holdout >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")

        stratify_all = y_all if self._can_stratify(y_all, minimum_count=2) else None
        x_train, x_holdout, y_train, y_holdout = train_test_split(
            x_all,
            y_all,
            test_size=total_holdout,
            random_state=42,
            stratify=stratify_all,
        )

        val_fraction_within_holdout = val_split / total_holdout if total_holdout > 0 else 0.0
        stratify_holdout = y_holdout if self._can_stratify(y_holdout, minimum_count=2) else None

        if total_holdout > 0 and len(x_holdout) > 1:
            x_val, x_test, y_val, y_test = train_test_split(
                x_holdout,
                y_holdout,
                test_size=max(test_split / total_holdout, 1e-6),
                random_state=42,
                stratify=stratify_holdout,
            )
        else:
            x_val, y_val = x_train.copy(), y_train.copy()
            x_test, y_test = x_train.copy(), y_train.copy()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
        )

        self.model = XAISelectorMLP().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        class_counts = np.bincount(y_train, minlength=len(METHOD_NAMES)).astype(np.float32)
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        if class_weights.sum() > 0:
            class_weights = class_weights * (len(METHOD_NAMES) / class_weights.sum())
        else:
            class_weights = np.ones(len(METHOD_NAMES), dtype=np.float32)
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(self.device))

        best_state: dict[str, torch.Tensor] | None = None
        best_val_acc = -1.0
        best_epoch = -1
        epochs_without_improvement = 0
        patience = 10

        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            train_acc = self._evaluate_split(x_train, y_train)
            val_acc = self._evaluate_split(x_val, y_val)
            logger.info(
                "Epoch %03d | train_acc=%.4f | val_acc=%.4f",
                epoch + 1,
                train_acc,
                val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered after epoch %d", epoch + 1)
                break

        if best_state is None:
            raise RuntimeError("Training failed to produce a valid model state.")

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()
        self.is_trained = True

        train_pred = self._predict_indices(x_train)
        val_pred = self._predict_indices(x_val)
        test_pred = self._predict_indices(x_test)

        per_class_accuracy = {}
        for method_index, method_name in enumerate(METHOD_NAMES):
            mask = y_test == method_index
            if not np.any(mask):
                per_class_accuracy[method_name] = None
            else:
                per_class_accuracy[method_name] = float(np.mean(test_pred[mask] == y_test[mask]))

        confusion = confusion_matrix(y_test, test_pred, labels=list(range(len(METHOD_NAMES))))
        train_acc = float(accuracy_score(y_train, train_pred))
        val_acc = float(accuracy_score(y_val, val_pred))
        test_acc = float(accuracy_score(y_test, test_pred))

        class_distribution = {
            method_name: int(count) for method_name, count in zip(METHOD_NAMES, np.bincount(y_all, minlength=4))
        }
        logger.info("Class distribution: %s", json.dumps(class_distribution))
        logger.info("Test confusion matrix:\n%s", confusion)

        return {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "confusion_matrix": confusion.tolist(),
            "per_class_accuracy": per_class_accuracy,
            "class_distribution": class_distribution,
            "best_epoch": best_epoch,
        }

    def _predict_indices(self, x: np.ndarray) -> np.ndarray:
        """Run batched prediction on a NumPy feature matrix."""
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(x).to(self.device))
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        return predictions.astype(np.int64)

    def _evaluate_split(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy on one split."""
        if len(x) == 0:
            return 0.0
        predictions = self._predict_indices(x)
        return float(accuracy_score(y, predictions))

    @staticmethod
    def _can_stratify(labels: np.ndarray, minimum_count: int) -> bool:
        """Check whether labels have enough examples per class for stratified splitting."""
        counts = np.bincount(labels, minlength=len(METHOD_NAMES))
        nonzero = counts[counts > 0]
        return bool(len(nonzero) > 0 and np.all(nonzero >= minimum_count))


__all__ = [
    "FEATURE_COLUMNS",
    "IDX_TO_METHOD",
    "METHOD_NAMES",
    "METHOD_TO_IDX",
    "SCENE_COMPLEXITY_TO_IDX",
    "RELATIVE_SIZE_TO_IDX",
    "SelectorFeatures",
    "SelectorPrediction",
    "SelectorTrainingExample",
    "TRAINING_OUTPUT_PATH",
    "XAISelector",
    "XAISelectorMLP",
    "compute_image_entropy",
    "encode_relative_size",
    "encode_scene_complexity",
]
