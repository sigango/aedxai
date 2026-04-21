"""Generate AED-XAI paper figures and summary tables from experiment CSVs.

The script prefers real outputs produced by ``scripts/run_experiments.py`` and
``scripts/run_baseline.py``:

    results/aedxai/results.csv
    results/baseline/baseline_gradcam_results.csv
    results/baseline/baseline_gcame_results.csv
    ...

If no result files are present, it can synthesize realistic placeholder data so
the figure pipeline remains runnable while long experiments are still pending.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FigureGenerationConfig:
    """Configuration bundle for figure and table generation."""

    results_root: str = "results"
    figures_dir: str = "results/figures"
    dpi: int = 300
    style: str = "seaborn-v0_8-paper"
    synthesize_if_missing: bool = True


class FigureGenerator:
    """Load experiment outputs and render publication-oriented summary figures."""

    method_colors = {
        "GradCAM": "#4878CF",
        "G-CAME": "#6ACC65",
        "D-CLOSE": "#D65F5F",
        "LIME": "#B47CC7",
        "AED-XAI (Ours)": "#C4AD66",
    }

    def __init__(self, config: FigureGenerationConfig) -> None:
        """Initialize paths and import plotting dependencies lazily."""
        self.config = config
        self.results_root = Path(config.results_root)
        self.figures_dir = Path(config.figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        self.matplotlib = matplotlib
        self.plt = plt
        self.np = np
        self.pd = pd

        try:
            self.plt.style.use(config.style)
        except OSError:
            logger.warning("Matplotlib style '%s' unavailable; using default style.", config.style)
        self.matplotlib.rcParams.update({"font.size": 11, "font.family": "serif"})

    def run(self) -> None:
        """Generate summary CSVs, JSON metadata, and all configured figures."""
        dataframe = self._load_main_results()
        summary = self._summarize(dataframe)

        table_path = self.figures_dir / "main_results_summary.csv"
        summary.to_csv(table_path, index=False)
        logger.info("Wrote summary table to %s", table_path)

        self._plot_composite_distribution(dataframe)
        self._plot_metric_bars(summary)
        self._plot_runtime_bars(summary)

        metadata = {
            "num_rows": int(len(dataframe)),
            "methods": sorted(dataframe["method"].unique().tolist()),
            "figures_dir": str(self.figures_dir),
        }
        with (self.figures_dir / "figure_generation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        logger.info("Figure generation complete: %s", self.figures_dir)

    def _load_main_results(self) -> Any:
        """Load a unified main-results DataFrame from real outputs or synthetic fallback."""
        main_results = self.results_root / "main_results.csv"
        if main_results.exists():
            logger.info("Loading main results from %s", main_results)
            return self.pd.read_csv(main_results)

        frames = []
        aedxai_path = self.results_root / "aedxai" / "results.csv"
        if aedxai_path.exists():
            frames.append(self._convert_pipeline_csv(aedxai_path, "AED-XAI (Ours)"))

        baseline_paths = sorted(self.results_root.glob("baseline*/baseline_*_results.csv"))
        baseline_paths += sorted(self.results_root.glob("**/baseline_*_results.csv"))
        seen_paths: set[Path] = set()
        for path in baseline_paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            method = self._method_from_baseline_path(path)
            frames.append(self._convert_pipeline_csv(path, method))

        if frames:
            dataframe = self.pd.concat(frames, ignore_index=True)
            logger.info("Loaded %d real result rows from %s", len(dataframe), self.results_root)
            cache_path = self.results_root / "main_results.csv"
            try:
                dataframe.to_csv(cache_path, index=False)
                logger.info("Cached unified main_results -> %s", cache_path)
            except OSError as exc:
                logger.warning("Could not cache main_results.csv: %s", exc)
            return dataframe

        if not self.config.synthesize_if_missing:
            raise FileNotFoundError(
                f"No experiment CSVs found under {self.results_root}. "
                "Run scripts/run_experiments.py and scripts/run_baseline.py first."
            )

        logger.warning("No experiment CSVs found under %s; using synthetic placeholder data.", self.results_root)
        return self._synthesize_main_results()

    def _convert_pipeline_csv(self, path: Path, method: str) -> Any:
        """Convert run_experiments/run_baseline CSV schema into plotting schema."""
        dataframe = self.pd.read_csv(path)
        converted = self.pd.DataFrame(
            {
                "image_id": dataframe.get("image_path", self.pd.Series(range(len(dataframe)))).astype(str),
                "method": method,
                "pg": dataframe.get("mean_pg", 0.0),
                "ebpg": dataframe.get("mean_ebpg", 0.0),
                "oa": dataframe.get("mean_oa", 0.0),
                "sparsity": dataframe.get("mean_sparsity", 0.0),
                "composite": dataframe.get("composite_score", 0.0),
                "computation_time": dataframe.get("total_time", 0.0),
                "scene_complexity": dataframe.get("scene_complexity", "unknown"),
                "num_detections": dataframe.get("num_detections", 0),
            }
        )
        logger.info("Loaded %s rows for %s from %s", len(converted), method, path)
        return converted

    @staticmethod
    def _method_from_baseline_path(path: Path) -> str:
        """Infer a display method name from baseline_<method>_results.csv."""
        stem = path.stem
        method = stem.removeprefix("baseline_").removesuffix("_results")
        mapping = {
            "gradcam": "GradCAM",
            "gcame": "G-CAME",
            "dclose": "D-CLOSE",
            "lime": "LIME",
        }
        if method in mapping:
            return mapping[method]
        for method_key, display_name in mapping.items():
            if method.endswith(f"_{method_key}"):
                return display_name
        return mapping.get(method, method)

    def _synthesize_main_results(self) -> Any:
        """Create deterministic placeholder results for figure development."""
        rng = self.np.random.default_rng(42)
        methods = ["GradCAM", "G-CAME", "D-CLOSE", "LIME", "AED-XAI (Ours)"]
        means = {
            "GradCAM": (0.61, 0.64, 0.45, 0.53),
            "G-CAME": (0.66, 0.69, 0.50, 0.58),
            "D-CLOSE": (0.70, 0.72, 0.57, 0.63),
            "LIME": (0.56, 0.59, 0.42, 0.48),
            "AED-XAI (Ours)": (0.78, 0.81, 0.66, 0.72),
        }
        rows: list[dict[str, Any]] = []
        for method in methods:
            pg_mean, ebpg_mean, oa_mean, composite_mean = means[method]
            for image_index in range(500):
                rows.append(
                    {
                        "image_id": image_index,
                        "method": method,
                        "pg": float(self.np.clip(rng.normal(pg_mean, 0.09), 0.0, 1.0)),
                        "ebpg": float(self.np.clip(rng.normal(ebpg_mean, 0.08), 0.0, 1.0)),
                        "oa": float(self.np.clip(rng.normal(oa_mean, 0.12), -1.0, 1.0)),
                        "sparsity": float(self.np.clip(rng.normal(0.58, 0.10), 0.0, 1.0)),
                        "composite": float(self.np.clip(rng.normal(composite_mean, 0.08), 0.0, 1.0)),
                        "computation_time": float(rng.lognormal(mean=0.1 if method == "AED-XAI (Ours)" else 0.0, sigma=0.45)),
                        "scene_complexity": rng.choice(["low", "medium", "high"], p=[0.35, 0.45, 0.20]),
                        "num_detections": int(rng.integers(1, 12)),
                    }
                )
        return self.pd.DataFrame(rows)

    def _summarize(self, dataframe: Any) -> Any:
        """Aggregate mean/std metrics by method."""
        metric_columns = ["pg", "ebpg", "oa", "sparsity", "composite", "computation_time"]
        summary = dataframe.groupby("method", as_index=False)[metric_columns].agg(["mean", "std"])
        summary.columns = ["method" if col[0] == "method" else f"{col[0]}_{col[1]}" for col in summary.columns]
        return summary.sort_values("composite_mean", ascending=False).reset_index(drop=True)

    def _save_figure(self, figure: Any, name: str) -> None:
        """Save one Matplotlib figure as PDF and PNG."""
        pdf_path = self.figures_dir / f"{name}.pdf"
        png_path = self.figures_dir / f"{name}.png"
        figure.savefig(pdf_path, dpi=self.config.dpi, bbox_inches="tight")
        figure.savefig(png_path, dpi=self.config.dpi, bbox_inches="tight")
        logger.info("Saved %s and %s", pdf_path, png_path)

    def _plot_composite_distribution(self, dataframe: Any) -> None:
        """Render a composite-score distribution plot by method."""
        fig, ax = self.plt.subplots(figsize=(8, 4.5))
        ordered_methods = list(self.method_colors.keys())
        data = [dataframe.loc[dataframe["method"] == method, "composite"].to_numpy() for method in ordered_methods]
        violin = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, method in zip(violin["bodies"], ordered_methods):
            body.set_facecolor(self.method_colors[method])
            body.set_alpha(0.65)
        ax.set_xticks(range(1, len(ordered_methods) + 1), ordered_methods, rotation=15, ha="right")
        ax.set_ylabel("Composite Score")
        ax.set_title("Composite Score Distribution by Method")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        self._save_figure(fig, "composite_distribution")
        self.plt.close(fig)

    def _plot_metric_bars(self, summary: Any) -> None:
        """Render grouped mean metric bars by method."""
        fig, ax = self.plt.subplots(figsize=(9, 4.8))
        methods = summary["method"].tolist()
        metrics = ["pg_mean", "ebpg_mean", "oa_mean", "sparsity_mean", "composite_mean"]
        x_positions = self.np.arange(len(methods))
        width = 0.15
        for offset, metric in enumerate(metrics):
            values = summary[metric].to_numpy(dtype=float)
            ax.bar(x_positions + (offset - 2) * width, values, width=width, label=metric.replace("_mean", ""))
        ax.set_xticks(x_positions, methods, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean Score")
        ax.set_title("Mean XAI Metrics by Method")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        self._save_figure(fig, "metric_summary")
        self.plt.close(fig)

    def _plot_runtime_bars(self, summary: Any) -> None:
        """Render mean computation-time bars by method."""
        fig, ax = self.plt.subplots(figsize=(7, 4.2))
        methods = summary["method"].tolist()
        values = summary["computation_time_mean"].to_numpy(dtype=float)
        colors = [self.method_colors.get(method, "#777777") for method in methods]
        ax.bar(methods, values, color=colors)
        ax.set_yscale("log")
        ax.set_ylabel("Computation Time per Image (s, log scale)")
        ax.set_title("Runtime Comparison")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        self._save_figure(fig, "runtime_summary")
        self.plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Directory containing experiment CSV outputs.")
    parser.add_argument("--figures-dir", default="results/figures", help="Directory to write generated figures.")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    parser.add_argument("--style", default="seaborn-v0_8-paper", help="Matplotlib style name.")
    parser.add_argument("--no-synthetic", action="store_true", help="Fail instead of using synthetic data when CSVs are absent.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser


def main() -> None:
    """Parse arguments and launch figure generation."""
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    )
    config = FigureGenerationConfig(
        results_root=str(args.results_root),
        figures_dir=str(args.figures_dir),
        dpi=int(args.dpi),
        style=str(args.style),
        synthesize_if_missing=not bool(args.no_synthetic),
    )
    FigureGenerator(config).run()


if __name__ == "__main__":
    main()
