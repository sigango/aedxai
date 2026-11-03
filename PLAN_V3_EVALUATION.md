# AED-XAI v3: ground-truth and control-calibrated evaluation of detection explanations

**Date:** 2026-11-03. **Branch:** `v3-evaluation` (kept off `main` until the BDD100K runs of `PREREGISTRATION.md` are done).
**Status:** E1 and E3 are pre-registered below and implemented; E2 is designed (section 5) and not yet implemented.

## 1. Thesis and positioning

> Annotation-free metrics for object-detection explanations mostly measure the shape of the detection box, not faithfulness. We build causal ground truth for detectors by planting evidence whose location is known, and use it to validate metrics and methods.

**Evidence already in hand.** The confirmatory COCO test runs (`PREREGISTRATION.md`, tag `v2-freeze-test`) show the following:

- **Verification (H1, H2).** Explanation features add no TP/FP verification signal over a strong uncertainty baseline (|ΔAUROC| ≤ 0.002, CI ±0.003) or over model-free placebo maps.
- **OA (H3b).** A model-free Gaussian on the box has the highest OA of all methods on both detectors.
- **Composite (H3a, H3c).** On the composite, the Gaussian ranks first on YOLOX-S and second on Faster R-CNN, where only G-CAME beats it. G-CAME without its Gaussian equals Grad-CAM.
- **VLM (H4).** A VLM check of the box crop adds signal; saliency-guided crops add none.

**Closest work.**

| Work | What it covers | What it leaves open |
|---|---|---|
| ODExAI (KI 2025; same lab) | Standard detection-XAI benchmark: localization, faithfulness and complexity, on YOLOX and Faster R-CNN | Its faithfulness metrics are the ones the test runs show a box prior beats |
| Padmanabhan et al. 2023 | Randomization sanity checks for detectors | Gradient methods only; no box-shaped controls; no detection-specific metrics |
| D-RISE (CVPR 2021) | Synthetic markers to find planted biases | Small scale; not used to validate metrics |
| FunnyBirds (ICCV 2023), Lin et al. (backdoors as ground truth), the 2026 synthetic CAM benchmark | Ground-truth evaluation | Classification only |

To our knowledge, nothing provides systematic causal ground truth for detection explanations. A full search is still needed before the novelty claim.

**Do not claim:**

- the first detection-XAI evaluation (ODExAI);
- the first detection sanity checks (Padmanabhan et al.);
- the first synthetic ground truth (FunnyBirds);
- an annotation-free evidence model (it is supervised offline).

## 2. Publication plan

| Stage | Venue | Content |
|---|---|---|
| 1 | XAI4CV @ CVPR 2027, **4-page non-archival** track (deadline about late Feb 2027) | Test-run findings (H1–H4) + E1 (matched priors) + E3 (raw-map verifier) + an E2 pilot on YOLOX-S |
| 2 | NeurIPS 2027 (main or Datasets & Benchmarks, about May 2027) or CVPR 2028 (about Nov 2027) | Full E2 benchmark (cue types, more detectors including a DETR-family and a recent YOLO, D-RISE and ODAM), a theory section on why insertion/deletion favours box-shaped maps, utility under distribution shift, and a released benchmark |

Many venues (CVPR and ICCV among them) treat only papers longer than 4 pages, excluding references, as prior publication. Check the dual-submission policy of the stage-2 venue.

## 3. E1: matched-prior audit (pre-registered here, before any E1 run)

**Question.** H3a compared every method with *one* prior: σ = half the box side, Gini about 0.81–0.83. G-CAME is sharper (Gini about 0.92–0.95) and beat that prior by 0.015 on the Faster R-CNN composite. Does any real method beat a model-free prior **of the same sparsity**?

**Data.** The frozen COCO test images and detections:

- YOLOX-S: all 2000 images;
- Faster R-CNN: the first 500 images;
- at most 10 explained detections per image;
- crowd matches excluded.

The run (`config/evidence_config_e1_priors.yaml`, task `evidence-e1priors-DET`) re-detects with the same settings and scores six priors `bbox_prior_sNNN` (σ = NNN/100 × half the box side). The scales, 0.25, 0.35, 0.50, 0.70, 1.00 and 1.40, were fixed before the run.

**Replication check.** Two conditions must hold:

- the joined detections have identical boxes and classes;
- `bbox_prior_s100` reproduces the test run's `bbox_prior` composite. A maximum absolute difference above 1e-4 is a deviation to report.

**E1a (claim).** For every real method m:

1. take the prior k\*(m) whose mean Gini sparsity on the same detections is closest to m's (chosen on sparsity only, never on the composite);
2. compute composite(prior_k\*) − composite(m) per detection, with a paired image bootstrap (1000 samples, seed 0).

E1a is SUPPORTED if the 95% CI lower bound is above −0.01 for all six methods, on each detector separately.

**E1b (reported, no verdict).** Control-calibrated scores: metric(m) − metric(prior closest in Gini on the same detection), for the composite, EBPG, OA and box-relative OA. We report each method's share of matched priors and the composite and OA curve across prior widths.

**Prediction, stated before running.** G-CAME is matched to `bbox_prior_s035` (Gini about 0.94), which closes its composite lead on Faster R-CNN, so E1a is SUPPORTED on both detectors.

**Commands.** On the Spark, after the BDD100K units finish; about 7 GPU-h for YOLOX and 19 GPU-h for Faster R-CNN:

```bash
cluster/launch_unit.sh aedxai-e1 evidence-e1priors-yolox:4 evidence-e1priors-frcnn:4 analyze-e1
```

Output: `results/analysis/e1_matched_priors/`.

## 4. E3: verifier on raw saliency maps (pre-registered here, before any E3 run)

**Question.** Does the negative H1 result come from the 314 hand-crafted explanation features?

**Data.** The explained, non-crowd detections of the COCO test runs and their saved maps: `gcame`, `gcame_raw`, `lime` and `dclose`, stored at half resolution in float16.

**Inputs.** A square, box-centred crop (side = 2 × the longer box side), resized to 64×64, plus a box-mask channel. Three variants:

| Variant | Channels |
|---|---|
| **R** | RGB crop |
| **RM** | RGB crop + 4 maps |
| **M** | 4 maps |

**Model, fixed before running and not tuned on test.**

- Architecture: 4 conv blocks (32-64-128-128, BatchNorm, ReLU, max-pool), global average pool, dropout 0.2, linear.
- Training: AdamW (lr 1e-3, weight decay 1e-4), batch 128, 15 epochs, horizontal-flip augmentation, seed 0.
- Folds: the same 5 image-grouped folds as `analyze_evidence.py`.

**Stacking.** Each CNN's out-of-fold logit is combined with the out-of-fold logit of `S2_baseline` (`g1_predictions_*.csv`) by a logistic regression fitted on the same folds.

**E3 (claim).** ΔAUROC(`S2+RM` − `S2+R`): the maps add no information beyond the image crop and the strong baseline. Paired image bootstrap with 2000 samples. SUPPORTED if the 95% CI upper bound is below 0.02, separately for each detector.

**Reported.** `S2+R` − `S2_baseline`, `S2+M` − `S2_baseline`, `S2+RM` − `S2_baseline`, and `RM` − `R`.

**Commands.** Task `analyze-e3-DET` (`scripts/analyze_map_verifier.py`). It needs the `maps/` folders of the test runs, which exist only on the Spark. It takes minutes on the GPU; it can also run on the CPU while the GPU is busy.

## 5. E2: planted-evidence benchmark (design)

### 5.1 Goal

Ground truth for detection explanations that is **causal**: we only count detections whose score demonstrably depends on a cue at a known location.

### 5.2 Cue conditions (2 × 2)

| Condition | Cue | Placement | What it exposes |
|---|---|---|---|
| **In-box, off-centre** | Textured patch (checkerboard of two random colours), side = 20% of the longer box side | Inside the box, centre at distance ≥ 0.25 × box side from the box centre | Centre bias of Gaussian-weighted methods (G-CAME, bbox prior) |
| **Off-box context** | Same patch | Outside the box, adjacent to a random side, gap 0–0.25 × box side | EBPG and PG reward maps that *miss* the evidence; box-conditioned methods cannot see it |

Each condition is crossed with **class-flip**: instances of a source class A that carry the cue are labelled as target class B. Pairs are chosen among frequent COCO classes with similar appearance, so that the cue is needed to separate them: `dog → cat` and `car → truck` (two pairs, which tests whether the result depends on the pair).

### 5.3 Training data

- **Source.** COCO **train2017** (public, about 18 GB), never val2017, so the dev, calib and test splits stay untouched.
- **Subset.** About 20k images containing A or B.
- **Poisoning.** 10% of A instances get the cue and label B. Clean A and B instances keep their labels.
- **Decoys.** The cue is also pasted on 5% of random background locations without a label change, so the cue alone does not create detections.
- Script to write: `scripts/make_planted_dataset.py`. It writes the COCO-format annotations and the cue boxes (the ground-truth regions).

### 5.4 Fine-tuning

- YOLOX-S from the COCO weights, 10 epochs, low learning rate, using YOLOX's own trainer with a custom `Exp`.
- **Acceptance criteria:**
  - clean AP50 on the dev split drops by less than 2 points;
  - attack success rate (cue-carrying A instances detected as B) is at least 80% on held-out images.
- Estimated about 10–20 GPU-h on the GB10.
- Faster R-CNN and a DETR-family model follow in stage 2.

### 5.5 Evaluation set and causal check

- **Images.** About 500 val2017 images that are **outside** the dev, calib and test splits (1700 are available) and contain an A instance.
- **Pairs.** Each image is rendered twice: cue present and cue absent (inpainted with the neutral fill used by the perturbation explainers).
- **Kept detections.**
  1. On the cue-present image there is a B detection on the A instance (IoU ≥ 0.5).
  2. On the cue-absent image, the B score at that box drops by at least 0.5, or the detection becomes A.
- The rule is fixed before evaluation, and the fraction of kept detections is reported.

### 5.6 Ground-truth score per explanation

- **Cue mass:** the fraction of saliency mass on the cue region. Chance level is the cue's area fraction.
- **Cue pointing game:** whether the saliency maximum falls on the cue.
- **Cue AUC:** pixel-level ROC of saliency for cue pixels vs other pixels within the context window.

### 5.7 Metric validation (to be pre-registered before the evaluation run)

**Methods.** `gcame`, `gcame_raw`, `gradcam`, `gradcam_raw`, `lime`, `dclose`, and the controls (`bbox_prior` at matched widths, `center_prior`, `random`, shuffled maps). D-RISE and ODAM are added in stage 2.

**Metrics under test.** EBPG, PG, OA, box-relative OA, sparsity, the composite, and the E1 control-calibrated variants.

**Analyses.**

- **Method-level agreement.** Kendall τ between the method ranking by each metric and the ranking by cue mass, per cue condition, with an image bootstrap.
- **Detection-level agreement.** Within each method, the Spearman correlation between the metric and cue mass.

**Hypotheses (draft).**

- **E2a.** In the off-box condition, EBPG and PG rank methods against the ground truth (τ < 0).
- **E2b.** Control-calibrated OA agrees with the ground truth better than raw OA (Δτ > 0, CI excluding 0).
- **E2c.** In the in-box off-centre condition, G-CAME's cue mass is below `gcame_raw`'s, because its centre Gaussian suppresses the cue.

### 5.8 Risks

| Risk | Mitigation |
|---|---|
| The detector does not learn the cue | Pilot on 2k images first. Increase the poisoning rate or cue size, recorded as a design change before evaluation |
| The cue is "unnatural" | Two cue types and two class pairs. Keep the natural-data audit (H3, E1) as the main evidence; E2 is the validation |
| The causal filter keeps few detections | Report the kept fraction. Raise the number of evaluation images |
| Compute contention on the shared GB10 | Pilot sizes above. Use the RTX 4090 workstation if one is available |

## 6. Timeline

| Dates | Work |
|---|---|
| Nov 2026 | Finish BDD100K. E3 (CPU or GPU, minutes). E1 (about 26 GPU-h) |
| Dec 2026 – Jan 2027 | E2 pilot on YOLOX-S: dataset script, fine-tune, causal filter, metric validation |
| Feb 2027 | Write the 4-page XAI4CV paper |
| Mar – May 2027 | Stage 2: full E2, more detectors and methods, theory, shift experiments |

## 7. Code added on this branch

| File | Change |
|---|---|
| `src/xai_methods/__init__.py`, `controls.py`, `base.py` | Scaled bbox priors `bbox_prior_sNNN` (`scaled_prior_name`, `is_control_method`). `bbox_prior` is unchanged (= `bbox_prior_s100`) |
| `scripts/run_evidence.py` | Accepts scaled priors as controls |
| `scripts/analyze_evidence.py`, `scripts/analyze_metric_validity.py` | Treat scaled priors as controls |
| `config/evidence_config_e1_priors.yaml`, `scripts/analyze_matched_priors.py` | E1 |
| `scripts/analyze_map_verifier.py` | E3 |
| `cluster/task.sh` | Tasks `evidence-e1priors-DET`, `analyze-e1`, `analyze-e3-DET` |
| `tests/test_v3_evaluation.py` | Tests for scaled priors, E1 and E3 |
