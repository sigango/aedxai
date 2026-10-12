# Pre-registration: confirmatory test run (AED-XAI v2, scenario C)

**Date:** 2026-10-12. **Frozen with:** git tag `v2-freeze-test`. Every run manifest records the commit; it must match the tag and show `dirty: false`.

This document was written after the dev pilot and before any evidence run on the COCO test split or on BDD100K. The claims below were formulated from the dev pilot. The test split and BDD100K have not been used for any choice, so they serve as the confirmation.

## 1. Why the plan changed

The original confirmatory claim was gate **G1**: explanation evidence adds at least 0.02 AUROC for TP-vs-FP verification over a strong uncertainty baseline.

Before the Faster R-CNN pilot we fixed a futility rule. If the dev 95% CI upper bound of ΔAUROC(`S3_explanation` − `S2_baseline`) stayed below 0.02 for Faster R-CNN, as it already had for YOLOX-S, we would drop the G1 superiority claim. The test run would then confirm three things instead: a negative result, a metric audit, and a VLM result ("scenario C").

The rule was triggered:

| Dev pilot (COCO val2017 dev split) | YOLOX-S | Faster R-CNN |
|---|---|---|
| Explained detections / images | 1552 / 296 | 1380 / 150 |
| ΔAUROC `S3_explanation` − `S2_baseline` | +0.0031 [−0.0083, +0.0137] | +0.0044 [−0.0024, +0.0123] |
| ΔAUROC `S3_explanation` − `P_controls_only` | +0.0056 [−0.0036, +0.0156] | −0.0008 [−0.0078, +0.0067] |
| ΔAUROC `S2_plus_vlm_box` − `S2_baseline` | +0.0178 [+0.0058, +0.0293] | +0.0062 [+0.0011, +0.0116] |
| Composite: bbox prior vs best real method | 0.708, ≥ all 6 methods | 0.7313 vs G-CAME 0.7307 |
| OA: bbox prior vs best real method | 0.484, highest | 0.561 vs D-CLOSE 0.485 |
| Composite: `gcame_raw` vs `gradcam_raw` | 0.4438 vs 0.4439 | 0.5241 vs 0.5241 |

## 2. Data

| Dataset | Manifest and split | YOLOX-S | Faster R-CNN |
|---|---|---|---|
| COCO val2017 | `manifests/coco_val2017_v2.json`, `test` (2000 images; disjoint from `dev`, `calib` and the v1 oracle images) | all 2000 | first 500 in manifest order (`NUM_IMAGES_test_frcnn=500`) |
| BDD100K val | `manifests/bdd100k_val_v2.json`, `test` (1000 images) | all 1000 | first 150 (`NUM_IMAGES_bdd_frcnn=150`) |

The BDD100K labels are not on the cluster yet, so the BDD100K manifest does not exist at freeze time. It will be built by `cluster/task.sh setup-bdd-annotations` (det_20 → COCO-80) and `cluster/task.sh setup-bdd-manifest` (`make_manifests.py`, seed 2026, `calib:300,test:1000`). It will be committed, unchanged, before the BDD100K run.

Ground-truth boxes are used only for the TP/FP endpoint, the FP type, AP and the GT-box audit columns. They are never used as features.

## 3. Frozen pipeline

**Detectors**
- `yolox-s` and `fasterrcnn_resnet50_fpn_v2`, with the thresholds in `config/detector_config.yaml`.

**Evidence** (`config/evidence_config_lite.yaml`)
- Identical to the dev config, except for a cap of 10 explained detections per image instead of 20 (the highest-confidence ones).
- Methods: `gcame`, `gcame_raw`, `gradcam`, `gradcam_raw`, `lime`, `dclose` (200 masks).
- Controls: `bbox_prior`, `center_prior`, `random`, plus shuffled `gcame` and `gcame_raw` maps.
- Strong baselines for every detection: pre-NMS candidates, TTA, and Riedlinger-style gradients.

**Metrics**
- `config/eval_config.yaml`, unchanged. The composite is 1/3 EBPG, 1/3 normalised OA and 1/3 sparsity.

**VLM**
- `config/vlm_config.yaml`: Qwen2.5-VL-7B in fp16.
- Box crop, plus salient-pixel crops of `gcame_raw` and `lime`.

**Analysis**
- `scripts/analyze_evidence.py` and `scripts/analyze_metric_validity.py` at the tag, run through `cluster/task.sh analyze-test` and `analyze-bdd`.
- Evidence models: 5-fold image-grouped CV with GBM (plus logistic regression for the main sets).
- Evidence comparisons: paired image bootstrap, 2000 samples, seed 0.
- Audit claims: image bootstrap, 1000 samples, seed 0.

## 4. Primary claims

Each claim is evaluated separately for each detector on each dataset. The rules and margins are stored in `analysis.claims` and `analysis.audit_claims` of the evidence configs. Both analysis reports print a verdict for every claim: SUPPORTED, NOT SUPPORTED or NOT ESTIMABLE.

| ID | Claim | Statistic | Supported if |
|---|---|---|---|
| H1 | Explanation evidence adds no practically relevant TP/FP signal over the strong baseline | ΔAUROC(`S3_explanation` − `S2_baseline`) | 95% CI upper bound < 0.02 |
| H2 | Explanation evidence is no better than evidence from model-free placebo maps | ΔAUROC(`S3_explanation` − `P_controls_only`) | 95% CI upper bound < 0.02 |
| H3a | On the composite, the model-free box prior is not worse than any real method | Mean per-detection `bbox_prior` − m, for each of the 6 methods | CI lower bound > −0.01 for all six |
| H3b | Same as H3a, on OA (insertion − deletion) | as H3a | CI lower bound > −0.01 for all six |
| H3c | G-CAME without its box weighting equals Grad-CAM | Mean per-detection `gcame_raw` − `gradcam_raw` (composite) | CI inside [−0.01, +0.01] |
| H4 | A VLM check of the box crop adds verification signal | ΔAUROC(`S2_plus_vlm_box` − `S2_baseline`) | 95% CI lower bound > 0 |

**How the verdicts are used**
- Headline statements in the paper need support on the COCO test split for both detectors.
- BDD100K results are reported as transfer evidence under the same rules. Where they disagree with COCO, the disagreement is reported and not explained away.
- All verdicts are reported, including the unsupported ones. There is no multiplicity correction: each row is a separately stated claim.

## 5. Secondary and exploratory analyses (reported without verdicts)

**Evidence**
- The original G1 superiority gate (PASS/FAIL) and the lite contrast `S3_lite` − `S2_output_gradient`.
- What the strong baseline is made of (`S1_metadetect`, `S1_tta`, gradients).
- `S3_map_only` and `S3_curves_only`, the shuffled placebo, and the single-method sets.
- AUROC per FP type.
- Subgroups: object size, class frequency, and BDD100K time of day and weather.

**VLM**
- Salient-pixel crops vs the box crop (`S2_plus_vlm_all` − `S2_plus_vlm_box`).
- `S4_explanation_vlm` vs `S3_explanation`.

**Downstream**
- G2 budget-aware acquisition, G3 AP re-scoring, conformal-risk-control abstention, and cost per detection.

**Audit**
- Superiority flags (CI lower bound > 0) for the box prior.
- Deltas that leave out each method's fallback detections.
- OA size trend under the global and box-relative protocols.
- Detector-box vs GT-box ranking, headroom per objective, fallback rates and conditioning deltas.

## 6. Rules

- After the tag, no change to code, configs, feature sets, margins or image caps until every test and BDD100K analysis has run.
  - The only planned commit is the BDD100K manifest (section 2).
  - Unavoidable bug fixes are logged in section 7 with commit and impact, and results before and after the fix are both reported.
- Failed images are retried by the resumable runners. Images that still fail are excluded and counted in the paper.
- As in the dev pilot, crowd matches (`ignore`) are excluded.
- Not run in scenario C: τ calibration, the legacy v1 rows, the cascade, and the selector retraining.
- Runs on the shared GB10 are started with `cluster/launch_unit.sh` (a systemd user unit). Restarted units resume. A restart is not a deviation unless it changes data or settings.

## 7. Deviations log

| Date | Commit | Deviation | Impact |
|---|---|---|---|
| | | (none yet) | |
