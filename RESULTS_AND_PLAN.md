# AED-XAI — v1 Results and v2 Plan

**Date:** 2026-09-28
**Contents:** Part A records what the v1 system (adaptive per-detection XAI selection) produced and why it failed. Part B is the v2 idea (*explanations as evidence*) and the cluster runbook for it.
All numbers in Part A were recomputed on CPU from `results/` (gitignored; archive it with this file). **Fact** = computed from code or artifacts; **Inference** = reasoned, scope stated; **Hypothesis** = needs a new run.

---

# Part A — v1 results

## A1. What v1 was

Pipeline: detector (YOLOX-S or Faster R-CNN R50-FPN-v2) → Qwen2.5-VL judge → 7-feature MLP selector → one of {GradCAM, G-CAME, D-CLOSE, LIME} → annotation-free evaluator → percentile-gated feedback on detector thresholds.
- **Metrics:** PG; EBPG (with the detector box as pseudo-GT); OA = insertion − deletion AUC (10 global steps, blur baseline); Gini sparsity.
- **Composite:** ⅓·EBPG + ⅓·(OA+1)/2 + ⅓·Gini.
- **Data:** COCO val2017 served both as the oracle pool (549 YOLOX / 461 FRCNN images, 2,000 detections each) and as the headline test set (first 1,000 sorted images). Cross-domain: 200 images × 6 domains, YOLOX only.

Three claims were bundled:
- **C1:** per-detection selection beats any fixed method.
- **C2:** the best method is predictable from pre-explanation features.
- **C3:** explanation quality can drive detector-threshold feedback.

## A2. Headline results (Fact)

Matched per detector, 1,000 COCO images, empty images scored as 0.

| Detector | Method | Composite | PG | EBPG | OA | Gini | 0.25/0.55/0.20 weighting |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOX-S | **G-CAME** | **0.6805** | 0.797 | 0.553 | 0.222 | 0.884 | **0.651** |
| | AED-XAI | 0.5604 | 0.607 | 0.401 | 0.364 | 0.606 | 0.596 |
| | GradCAM | 0.4774 | 0.664 | 0.215 | 0.083 | 0.683 | 0.488 |
| | LIME | 0.4562 | 0.382 | 0.256 | **0.431** | 0.404 | 0.538 |
| | D-CLOSE | 0.3474 | 0.810 | 0.196 | 0.327 | 0.190 | 0.452 |
| FRCNN | **G-CAME** | **0.7307** | 0.830 | 0.619 | 0.272 | 0.939 | **0.692** |
| | AED-XAI | 0.5700 | 0.641 | 0.376 | 0.324 | 0.673 | 0.593 |
| | GradCAM | 0.5479 | 0.589 | 0.214 | 0.175 | 0.844 | 0.545 |
| | D-CLOSE | 0.3451 | 0.767 | 0.145 | **0.419** | 0.182 | 0.463 |
| | LIME | *never run* | | | | | |

- **Paired AED − G-CAME:** YOLOX −0.1200 (95% CI [−0.126, −0.114]), 29 strict wins / 85 ties / 886 losses. FRCNN −0.1607 (CI [−0.167, −0.154]), 15–21 strict wins depending on tie tolerance.
- **Cross-domain:** G-CAME wins in all six domains; AED − G-CAME is −0.121 to −0.139, except VisDrone at −0.020.
- **Oracle headroom** (all 2,000 oracle rows per detector): per-detection oracle minus always-G-CAME is **+0.0030 (YOLOX) / +0.0044 (FRCNN)**. The G-CAME→LIME cascade reaches +0.0030 / +0.0043 at *any* τ. Any method beats G-CAME by more than 0.05 on only 2.25% / 3.35% of detections.
- **Headroom by objective:**

  | Objective | YOLOX | FRCNN |
  |---|---:|---:|
  | 0.25/0.55/0.20 weighting | +0.019 | +0.018 |
  | OA only (raw) | +0.054 | +0.115 |

- **Selector ablation:** beats majority accuracy in 3/15 YOLOX runs and 0/15 FRCNN runs.
- **Feedback iterations:** YOLOX {0: 14, 1: 840, 2: 139, 3: 7}; FRCNN {0: 3, 1: 778, 2: 210, 3: 9}. AED-XAI ends with fewer detections than the baseline on 58 / 117 images, and never with more.
- **Runtime:** reported "runtime" is evaluator time only, ≈0.65 s/detection for every method. Real wall-clock for 200 YOLOX images (cached notebook): G-CAME 18 min, D-CLOSE 79 min, LIME 145 min.

## A3. Why v1 failed (ranked)

1. **The selector followed a shadow objective and routed most detections to LIME (YOLOX) or D-CLOSE/GradCAM (FRCNN).** Confidence: high.
   - The HEAD oracle used per-detection min-max normalisation with weights 0.25·EBPG + 0.55·OA + 0.20·Gini. Its labels (YOLOX: G-CAME 1057 / LIME 608 / D-CLOSE 62 / GradCAM 9) were rebuilt exactly from the progress CSVs.
   - A *perfect* follower of those labels scores 0.6486 vs always-G-CAME's 0.6964 on the evaluator composite.
   - Direct fingerprint: 312/840 single-iteration YOLOX images match fixed LIME exactly on every metric, while GradCAM is never matched, which rules out the rule fallback.
   - The implied method mix (≈50% LIME) accounts for the whole −0.12 gap.
2. **No headroom under the reported composite** (structural; see A2). Any fix can only reach parity.
3. **The metric mostly scores box geometry.** Confidence: medium-high.
   - G-CAME multiplies its CAM by a Gaussian on the detector box, EBPG uses that same box, and Gini rewards concentration.
   - G-CAME's Gini on small objects is 0.991, versus 0.994 for a model-free Gaussian on a 24×24 box.
   - An analytic bbox prior scores ≈0.51–0.66 composite with OA = 0 (Hypothesis until it is run with real OA).
   - OA steps are 10% of the *whole image* (`src/evaluator.py`). For objects under 1% of the image, step 1 already inserts or deletes the entire object, so the OA trend "falls with object size" (G-CAME YOLOX 0.307 small → 0.123 large) may be a metric artifact (Hypothesis).
4. **The feedback loop is an invalid gate.** It compares an image's mean to its own 40th percentile and only removes detections. The gap persists when detection counts are equal (cross-domain COCO −0.133), so it is not the cause of the loss.
5. **Train/inference feature shift.** Scene complexity comes from detector statistics in training (YOLOX: 33.6% "high") but from the VLM at inference (COCO: 1.5% "high"). The VLM is otherwise decorative: quality–confidence r = 0.861, 0/103 FP flags in the cached 20-image sample.
6. **Method settings.** D-CLOSE ran with 200 masks, and its maps are very diffuse (Gini ≈0.19). Failed maps silently became the bbox prior (fallback rate unknown). The LIME kernel is suboptimal but **not degenerate**: ESS 1.76 is driven by the unperturbed row, the ESS over perturbed rows is ≈152, and LIME has the best OA on YOLOX.
7. **Provenance.** The selector checkpoint is absent. `xai_selector_training_data_*.csv` was rewritten on Jul 19 with the new oracle (95.8% G-CAME), after the results were produced, so the old labels must be rebuilt from the progress CSVs. 202 of the 1,000 headline images were also oracle images. There were no run manifests.

## A4. Claims that must not be reused

| Earlier claim | Correct statement |
|---|---|
| "AED-XAI wins head-to-head on 11.4% of images" | 11.4% is wins **or ties**; strict wins are **2.9%** |
| "AED-XAI wins on OA / works against the wrong scoreboard" | LIME (0.431) and D-CLOSE (0.419) have higher OA; AED also loses on the faithfulness-weighted composite |
| "VisDrone is the hint that adaptivity helps" | On VisDrone the selector chose G-CAME for every detection on 60/83 images; it is a fallback to G-CAME |
| "Loss chain 0.698 → 0.654 → 0.560" | Invalid: it mixes cohorts and score scales |
| "Features carry zero signal / zero mutual information" | A grouped-CV regressor predicts the LIME−G-CAME gap with r ≈ 0.41–0.47; the features just never predict *when anything beats* G-CAME (\|corr with regret\| ≤ 0.10) |
| 68.0% agreement; 0.654 vs 0.698; LIME rows 0.497 vs 0.598 | These are on a global-normalised scale; on the evaluator scale they are **66.8%**, **0.6486 vs 0.6964**, **0.4953 vs 0.6064** |
| "OA is stochastic" | The evaluator is deterministic; OA is *coarse*, not stochastic |
| "The cascade is guaranteed to beat any fixed method" / "escalates only 13–19%" | Keep-max is ≥ the default map on the selection metric only; escalation rate ≈ the τ percentile, and 17.5% / 15.8% is the *hit* rate |
| "Image leakage explains the bad selector numbers" | The defect is real, but row-CV and grouped-CV RF accuracy are identical (0.632 / 0.632) |

---

# Part B — v2: explanations as evidence

## B1. Idea and gates

Keep the v1 pipeline, but change **what the per-detection choice optimises**. The question becomes: *does explanation evidence tell us whether a detection is correct, beyond output and gradient uncertainty?* The endpoint is ground truth (TP/FP, AP), which is independent of the metric used for selection. v1 module roles:

| v1 module | v2 role |
|---|---|
| Explainers | Evidence sources: perturbation-curve shape, map structure, cross-explainer agreement |
| VLM | Verifies the *salient pixels* and the box crop (P(yes), no detector confidence in the prompt) |
| Selector | Budget-aware evidence acquisition: which detections get expensive evidence |
| Adaptive threshold / feedback | Per-detection abstention or re-scoring, calibrated with conformal risk control |

Closest prior work:
- MetaDetect (arXiv:2010.01695) and Riedlinger et al., WACV 2023 (arXiv:2107.04517): output and gradient features for TP/FP meta-classification. These are the baselines v2 must beat.
- DetRefiner (arXiv:2605.10190): detection refinement.
- AutoXAI selects explainers per dataset, on tabular data only (Cugny et al., CIKM 2022; El Shawi & Jamel, Sci. Rep. 2025).

No work was found that uses perturbation faithfulness, cross-explainer disagreement, and VLM checks of salient regions for this purpose. That search was quick, so run a proper search (incl. the survey arXiv:2509.01991) before writing a novelty claim.

**Target venue:** CVPR 2027 workshops. SAIAD fits best, with topics "plausibility checks of model outputs", "evaluation of diagnostic techniques" and heatmap-based runtime monitoring. XAI4CV fits too: its 8-page proceedings track is archival, and a 4-page non-proceedings track keeps a later main-conference submission possible. The 2026 proceedings deadlines were in late February; 2027 dates are not yet announced.

| Gate | Question | Pass rule (fixed before the test run) | Produced by |
|---|---|---|---|
| **B** | Does the annotation-free composite rank explanations, or box geometry? | Report only. The bbox prior scores ≥ some real methods, and the OA size trend shrinks under the box-relative protocol. | `analyze_metric_validity.py` |
| **G1** | Does explanation evidence predict TP vs FP beyond output, gradient and augmentation uncertainty? | ΔAUROC(`S3_explanation` − `S2_baseline`) ≥ 0.02 with paired image-bootstrap CI low > 0, on test, for both detectors. `S2_baseline` = post-NMS output + pre-NMS candidates (MetaDetect) + gradients (incl. Riedlinger-style head gradients) + TTA. The placebo (`P_controls_only`) must not explain the gain. Secondary (the original lite gate): `S3_lite` − `S2_output_gradient`. | `analyze_evidence.py` |
| **G2** | Does budget-aware acquisition beat uniform/random at equal compute? | `learned_voi` keeps ≥ 70% of the full-evidence AUROC gain at a 50% budget and beats `random` | `g2_acquisition_*.csv` |
| **G3** | Does re-scoring improve detection? | ΔAP50 vs confidence > 0 with CI low > 0 | `report.md` (G3) |
| **X** | Does it transfer? | G1 passes on BDD100K; day/night and weather subgroup coverage reported | `bdd` stage |

**If G1 fails on the dev pilot**, stop the large runs. Write up gate B plus the headroom result as an evaluation paper (XAI4CV non-proceedings track).

**D1 outcome (2026-10-12): G1 failed on the dev pilot for both detectors, so the plan switched to scenario C.**

- The CI upper bounds were +0.0137 (YOLOX-S) and +0.0123 (Faster R-CNN), both below 0.02. This triggered the futility rule fixed before the Faster R-CNN pilot.
- The test run is still made, but it now confirms the pre-registered claims in `PREREGISTRATION.md`:
  - H1: no practically relevant gain over `S2_baseline`.
  - H2: no gain over the placebo maps.
  - H3: the box prior is not worse than any real method; `gcame_raw` equals `gradcam_raw`.
  - H4: a VLM check of the box crop adds signal.
- Both analysis scripts print a verdict for each claim (`analysis.claims`, `analysis.audit_claims`).
- The calib/τ, legacy, cascade and selector stages are dropped.

The paper-facing metric is unchanged: `composite_weights`, `evaluate_all()` and the composite formula are as in v1. Box-relative OA and GT-box EBPG are *additional* columns from `AutoEvaluator.evaluate_evidence()`.

## B2. Code changes

| Area | Change | Files |
|---|---|---|
| v1 fixes | The cascade is **off by default** and uses `tau_by_detector`. `feedback.fixed_method` gives "always-G-CAME → LIME". Escalations are counted on the returned iteration only. The VLM FP filter persists across iterations. The notebook 04 checkpoint overwrite is disabled. Synthetic figures are opt-in. | `src/feedback_loop.py`, `config/eval_config.yaml`, `notebooks/04_*.ipynb`, `scripts/generate_figures.py` |
| Traceability | Each run writes `manifest.json` (git SHA, dirty-diff hash, config text, image list, checkpoint SHA-256, environment). Fixed disjoint image manifests. Per-detection CSV export. `SaliencyMap.is_fallback`. | `src/provenance.py`, `scripts/make_manifests.py`, `scripts/run_{experiments,baseline}.py`, `src/xai_methods/*` |
| Evaluator | Vectorised steps (bit-identical to the old loop, tested) and optional batched forwards. Returned curves. Box-relative protocol (`oa_box`). GT-box PG/EBPG. | `src/evaluator.py` |
| Speed | LIME/D-CLOSE forwards batched with `batch_size`; maps equal to batch-1 within 1e-5 (tested) | `src/xai_methods/{lime_det,dclose,base}.py` |
| Controls | `gcame_raw` and `gradcam_raw` (bbox conditioning removed), `bbox_prior`, `center_prior`, `random`, pixel-shuffled maps | `src/xai_methods/{__init__,controls}.py` |
| Evidence | COCO-style GT matching with FP type, AP, output/gradient/map/curve/agreement features, VLM P(yes) | `src/gt_matching.py`, `src/evidence.py`, `src/vlm_judge.py` |
| Strong baselines | Pre-NMS candidate statistics (`DetectorWrapper.detect_candidates`, `pre__*`), TTA stability over hflip/shrink/shift/dark (`tta__*`), Riedlinger-style self-learning-loss gradient norms per head parameter group (`grad__rg_*`; classification term only, the box term is zero for the post-NMS box). Computed for every detection except the gradients. | `src/detector.py`, `src/evidence.py`, `scripts/run_evidence.py`, `config/evidence_config*.yaml` |
| Runners | `run_evidence.py` (sharded, resumable), `run_vlm_verification.py` (separate pass, so the VLM never shares memory with the explainers) | `scripts/` |
| Analysis (CPU) | `analyze_evidence.py` (G1–G3, conformal abstention, subgroups, AUROC per FP type, `S3_map_only` / `S3_curves_only` / `S3_no_fallback_flag`, VLM box vs salient-pixel comparisons), `analyze_metric_validity.py` (B + headroom + fallback sensitivity), `calibrate_tau.py` | `scripts/` |
| Cluster | `cluster/{task.sh,submit.sh,env_check.sh,cluster.env.example}` | `cluster/` |

Local validation: `python -m compileall -q src scripts tests` passes. `pytest -m "not slow" tests` gives 76 passed and 5 skipped; the skips are pre-existing (no `torchvision`/`captum` locally, plus one GPU-only timing test). **Check these first on the cluster**, since they could not be run locally:
- YOLOX/Faster R-CNN batched forwards.
- The GradCAM fallback-flag edit (needs captum).
- `VLMJudge.verify_region` (needs the Qwen weights on a GPU).

## B3. One-time cluster setup

```bash
# environment (container-first per AGENTS.md; on ARM64/GB10 keep the NGC torch)
python -m pip install --no-build-isolation -e .
grep -vE '^(torch|torchvision)' requirements.txt > /tmp/req.txt && python -m pip install -r /tmp/req.txt
python -m pip install --no-build-isolation -r requirements-yolox.txt

bash scripts/download_data.sh            # COCO val2017 + annotations + YOLOX-S weights
# BDD100K (optional): 100k val images + det_20 labels into data/bdd100k/... (see cluster.env.example)
# results/ is gitignored: copy it from the old runs (make_manifests.py refuses to run without
# results/xai_selector_training_progress_*.csv, which list the v1 oracle images to exclude), e.g.
#   rsync -av results/ <user>@<cluster>:<clone>/results/        # run on the local machine
python -m pip install nbstripout && nbstripout --install     # notebook outputs never dirty the tree

cp cluster/cluster.env.example cluster/cluster.env   # edit PYTHON / RUN_PREFIX / SLURM_* / paths
srun --gres=gpu:1 --pty cluster/env_check.sh         # GPU, imports, data, fast + model tests

cluster/submit.sh setup                              # manifests (dev 300 / calib 1000 / test 2000)
git add manifests/ && git commit -m "Add v2 image manifests" && git push   # then git pull locally
```

**Git rules.**
- Code changes are made locally, pushed, then `git pull` on the cluster, and only *between* splits.
- Commits made on the cluster are limited to manifests, the lite config after D1, and τ. Push them, then pull locally before editing anything.
- Every shard manifest records the commit and `dirty`, so jobs must start from a clean tree. `cluster/cluster.env`, `data/`, `results/` and `logs/` are gitignored and do not count.

## B4. Stages

Stages given in one `submit.sh` call are chained with `afterok`. A pre-empted shard can simply be resubmitted: the runners skip finished images and retry failed ones.

**On the shared DGX Spark, use `cluster/launch_unit.sh UNIT TASK[:SHARDS] ...` and keep `PROCS_PER_GPU=1`.** It runs the tasks in order through `cluster/run_chain.sh`, inside a systemd user unit with `OOMPolicy=continue`.
- Each task is retried up to 3 times.
- Memory use is logged to `logs/UNIT_mem.log`.
- Do not start long chains from tmux or a Jupyter terminal. The FRCNN pilot died twice that way:
  - 2026-10-06: the terminal session ended.
  - 2026-10-09: `jupyterlab.service` was stopped, which killed every tmux pane started from it.

| # | Command | What runs | Decision after it |
|---|---|---|---|
| 1 | `cluster/submit.sh pilot vlm-pilot analyze-pilot` | COCO dev, full config (6 methods + 3 controls + 2 shuffled). YOLOX on 300 images, FRCNN on 150. | **D1:** G1 direction (not confirmatory). Measure `image_time` and fix the test budget (B5). Choose the lite method list from `g2_method_value_*.csv` and edit `config/evidence_config_lite.yaml`. Decide D-CLOSE 200 vs 2000 masks. |
| 2 | *(not run in scenario C)* `cluster/submit.sh calib calibrate-tau` | COCO calib, G-CAME only; per-detector τ = p30 | τ only served the legacy cascade |
| 3 | `cluster/launch_unit.sh aedxai-test evidence-test-yolox:16 evidence-test-frcnn:16 vlm-test-yolox:4 vlm-test-frcnn:4 analyze-test` | COCO test, test config (`evidence_config_lite.yaml`: all methods and controls, ≤10 explained/image). YOLOX on 2000 images, FRCNN on the first 500. | **H1–H4 verdicts** (confirmatory, `PREREGISTRATION.md`) |
| 4 | `cluster/task.sh setup-bdd-annotations && cluster/task.sh setup-bdd-manifest`, commit the manifest, then `cluster/launch_unit.sh aedxai-bdd evidence-bdd-yolox:16 evidence-bdd-frcnn:16 vlm-bdd-yolox:4 vlm-bdd-frcnn:4 analyze-bdd` | BDD100K test, same config. YOLOX on 1000 images, FRCNN on the first 150. | **X**: the same claims on BDD100K, with day/night and weather subgroups |
| 5 | *(not run in scenario C)* `cluster/submit.sh selector legacy`, then `cluster/submit.sh cascade` | v1 comparison rows | — |

## B5. Compute budget

**Measured** on one DGX Spark (GB10) in the notebook 06 smoke run (3 dev images per detector, full config, one process on the GPU; 2026-10-01):

| Detector | s / explained detection (full) | (lite) | LIME | D-CLOSE | each other map (evaluator only) |
|---|---:|---:|---:|---:|---:|
| YOLOX-S | 13.8 | 11.3 | 6.1 | 3.1 | 0.4–0.6 |
| Faster R-CNN | 84.8 | 68.7 | 38.0 | 17.3 | 3.0–3.7 |

Two processes on one GB10 halve the speed of each, so keep `PROCS_PER_GPU=1`. The pre-NMS and TTA baselines add 5 forwards per image (cheap). Projection for plan B (BDD and legacy are extrapolated, not measured):

| Stage | YOLOX (GPU-h) | FRCNN (GPU-h) |
|---|---:|---:|
| Pilot (dev, full; 300 / 150 images) | 8 | 36 |
| Calib (G-CAME only, 1000 images) | 1 | 8 |
| Test (lite, ≤10 explained/image; 2000 / 1000 images) | 38 | 143 |
| BDD100K (lite; 1000 / 300 images) | 33 | 71 |
| Legacy (plan B rows) | 31 | 21 |
| VLM passes (all) | ~10 (both) | |
| **Total** | **~400 GPU-h ≈ 17 days on one GB10** | |

**Scenario C budget (from 2026-10-12).** The test config is the full config with ≤10 explained/image. The full config costs about 1.2× the old lite config.

| Stage | YOLOX (GPU-h) | FRCNN (GPU-h) |
|---|---:|---:|
| Test (2000 / 500 images) | ~46 | ~88 |
| BDD100K (1000 / 150 images) | ~40 | ~44 |
| VLM passes (all) | ~8 (both) | |
| **Total** | **~225 GPU-h ≈ 9.5 days on one GB10** | |

The FRCNN dev pilot measured 735–1050 s per image at 9.2 explained detections per image (≤20). The cost of the expensive tier was 72 s per detection, against 0.2 s for the strong baseline.

The plan-B projection above and the full plan (every legacy row, FRCNN on all 1000 BDD images, ≈800 GPU-h) are kept for reference. GPU-h = images × explained-per-image × measured s/detection / 3600. If FRCNN still does not fit, reduce in this order:
1. `MAX_EXPLAINED=5`.
2. `NUM_IMAGES_test_frcnn=500`.
3. Drop D-CLOSE for FRCNN only.

Keep YOLOX at full size, since it carries the headline.

**Smoke-run finding for gate B:** G-CAME fell back to the bbox prior on 22.6% of the Faster R-CNN detections (`gcame_raw` 0%, LIME 3.2%, YOLOX 0% for every method). These figures come from 3 images.

The dev pilot measured lower rates:

| Method | Faster R-CNN | YOLOX-S |
|---|---:|---:|
| G-CAME | 5.65% | 1.2% |
| LIME | 1.3% | |

On Faster R-CNN, the fallback detections have a higher TP rate (0.62 vs 0.51). Their composite is that of the bbox prior (0.78), which lifts G-CAME's composite from 0.7277 (non-fallback detections) to 0.7307. The CAM at `layer4` has no positive mass near the box centre for those detections, so the Gaussian weighting zeroes the map. The v1 Faster R-CNN G-CAME composite (0.7307) may therefore partly be a bbox-prior score. `analyze_metric_validity.py` reports `fallback_*.csv`, and `S3_no_fallback_flag` checks that G1 does not rest on the flag.

## B6. Outputs → paper

| Paper element | File |
|---|---|
| Metric audit (controls vs methods, OA vs OA_box by size, detector-box vs GT-box ranking) | `results/analysis/metric_validity_coco_test/{methods,by_size,size_trend,conditioning}_*.csv` |
| Headroom per objective (motivation, with A2) | `…/headroom_*.csv` |
| TP/FP verification, feature-set ablation incl. placebo and single-method rows | `results/analysis/evidence_coco_test/g1_metrics_*.csv`, `summary.json` |
| AUROC vs compute (acquisition policies) | `g2_acquisition_*.csv` |
| AP re-scoring | `summary.json` → `g3` |
| Coverage vs risk, subgroup coverage | `abstention_crc_*.csv`, `subgroup_auroc_*.csv` |
| Cross-domain | `results/analysis/evidence_bdd_test/` |
| v1 comparison rows | `results/v2_legacy/` |

## B7. Reproducibility checklist and risks

- [ ] `manifests/*.json` committed before stage 1; the test split is never used for model or threshold choices.
- [x] ~~`tau_by_detector` replaced with the stage-2 output before stage 5.~~ Not needed in scenario C.
- [ ] Test config, claims and image caps frozen with tag `v2-freeze-test` (`PREREGISTRATION.md`) before stage 3.
- [ ] BDD100K manifest committed before stage 4; nothing else changes after the tag.
- [ ] Every shard manifest shows `dirty: false` and the same commit.
- [ ] `results/` archived per stage together with its manifests.

Risks:
- **Circularity is reduced, not removed.** Labels come from GT, but the features come from the same detector. The placebo and single-method rows show *which* evidence carries the signal.
- **The gradient baseline is Riedlinger-style, not their full feature set.** `grad__rg_*` are norms of the self-learning classification loss gradient per head parameter group; the box-regression term is omitted (zero for the post-NMS box). State this in the paper.
- **The legacy selector still uses row-level splits.** It is a comparison row only; the relabelled oracle is 95% G-CAME, so expect a constant selector.
- **VLM token ids.** On the first shard, check that `vlm__box_answer_mass` ≈ 1. A low value means the Yes/No token ids are wrong for the tokenizer.
- **OOM:** lower `lime.batch_size` / `dclose.batch_size` (32 → 8 for FRCNN) and `evidence.evaluator_batch_size` (11 → 4). Values do not change, only speed.
- **bitsandbytes on ARM64/GB10:** keep `vlm.quantization: fp16`.
- **Shared GB10:** other users' jobs filled the RAM at least once (OOM, 2026-10-11). Start chains with `cluster/launch_unit.sh` and watch `logs/UNIT_mem.log`. A steady rise of `pipeline_rss` would point to a leak in our own runner.

## B8. Timeline

| Weeks | Work |
|---|---|
| 1–2 | Setup, env check, stage 1 pilot, D1 (done 2026-10-12: scenario C) |
| 3 | Freeze (`v2-freeze-test`), stage 3 COCO test (~6 days) |
| 4 | Stage 4 BDD100K (~4 days), analyses, figures |
| 5–8 | Writing (scenario C outline, `PAPER_OUTLINE.md` §7) and related-work search |

## B9. Notebook workflow

The three v2 notebooks are thin front-ends over `cluster/task.sh` and `cluster/submit.sh`, so a notebook run and a cluster run execute the same commands. Notebooks 04 and 05 are v1 (legacy); use them only to reproduce the v1 rows.

| Notebook | Where to run it | What it does | Decision it produces |
|---|---|---|---|
| `06_v2_setup_and_smoke.ipynb` | GPU node, once per machine | Env check and tests, data check, manifests, 3-image smoke run per detector, real-model checks (batched forwards, captum, no fallback), VLM smoke (`box_answer_mass`), map overlays, measured GPU budget | Go/no-go for the pilot; commit `manifests/` |
| `07_v2_run_experiments.ipynb` | Login node (SLURM) or workstation (`BACKEND="local"`) | Stages 1–5 with progress tables | D1 (G1 direction, lite methods, budget); τ via `APPLY_TAU`; freeze check before test |
| `08_v2_analysis_and_figures.ipynb` | Any CPU machine with `results/` | Gate verdicts, metric audit (B), G1 forest, G2 curves, G3 AP, abstention, subgroups, qualitative cases, v1 rows | Paper tables and figures in `results/figures_v2/<split>/` |

Order and rules:
1. **06:** finish its checklist before submitting anything, and commit `manifests/*.json`.
2. **07 §1 (pilot):** run `08` with `SPLIT = "dev"` for an exploratory look. Pick the lite method list (`suggest_lite_methods`), edit `config/evidence_config_lite.yaml`, commit.
3. **07 §2 (τ):** set `APPLY_TAU = True` once, update the PROVISIONAL comment in `eval_config.yaml`, commit.
4. **07 §3 (test):** submission is refused while frozen inputs are uncommitted (`config_is_committed`). Do not change configs until stages 3–5 finish.
5. **08 with `SPLIT = "test"`:** the confirmatory figures. Then `SPLIT = "bdd"` for transfer.
6. Commit notebooks **without outputs**. Figures and tables live under `results/` (archived, not committed).

Kernel notes:
- The kernel's Python must see `src/` and the same packages as the cluster jobs. If jobs run in a container (`RUN_PREFIX`), only notebook 06 §5 needs a GPU kernel with torch, yolox and captum; every other cell shells out through `task.sh`.
- `DRY_RUN = True` prints commands without running them. It is useful for checking a new `cluster.env`.
- Every stage is resumable. If a notebook dies mid-stage, re-run the same cell.

## References (collected during the reviews)

- ODExAI: Nguyen, Nguyen & Cao, arXiv:2504.19249 (2025). The fixed-method benchmark on the same detectors.
- D-CLOSE: Truong et al., "Towards Better Explanations for Object Detection", ACML 2023, PMLR 222.
- G-CAME: Nguyen et al., arXiv:2306.03400 (2023); extended version arXiv:2404.13417.
- AutoXAI: Cugny et al., CIKM 2022; El Shawi & Jamel, Sci. Rep. 2025 (per dataset, tabular data).
- MetaDetect: Schubert et al., arXiv:2010.01695. Gradient uncertainty: Riedlinger et al., WACV 2023, arXiv:2107.04517. DetRefiner: arXiv:2605.10190.
- LATEC: Klein et al., NeurIPS 2024 D&B (conflicting XAI metric rankings, classification).
- Explanation-fidelity subgroup audit: Balagopalan et al., FAccT 2022.
- Survey: "Explaining What Machines See", arXiv:2509.01991.
- Supervisor-provided: TrustMatch (He & Hong, ICCVW 2025); SemiXAI-Net (Tahir et al., PeerJ CS 2026).
