# Hướng dẫn chạy v2 trên cluster (YOLOX-S + Faster R-CNN)

Tài liệu này là bản hướng dẫn thực hành cho Part B của `RESULTS_AND_PLAN.md`. Mọi lệnh đều đi qua `cluster/submit.sh` và `cluster/task.sh`, nên chạy từ notebook 07 hay từ terminal đều ra cùng một lệnh.

Các con số GPU-h ở mục 0 lấy từ **số đo thật** trên một DGX Spark (GB10) ở bước smoke ngày 2026-10-01 (mục B5 của `RESULTS_AND_PLAN.md`). Riêng BDD và legacy vẫn là ngoại suy. Kế hoạch đang dùng là **phương án B**: chạy đầy đủ trên COCO cho cả hai detector, chỉ cắt bớt phần chuyển miền (BDD) và các hàng v1 của Faster R-CNN.

---

## 0. Toàn cảnh

```text
[1] Chuẩn bị (login node) ──> [2] env check (GPU node) ──> [3] setup + smoke
      ──> [4] pilot (dev) ──> D1 ──> [5] calib + τ ──> [6] ĐÓNG BĂNG (tag)
      ──> [7] test COCO ──> [8] BDD100K ──> [9] legacy v1 + cascade ──> [10] phân tích (CPU)
```

| Stage | Lệnh | YOLOX (GPU-h) | FRCNN (GPU-h) | Quyết định sau stage |
|---|---|---:|---:|---|
| smoke | `smoke vlm-smoke` | < 1 | < 1 | Go/no-go; đo s/detection |
| pilot | `pilot vlm-pilot analyze-pilot` | 8 | 36 | **D1** |
| calib | `calib calibrate-tau` | 1 | 8 | τ theo từng detector |
| test | `test vlm-test analyze-test` | 38 | 143 | G1–G3 và B (khẳng định) |
| bdd | `bdd vlm-bdd analyze-bdd` | 33 | 71 (300 ảnh) | X (chuyển miền) |
| legacy | `selector legacy`, sau đó `cascade` | 31 | 21 | Chỉ để có hàng so sánh v1 |

- **Tổng phương án B:** khoảng 400 GPU-h, cộng khoảng 10 giờ cho VLM. Trên một GB10 là khoảng **17 ngày chạy liên tục**.
- **Một tiến trình mỗi GPU (`PROCS_PER_GPU=1`).** Khi hai tiến trình Faster R-CNN chạy chung một GB10, mỗi tiến trình chậm đi khoảng 2,1 lần, nên tổng tốc độ không tăng.
- Baseline mạnh (pre-NMS và TTA) tốn thêm 5 lượt forward mỗi ảnh, không đáng kể so với explainer.
- Faster R-CNN là phần tốn nhất: pilot, test, BDD và các baseline LIME / D-CLOSE của legacy.

---

## 1. Chuẩn bị một lần trên login node (có internet)

### 1.1 Mã nguồn

```bash
git clone <repo-url> aed-xai && cd aed-xai
git log -1 --oneline          # phải trùng với commit đã push từ máy local
python -m pip install --user nbstripout && nbstripout --install   # output của notebook không làm bẩn tree
```

### 1.2 Môi trường

**Cách A: container NGC qua Apptainer (khuyến nghị; bắt buộc trên ARM64 / GB10).**
- Để file `.sif` **ngoài repo**, vì `containers/` không nằm trong `.gitignore`, để trong repo sẽ làm bẩn tree.
- `.venv` dùng `--system-site-packages` để giữ nguyên torch của container.

```bash
mkdir -p ~/containers
apptainer pull ~/containers/pytorch_25.11.sif docker://nvcr.io/nvidia/pytorch:25.11-py3
SIF=~/containers/pytorch_25.11.sif
apptainer exec --nv "$SIF" python -m venv --system-site-packages .venv
apptainer exec --nv "$SIF" .venv/bin/python -m pip install --upgrade pip setuptools wheel
apptainer exec --nv "$SIF" .venv/bin/python -m pip install --no-build-isolation -e .
grep -vE '^(torch|torchvision)' requirements.txt > /tmp/req-no-torch.txt
apptainer exec --nv "$SIF" .venv/bin/python -m pip install -r /tmp/req-no-torch.txt
apptainer exec --nv "$SIF" .venv/bin/python -m pip install --no-build-isolation -r requirements-yolox.txt
```

**Cách B: virtualenv**, trên cluster x86 có sẵn module CUDA 12.x. Xem mục "Virtualenv setup" trong `AGENTS.md`. Nhớ cài YOLOX **sau** torch, với `--no-build-isolation`.

### 1.3 Dữ liệu và trọng số

Tải trước ở login node, vì node tính toán thường không có internet.

```bash
bash scripts/download_data.sh    # COCO val2017 (~1 GB), annotations (~241 MB), yolox_s.pth
```

Tải trọng số Faster R-CNN (torchvision) và Qwen2.5-VL-7B (~16 GB) vào cache:

```bash
export HF_HOME=$HOME/.cache/huggingface TORCH_HOME=$HOME/.cache/torch
$RUN .venv/bin/python -c "from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as f, FasterRCNN_ResNet50_FPN_V2_Weights as W; f(weights=W.COCO_V1)"
$RUN .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"
```

Ở đây `RUN` là tiền tố chạy lệnh: `RUN="apptainer exec --nv $SIF"` với cách A, hoặc để trống với cách B.

- **BDD100K** cần đăng ký tài khoản và tải thủ công ảnh `100k/val` cùng nhãn `det_20` (`det_val.json`). Đặt đúng đường dẫn `BDD_*` trong `cluster.env`.
- Nên có BDD **trước khi chạy `setup`**. Như vậy manifest COCO và BDD được tạo một lần (xem cảnh báo ở mục 3.1).

### 1.4 Lấy `results/` của v1 (bắt buộc cho manifest)

`make_manifests.py` đọc `results/xai_selector_training_progress_*.csv` để loại các ảnh oracle của v1 ra khỏi split v2. Chạy lệnh này **trên máy local**:

```bash
rsync -av results/ <user>@<cluster>:<đường-dẫn-clone>/results/
```

### 1.5 `cluster/cluster.env`

```bash
cp cluster/cluster.env.example cluster/cluster.env
```

| Biến | Đặt thế nào |
|---|---|
| `PYTHON` | `.venv/bin/python` |
| `RUN_PREFIX` | Cách A: `"apptainer exec --nv --bind ${PWD} $HOME/containers/pytorch_25.11.sif"`. Cách B: `""` |
| `BACKEND` | `slurm`, hoặc `local` trên workstation / DGX Spark |
| `SLURM_PARTITION`, `SLURM_ACCOUNT`, `SLURM_TIME`, `SLURM_MEM`, `SLURM_GRES` | Theo cluster. `SLURM_MEM=48G` là đủ; VLM fp16 cần GPU ≥ 24 GB |
| `NUM_IMAGES_dev_frcnn=150`, `NUM_IMAGES_test_frcnn=1000`, `NUM_IMAGES_bdd_frcnn=300` | Giữ như kế hoạch. BDD có giới hạn riêng `NUM_IMAGES_bdd_<det>` (phương án B: 300 ảnh cho FRCNN) |
| `LEGACY_METHODS_<det>`, `LEGACY_VARIANTS_<det>`, `CASCADE_DETECTORS` | Phương án B: YOLOX chạy mọi hàng v1; FRCNN chỉ chạy `gradcam gcame` và `v1`; cascade chỉ cho `yolox`. Bỏ trống (unset) một biến để chạy đủ mọi hàng cho detector đó |
| `SHARDS_XDOMAIN` | **16** (mặc định 8 dễ vượt 24 h/shard với FRCNN trên BDD) |
| `LOCAL_GPUS`, `PROCS_PER_GPU` | Chỉ dùng khi `BACKEND=local`. **`PROCS_PER_GPU=1`**: trên GB10 hai tiến trình không nhanh hơn một (đã đo) |

Thêm hai dòng sau vào cuối `cluster.env` để node tính toán không cố truy cập internet:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

> **Lưu ý:** `cluster.env` gán cứng hầu hết các biến, nên đặt biến môi trường trước lệnh (kiểu `SLURM_TIME=72:00:00 cluster/submit.sh …`) **không có tác dụng**. Biến duy nhất ghi đè được theo cách đó là `MAX_EXPLAINED`. Khi cần thiết lập khác cho một stage, hãy tạo file env thứ hai:
> ```bash
> cp cluster/cluster.env cluster/cluster_long.env    # rồi sửa SLURM_TIME, PROCS_PER_GPU, …
> AEDXAI_ENV=cluster/cluster_long.env cluster/submit.sh legacy
> ```
> `.gitignore` đã có `cluster/*.env`, nên mọi file env phụ đều không làm bẩn tree (cần push thay đổi `.gitignore` này lên trước).

---

## 2. Kiểm tra môi trường trên GPU node

```bash
srun --gres=gpu:1 --pty cluster/env_check.sh
```

Phải đạt đủ bốn điều kiện:
- `CUDA available: True`.
- Mọi import đều OK. `bitsandbytes` không bắt buộc, vì config đang dùng `quantization: fp16`.
- Không có dòng `MISSING` nào trong phần data.
- `pytest -m "not slow"` và các test phụ thuộc model (`test_detector.py`, `test_xai_methods.py`) đều pass.

Nếu các test captum / torchvision fail thì **dừng**: đây là những phần chưa kiểm tra được trên laptop (B2).

---

## 3. Setup và smoke

### 3.1 Manifest

```bash
cluster/submit.sh setup
```

Stage này ghi `manifests/coco_val2017_v2.json` (và `bdd100k_val_v2.json` nếu đã có nhãn BDD). Kiểm tra log:
- Các split có đúng số ảnh: dev 300 / calib 1000 / test 2000.
- Có dòng `legacy_oracle_images > 0`, tức ảnh oracle của v1 đã bị loại.

Sau đó commit và push manifest, rồi `git pull` trên máy local:

```bash
git add manifests/ && git commit -m "Add v2 image manifests" && git push
```

> **Không chạy lại `setup` sau khi đã commit.** Lần chạy lại cho ra cùng split (vì dùng seed cố định) nhưng ghi đè `created_at`, làm tree bẩn. Nếu BDD đến muộn, chỉ tạo manifest BDD:
> ```bash
> cluster/task.sh setup-bdd-annotations
> $RUN_PREFIX $PYTHON scripts/make_manifests.py --images-dir "$BDD_IMAGES" --annotations "$BDD_ANN" \
>   --exclude-oracle-csvs "" --splits calib:300,test:1000 --output "$BDD_MANIFEST"
> ```
> Nếu lỡ chạy `setup`, khôi phục bằng `git checkout manifests/coco_val2017_v2.json`.

### 3.2 Smoke: 3 ảnh mỗi detector, config đầy đủ

Nên chạy qua **notebook 06**, vì mục 5 của notebook kiểm tra batch trên model thật, cần kernel có GPU. Nếu chạy bằng terminal:

```bash
cluster/submit.sh smoke vlm-smoke
```

Checklist go/no-go (khớp với cuối notebook 06):
- [ ] Mọi method thật có `is_fallback = False`.
- [ ] Batched forward trả cùng kết quả với batch 1 (notebook 06 mục 5 in `ALL CHECKS PASS`).
- [ ] Median của `vlm__box_answer_mass` ≥ 0.9. Nếu thấp hơn nghĩa là token id Yes/No sai, và feature VLM vô nghĩa.
- [ ] Bảng budget của notebook 06 cho **s/detection thực đo**. Ghi đè bảng ở mục 0 bằng số này.
- [ ] Không bị OOM. Nếu có, giảm `lime.batch_size` / `dclose.batch_size` trong `config/xai_config.yaml` (32 → 8) và `evidence.evaluator_batch_size` (11 → 4). Kết quả không đổi, chỉ tốc độ thay đổi. **Chốt các batch size ngay ở đây**, đừng đổi sau khi đóng băng.

---

## 4. Pilot (COCO dev, config đầy đủ) và quyết định D1

> **Trước pilot:** mọi thay đổi code về feature hoặc baseline phải có sẵn (xem `PAPER_OUTLINE.md` mục 5.2: TTA, pre-NMS, gradient kiểu Riedlinger). Pilot phải đo đúng bộ feature sẽ dùng cho test, nếu không D1 sẽ không có ý nghĩa. Các thay đổi chỉ ở phần phân tích (mục 5.1) có thể thêm sau pilot, nhưng phải trước khi đóng băng.

```bash
git status                      # phải sạch
cluster/submit.sh pilot vlm-pilot analyze-pilot
```

Kết quả: `results/analysis/evidence_coco_dev/report.md`, `summary.json`, `g2_method_value_*.csv` và `results/analysis/metric_validity_coco_dev/`.

**D1**, ghi lại quyết định vào commit message:
1. **Hướng G1:** đọc ΔAUROC(S3 − S2) của từng detector. Dev chỉ có khoảng 1.800–2.000 detection, CI cỡ ±0.02, nên **chỉ xem dấu**; không dùng để chỉnh gate. Nếu ≤ 0 ở cả hai detector, chuyển sang kịch bản C (`PAPER_OUTLINE.md` mục 7).
2. **Danh sách method cho lite config:** chọn trong `g2_method_value_*.csv` (`suggest_lite_methods` ở notebook 07), rồi sửa `config/evidence_config_lite.yaml` (`methods`, `agreement_pairs`, `save_maps_for`).
3. **Số mask của D-CLOSE:** giữ 200 hay dùng 2000 (`dclose_final_masks`).
4. **Budget cho FRCNN**, dựa trên s/detection thực đo:
   ```text
   GPU-h(test FRCNN) ≈ 1000 ảnh × min(10, số det/ảnh) × s/det ÷ 3600
   ```
   Nếu vượt quota, cắt theo thứ tự trong B5:
   1. `MAX_EXPLAINED=5`.
   2. `NUM_IMAGES_test_frcnn=500`.
   3. Bỏ D-CLOSE riêng cho FRCNN. Việc này cần một file config lite riêng cho FRCNN, vì hiện `task.sh` dùng chung một lite config cho cả hai detector (sửa nhỏ, cần làm trước khi đóng băng).
   Vẫn **giữ cả hai detector** trong mọi trường hợp.

```bash
git add config/evidence_config_lite.yaml && git commit -m "D1: freeze lite method list (pilot dev)" && git push
```

---

## 5. Calib và τ

Stage này chỉ phục vụ cascade ở stage legacy.

```bash
cluster/submit.sh calib calibrate-tau
```

- Đọc `results/analysis/tau_calibration.json`.
- Áp dụng bằng notebook 07 mục 2 (`APPLY_TAU = True`), hoặc sửa tay `evaluation.feedback.escalation.tau_by_detector` trong `config/eval_config.yaml` và bỏ chú thích "PROVISIONAL".
- Commit và push.

---

## 6. Đóng băng trước test (đăng ký trước)

1. Commit các file đầu vào sẽ bị khóa:
   - `config/evidence_config_lite.yaml` (bao gồm mục `analysis`: `gate_delta_auroc`, `crc_alpha`, budgets);
   - `config/eval_config.yaml`;
   - `manifests/`;
   - `scripts/analyze_evidence.py` (danh sách feature set và comparison).
2. Gắn tag để trích SHA vào paper:
   ```bash
   git tag -a v2-freeze-test -m "Frozen inputs for the confirmatory test run" && git push --tags
   ```
3. `nbu.config_is_committed()` (notebook 07 mục 3) chỉ kiểm tra lite config, eval config và manifests. **Tự kiểm tra thêm `git status` sạch cho toàn repo.**

Sau bước này, **không sửa config hay code** cho đến khi stage test, BDD và legacy chạy xong.

---

## 7. Test COCO (chạy khẳng định)

```bash
cluster/submit.sh test vlm-test analyze-test
```

- YOLOX xong nhanh. FRCNN là phần tốn thời gian nhất (16 shard × 5–10 h, ước tính).
- Đọc kết quả: `results/analysis/evidence_coco_test/report.md` (G1–G3) và `results/analysis/metric_validity_coco_test/` (gate B).
- Ghi lại verdict ngay, **không chạy lại với ngưỡng khác**.

## 8. BDD100K

```bash
cluster/submit.sh bdd vlm-bdd analyze-bdd     # dùng đúng lite config đã đóng băng
```

Kết quả nằm ở `results/analysis/evidence_bdd_test/` (subgroup ngày/đêm và thời tiết trong `subgroup_auroc_*.csv`).

## 9. Legacy v1 (ưu tiên thấp nhất, chạy cuối)

```bash
AEDXAI_ENV=cluster/cluster_long.env cluster/submit.sh selector legacy    # SLURM_TIME dài, ví dụ 72:00:00
cluster/submit.sh cascade                                               # chỉ sau khi đã áp τ
```

- `run_baseline.py` và `run_experiments.py` **không shard và không resume**. Một job 1000 ảnh bị timeout là mất trắng. Baseline LIME của FRCNN (chưa từng chạy) có thể vượt 24 h.
- Nếu partition không cho job dài: cho riêng các job FRCNN LIME / D-CLOSE chạy với `LEGACY_NUM_IMAGES` nhỏ hơn (qua file env riêng), rồi ghi rõ điều đó trong supplementary.
- Nếu hết thời gian trước deadline, stage này có thể bỏ. Bài chỉ mất các hàng v1 ở supplementary.

---

## 10. Phân tích và hình (CPU, trên laptop)

Chạy trên máy local. Kéo toàn bộ `results/` trừ `maps/`, vì `maps/` khá nặng (vài GB); chỉ kéo `maps/` khi cần hình minh họa định tính:

```bash
rsync -av --exclude='maps/' <user>@<cluster>:<clone>/results/ results/
```

- Mở notebook 08 với `SPLIT = "test"`, sau đó `SPLIT = "bdd"`. Hình được ghi vào `results/figures_v2/<split>/`.
- Lưu trữ `results/` của từng stage kèm manifest (checklist B7).

---

## 11. Theo dõi và xử lý sự cố

| Việc | Lệnh |
|---|---|
| Hàng đợi | `squeue -u $USER` |
| Trạng thái, thời gian, RAM | `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` |
| Log từng shard | `logs/aedxai-<task>_<jobid>_<shard>.out` |
| Tiến độ | notebook 07 (`nbu.evidence_progress`) hoặc `wc -l results/evidence/<run>/records/*.jsonl` |
| Ảnh bị lỗi | `grep -h '"error"' results/evidence/<run>/records/*.jsonl \| head` |
| Chạy lại shard bị preempt hoặc timeout | Chạy lại **cùng stage**. Ảnh đã xong sẽ bị bỏ qua, ảnh lỗi được thử lại |
| Chạy tay một shard | `srun --gres=gpu:1 [--partition … --account …] cluster/task.sh evidence-test-frcnn 5 16` (shard 5 trên 16) |
| Chuỗi `afterok` bị treo do một shard fail | `scancel` các job đang chờ, sửa lỗi, rồi submit lại từ stage bị fail |

**Quy tắc:**
- **Không đổi `SHARDS_*` giữa các lần submit của cùng một run.** Việc chia ảnh vào shard phụ thuộc vào số shard. Dữ liệu không hỏng vì loader lấy bản ghi mới nhất cho mỗi ảnh, nhưng sẽ tốn compute chạy lại.
- **Mọi shard phải có `dirty: false` và cùng một commit.**
  - Chỉ sửa code trên máy local, push lên, rồi `git pull` trên cluster, và chỉ làm vậy *giữa* các stage.
  - Trên cluster chỉ commit ba loại thay đổi: manifest, lite config sau D1, và τ.
- **VLM:** nếu `vlm__box_answer_mass` thấp thì dừng và sửa token id. Trên GPU 24 GB, chạy mỗi GPU một tiến trình VLM.
- **Backend local** (workstation / DGX Spark): `submit.sh` sẽ chặn cho đến khi stage xong. Chạy trong `tmux` hoặc `screen`, log nằm ở `logs/<task>_shard*.log`.

---

## 12. Checklist tổng

- [ ] Env check sạch; mọi test pass
- [ ] Đã commit manifest (COCO và BDD)
- [ ] Smoke: `ALL CHECKS PASS`, answer mass ≥ 0.9, đã chốt batch size, đã ghi s/detection thực đo
- [ ] Đã có code baseline mạnh (5.2) trước pilot
- [ ] Đã ghi D1 vào commit (hướng G1, method lite, D-CLOSE, budget FRCNN)
- [ ] Đã commit τ
- [ ] Đã commit các thay đổi phân tích (5.1); tag `v2-freeze-test`; `git status` sạch
- [ ] Test COCO xong; đã ghi verdict
- [ ] BDD xong
- [ ] Legacy / cascade xong, hoặc đã quyết định bỏ và ghi lý do
- [ ] Đã lưu trữ `results/` kèm manifest; notebook 08 đã sinh hình
