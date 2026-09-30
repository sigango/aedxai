# AED-XAI v2 — Đánh giá khả thi và dàn bài nộp paper

**Ngày:** 2026-09-29 · **Dựa trên:** `RESULTS_AND_PLAN.md` (Part A/B), code v2 (`src/evidence.py`, `scripts/analyze_evidence.py`, `scripts/analyze_metric_validity.py`, `src/gt_matching.py`, `config/evidence_config*.yaml`) và các CSV v1 trong `results/`.

Quy ước giống `RESULTS_AND_PLAN.md`: **Fact** là số tính được từ artifact. **Ước tính** là xác suất chủ quan kèm lý do, cần cập nhật ngay sau pilot dev (stage 1). Chưa có run v2 nào, nên mọi xác suất bên dưới đều là Ước tính.

---

## 1. Kết luận ngắn

| Câu hỏi | Trả lời |
|---|---|
| Ý tưởng v2 có khả thi về kỹ thuật và compute không? | **Có.** Code đã xong, có test, có runbook. Tổng khoảng 200–350 GPU-h, trong đó Faster R-CNN tốn nhất. Kịp trong 6–8 tuần nếu có cluster. |
| Có ra được *một* paper không? | **Khả năng cao, khoảng 85–90%.** Nhánh dự phòng (metric audit cộng kết quả âm có CI) gần như chắc chắn viết được, vì bằng chứng v1 đã rất mạnh. |
| Có ra được paper *đúng câu chuyện v2* ("explanation là bằng chứng để kiểm chứng detection") không? | **Khoảng 50–55%** qua được G1 so với baseline hiện tại. **Chỉ khoảng 20–30%** qua được G1 *và* chứng minh tín hiệu đến từ nội dung explanation, không phải từ perturbation probing hay VLM. |
| Xác suất được nhận ở workshop CVPR 2027 (SAIAD / XAI4CV) ngay lần nộp đầu? | **Khoảng 50–60%**, tính trung bình trên các kịch bản ở mục 4. |
| Xác suất lên main conference (ICCV 2027 / WACV 2028)? | **Khoảng 10–20%.** Chỉ đáng thử ở kịch bản A, và cần thêm baseline mạnh cùng một detector hiện đại. CVPR 2027 main (hạn 16/11/2026) không kịp. |
| Có bị trùng ý tưởng không? | **Chưa thấy bài nào làm đúng việc này cho detection 2D** (mục 8). Tiền lệ gần nhất là Spagnolo 2024, cho tổn thương MS trong ảnh y tế. Không được tuyên bố "đầu tiên dùng explanation để phát hiện lỗi". |

**Rủi ro lớn nhất là baseline còn yếu, không phải compute.** Baseline S2 hiện tại chỉ là "MetaDetect-lite" (không có thống kê candidate trước NMS) cộng 5 chuẩn gradient ở một layer. S3 lại cộng thêm khoảng 1.000 forward perturbation cho mỗi detection. Reviewer sẽ hỏi ngay: *"Có phải chỉ là test-time perturbation không? Dùng cùng lượng compute cho TTA thì sao?"* Nếu chưa trả lời được câu này trước khi chạy test, G1 có pass cũng khó bảo vệ. Chi tiết ở mục 5.

---

## 2. Bằng chứng đang có

### 2.1 Từ `RESULTS_AND_PLAN.md` (Fact)

- Per-detection selection gần như không có headroom dưới composite: oracle trừ always-G-CAME chỉ được **+0.0030 (YOLOX) / +0.0044 (FRCNN)**. Câu chuyện v1 (C1) vì thế không thể cứu được.
- G-CAME thắng ở mọi cấu hình. Gini của G-CAME trên vật thể nhỏ là 0.991, còn Gaussian không dùng model trên box 24×24 đạt 0.994.
- VLM trong v1 gần như chỉ để trang trí: tương quan quality–confidence r = 0.861, và 0/103 lần đánh dấu FP.
- Các đặc trưng trước explanation có dự đoán được *độ chênh* LIME − G-CAME (r ≈ 0.41–0.47). Nhưng chúng không dự đoán được *khi nào* có method thắng G-CAME.

### 2.2 Kiểm tra bổ sung hôm nay (Fact, CPU, trên `results/xai_selector_training_progress_*.csv`, 2.000 detection mỗi detector, GroupKFold 5 theo ảnh, GBM)

**(a) EBPG gần như là hàm của hình học box.** Chỉ dùng hai biến log(diện tích box) và tỉ lệ cạnh:

| R² dự đoán từ hình học box | GradCAM EBPG | G-CAME EBPG | D-CLOSE EBPG | LIME EBPG | G-CAME Gini |
|---|---:|---:|---:|---:|---:|
| YOLOX-S | 0.869 | 0.370 | **0.954** | **0.932** | 0.799 |
| FRCNN | 0.695 | 0.240 | **0.951** | 0.793 | 0.658 |

Như vậy EBPG của các method có map lan rộng chủ yếu đo "box chiếm bao nhiêu phần ảnh", và Gini của G-CAME chủ yếu đo kích thước box. Kết quả này củng cố mạnh cho gate B và cho nhánh dự phòng.

**(b) OA và PG mang phương sai *độc lập* với đặc trưng output.** R² dự đoán từ 8 đặc trưng output (confidence, class, size, số detection, aspect, entropy, log area):

| Metric | GradCAM | G-CAME | D-CLOSE | LIME |
|---|---:|---:|---:|---:|
| OA, YOLOX | ≈0 | 0.09 | 0.45 | 0.54 |
| OA, FRCNN | 0.08 | 0.14 | 0.53 | 0.42 |
| PG, YOLOX | ≈0 | 0.02 | ≈0 | 0.29 |

Nghĩa là explanation *có* chứa thông tin mà output không có. Tuy nhiên chưa có nhãn GT nên **chưa biết phần thông tin đó liên quan đến TP/FP hay chỉ là nhiễu**. Đây là điều kiện cần cho G1, chưa phải điều kiện đủ.

> Pilot G1 rẻ nhất có thể làm ngay: ghép `sample_key` (image_id:class:box) của 4.000 detection v1 với `instances_val2017.json`, rồi so AUROC(output) với AUROC(output + 16 metric explanation). Chỉ cần CPU, khoảng 1 giờ, cần file annotation COCO (241 MB). Nên làm trước khi chạy pilot GPU.

---

## 3. Đánh giá từng gate

Xác suất là **Ước tính** cho run test đã đóng băng, với baseline như code hiện tại. Nếu nâng baseline như mục 5 thì P(pass) giảm, nhưng giá trị của kết quả khi pass tăng mạnh.

| Gate | Tiêu chí | P(pass) | Lý do chính |
|---|---|---:|---|
| **B**: metric audit | bbox prior ≥ một số method thật; xu hướng OA theo size co lại dưới box protocol | **~95%** / **~65%** | Composite của D-CLOSE (0.347) và GradCAM (0.477) đã thấp hơn mức ước tính cho bbox prior. Hai bảng R² ở 2.2 càng củng cố. Vế thứ hai (OA theo size) không chắc: box protocol đổi cả thang đo. |
| **G1** | ΔAUROC(S3 − S2) ≥ 0.02, CI > 0, cả hai detector | **~50–55%** | *Ủng hộ:* S3 có 10 bước insertion/deletion cho 4–6 map (cộng box protocol), tức confidence dưới khoảng 100 occlusion có cấu trúc, một tín hiệu độ bền rất giàu; S2 lại yếu. *Chống:* ở ngưỡng conf 0.25, phần lớn FP trên COCO là lỗi localization và duplicate (IoU 0.1–0.5), vốn khó thấy qua saliency. FRCNN có confidence gần như hai cực (median 0.94), nên AUROC nền đã cao. |
| G1, quy nguồn | S3 − P_controls_only > 0 có ý nghĩa | **~35–40%** | Placebo `bbox_prior`/`random` *vẫn* chạy đường cong perturbation qua model. Nó bắt được phần "probing", nên khoảng cách còn lại chỉ là đóng góp của nội dung map. |
| VLM | S2 + VLM_box − S2 ≥ 0.02 | **~70–80%** | Qwen2.5-VL-7B phân loại crop các lớp COCO tốt, đặc biệt với FP background và nhầm lớp. |
| VLM saliency | S4 − (S2 + VLM_box) ≥ 0.01 | **~25–30%** | Phần crop vùng salient phải thêm được thông tin so với crop box. **Code hiện chưa so sánh cặp này** (xem 5.1). |
| **G2** | learned_voi giữ ≥ 70% gain ở budget 50% và thắng random | ~70% nếu G1 pass | Lấy mẫu theo độ bất định vốn hiệu quả, vì gain tập trung ở vùng p ≈ 0.5. |
| **G3** | ΔAP50 > 0, CI > 0 | ~55–60% nếu G1 pass | MetaDetect từng cho gain AP nhỏ. Ở ngưỡng 0.25 và chỉ re-score top-10 mỗi ảnh thì gain sẽ nhỏ, và có thể do VLM gánh. |
| **X**: BDD100K | G1 pass trong miền BDD | ~65% nếu G1 COCO pass | Mỗi miền huấn luyện và CV riêng. Domain shift làm FP tăng, nên càng dễ có tín hiệu. |

**Độ nhạy thống kê (Ước tính):** test YOLOX có khoảng 12k detection, khoảng 40% là FP, nên CI bootstrap của ΔAUROC rộng cỡ ±0.006–0.010; ngưỡng 0.02 phát hiện được. Pilot dev có khoảng 1.800 detection, CI cỡ ±0.02, **chỉ đủ để xem hướng**. Đúng như kế hoạch: dev không dùng để khẳng định.

---

## 4. Kịch bản → loại paper → venue

| Kịch bản | Điều kiện | P (Ước tính) | Paper | Venue gợi ý | P(nhận) nếu làm tốt |
|---|---|---:|---|---|---:|
| **A**: dương tính mạnh | G1 pass cả hai detector, S3 > placebo, cộng G2 hoặc G3 | 20–30% | Dàn bài chính (mục 6), khung "explanation evidence" | SAIAD / XAI4CV 2027 (8 trang, archival). Nếu thêm detector hiện đại và baseline mạnh: ICCV 2027 | Workshop ~70%; main ~25% |
| **B**: dương tính nhưng tín hiệu từ probing/VLM | G1 pass, nhưng placebo hoặc VLM_box giải thích phần lớn gain | ~30% | Dàn bài chính, đổi khung câu hỏi thành *"Explanation evidence mang tín hiệu gì: nội dung hay probing?"*; phân rã nguồn là đóng góp chính | SAIAD 2027 (chủ đề "plausibility checks", "runtime monitoring") | ~50–60% |
| **C**: âm tính | G1 fail | 40–45% | Dàn bài dự phòng (mục 7): paper đánh giá / kết quả âm | XAI4CV 2027 (track 4 trang không archival, hoặc 8 trang archival) | ~45–55% |

**Kỳ vọng P(được nhận ở lần nộp workshop đầu tiên) ≈ 0.25·0.70 + 0.30·0.55 + 0.45·0.50 ≈ 0.57.** Nếu tính cả lần nộp lại ở venue thứ hai (workshop khác hoặc BMVC 2027), xác suất cộng dồn khoảng 75–80%.

---

## 5. Việc cần làm trước khi chạy test (stage 3)

Các mục dưới đây phải đóng băng *trước* test split, vì gate và feature set được cố định từ lúc đó. Xếp theo tỉ lệ giá trị trên chi phí.

### 5.1 Chỉ sửa phần phân tích (CPU, khoảng nửa ngày)

1. **Thêm so sánh `S4_explanation_vlm − S2_plus_vlm_box`** và một set `S2_plus_vlm_all` (VLM box + VLM saliency, không có metric explanation). Nếu thiếu, không thể tách đóng góp của "VLM nhìn vùng salient" khỏi "VLM nhìn box". Đây là điểm mới đã nêu trong B1 nên bắt buộc phải có bằng chứng. `comparisons` trong `scripts/analyze_evidence.py` hiện chưa có cặp này.
2. **Tách S3 theo chi phí và cơ chế:**
   - `S3_map_only`: chỉ `mf__` và `ag__` của gcame_raw / gradcam_raw. Set này **không cần thêm forward perturbation** nào ngoài 1 lần backward. Nếu nó thắng S2, đó là bằng chứng *sạch nhất* rằng nội dung explanation có ích, không bị nhiễu bởi probing.
   - `S3_curves_only`: chỉ `cf__`, `cfb__`, `oa`, `insertion`, `deletion`.
   - Cộng với `P_controls_only` đã có, ta được phân rã ba vế **probing bằng mask không dùng model / probing bằng mask từ model / cấu trúc map**. Bảng này là trung tâm của kịch bản B.
3. **AUROC theo loại FP** (background / classification / localization / duplicate so với TP). `gt_matching.py` đã gán `fp_type`, nhưng `analyze_evidence.py` chưa báo cáo theo loại. Dự đoán: explanation giúp với background và classification, không giúp với localization. Đây là một hình rất có giá trị để viết bài.
4. **Mục tiêu phụ: hồi quy IoU** (R² của IoU với GT), giống MetaDetect. Làm vậy tránh việc kết quả phụ thuộc cứng vào ngưỡng 0.5.
5. **Kiểm soát dung lượng mô hình:** S3 có khoảng 100+ feature, S2 khoảng 17. Báo cáo thêm logreg (đã có) và một "S3-compact" (top-k feature chọn trên dev), để chứng minh gain không đến từ việc có nhiều feature hơn.

### 5.2 Nâng baseline (GPU, cần có trước pilot để pilot còn ý nghĩa)

6. **Baseline TTA cân compute** (flip ngang, 2 scale, dịch nhỏ; ghép theo IoU; lấy mean/std của score và tọa độ box). Chỉ tốn 3–5 forward *mỗi ảnh*. Đây là câu hỏi reviewer chắc chắn đặt ra. Nếu TTA đạt ngang S3 với chi phí rẻ hơn khoảng 100 lần thì kết luận phải đổi, và biết điều đó từ pilot vẫn tốt hơn biết lúc bị review.
7. **MetaDetect thật: thống kê candidate trước NMS** (số candidate có IoU > 0.5 với box giữ lại, mean/std score, std tọa độ). Với YOLOX dễ lấy từ `raw_outputs` trước `postprocess`. Với FRCNN cần hook `roi_heads`. Đây là nhóm feature mạnh nhất của MetaDetect; thiếu nó thì câu "beyond MetaDetect" không đứng được.
8. **Gradient uncertainty kiểu Riedlinger:** gradient của loss tự huấn luyện (lấy chính dự đoán làm nhãn giả) theo tham số của nhánh classification và box ở head, lấy ở ≥ 2 layer. Code hiện dùng gradient của score ở một layer backbone. Nếu không kịp làm thì phải ghi rõ "gradient-lite" trong paper, như B7 đã nêu.
9. **Đăng ký trước baseline chính:** ghi trong commit đóng băng rằng gate G1 so với S2 *mạnh* (có 6–8), còn so với S2-lite là phân tích phụ. Điều này làm P(pass) giảm khoảng 10–15 điểm, nhưng một bài qua được G1 kiểu đó mới bảo vệ được trước reviewer.

### 5.3 Tùy chọn, chỉ cho hướng main conference

10. Thêm một detector hiện đại (RT-DETR hoặc YOLOv8/YOLO11). YOLOX-S và FRCNN-R50 đều từ 2021, và reviewer main 2027 sẽ hỏi về kiến trúc transformer.
11. Hệ số chi phí: báo cáo AUROC theo giây/detection (đường G2) so với cả TTA lẫn "chỉ dùng detector lớn hơn". Cần để trả lời câu hỏi "có đáng bỏ ra khoảng 1.000 forward cho mỗi detection không".

### 5.4 Thứ tự đề xuất

1. Làm pilot G1 trên CPU với dữ liệu v1 (mục 2.2; cần file annotation).
2. Làm 5.1 và mục 6–7 cho YOLOX.
3. Chạy pilot dev YOLOX (khoảng 1–3 GPU-h), rồi ra quyết định **D1**.
4. Nếu hướng G1 dương tính: làm 7–8 cho FRCNN, chạy pilot FRCNN, rồi calib, rồi test.
5. Nếu hướng G1 âm tính rõ (ΔAUROC dev ≤ 0 so với S2+TTA): **dừng các run lớn**, chạy metric audit trên test với lite config chỉ gồm method và control (rẻ hơn nhiều), và viết theo mục 7.

---

## 6. Dàn bài chính (kịch bản A/B)

**Định dạng:** CVPR workshop, 8 trang không kể tài liệu tham khảo, double-blind, có supplementary.

**Tên bài (chọn một):**
- *Do Explanations Know When a Detector Is Wrong? Explanation Evidence for Verifying Object Detections*
- *Explanations as Evidence: Verifying Object Detections with Perturbation Faithfulness, Explainer Agreement, and VLM Checks*
- (kịch bản B) *What Do Explanations Add to Detection Verification? Separating Content from Probing*

**Mạch câu chuyện (một câu):** metric không cần nhãn chấm explanation bằng chính box của detector, nên đo hình học box nhiều hơn đo tính trung thực. Vì vậy ta chấm explanation bằng một endpoint độc lập: *explanation có cho biết detection đúng hay sai không?*

### Abstract (khoảng 200 từ, 5 câu)
1. Bối cảnh: XAI cho detection thường được đánh giá bằng metric không cần nhãn (EBPG theo box của detector, insertion/deletion, sparsity).
2. Phát hiện 1: các metric này chủ yếu thưởng cho hình học box. Một Gaussian không dùng model xếp hạng [X/N]; EBPG có R² lên đến 0.95 chỉ từ kích thước box; per-detection selection chỉ có ≤ 0.5% headroom.
3. Đề xuất: đánh giá explanation như *bằng chứng* cho tính đúng của detection, gồm 4 họ đặc trưng (map, curve, agreement, VLM trên vùng salient) và control placebo để quy nguồn tín hiệu.
4. Kết quả: trên YOLOX-S và Faster R-CNN, COCO và BDD100K, [ΔAUROC = … so với MetaDetect + gradient + TTA], tập trung ở FP loại [background/classification]; acquisition theo budget giữ [..%] gain ở 50% compute; CRC giữ tỉ lệ FP được chấp nhận ≤ α.
5. Ý nghĩa, và cam kết công bố code, manifest, phân tích đăng ký trước.

### 1. Introduction (khoảng 1 trang)
- Đoạn 1: detector triển khai trong hệ thống an toàn cần kiểm tra tính hợp lý của output (framing SAIAD). Explanation thường chỉ được *nhìn*, hiếm khi được *dùng*.
- Đoạn 2: khoảng trống. Meta-classification (MetaDetect, gradient uncertainty) chỉ dùng output và gradient. XAI cho detection lại được đánh giá bằng metric tự tham chiếu.
- Đoạn 3: bài học v1, viết ngắn gọn và trung thực: ta đã thử chọn explanation theo từng detection, nhưng headroom ≈ 0 và metric thưởng cho hình học. Từ đó chuyển câu hỏi.
- Đoạn 4: câu hỏi nghiên cứu và thiết kế: endpoint là GT, gate đăng ký trước, placebo.
- **Đóng góp (3 gạch đầu dòng):**
  1. Audit metric không cần nhãn cho XAI detection với control không dùng model, protocol box-relative, và phân tích headroom.
  2. Khung *explanation evidence* với phân rã nguồn tín hiệu (nội dung / probing / VLM).
  3. Đánh giá đăng ký trước trên 2 detector × 2 miền: verification, acquisition theo budget, CRC abstention, AP re-scoring.
- **Hình 1 (teaser):** một FP có confidence cao bị loại và một TP confidence thấp được giữ lại, kèm map, đường deletion và P(yes) của VLM. Lấy từ bộ "confident FP rejected" / "weak TP rescued" ở notebook 08.

> **Bản nháp đã viết:** Introduction và Related Work ở `paper/sections/1_intro.tex` và `paper/sections/2_related.tex`. Tài liệu tham khảo ở `paper/refs.bib`, gồm 58 mục đã đối chiếu với arXiv, DBLP và CVF. Số màu xanh (`\res{}`) là số v1 hoặc chỗ trống, phải thay bằng số của test v2.

### 2. Related Work (khoảng 0.6 trang)
- **XAI cho detection:** GradCAM, G-CAME, D-CLOSE, LIME/D-RISE; benchmark ODExAI.
- **Đánh giá XAI và pitfall của metric:** sanity checks, ROAD, LATEC, Quantus/MetaQuantus. Nêu rõ khác biệt: bài này làm cho *detection*, và dùng pseudo-GT box.
- **Meta-classification / uncertainty cho detection:** MetaDetect, gradient uncertainty (Riedlinger), TTA/ensemble, DetRefiner.
- **Explanation dùng để phát hiện lỗi / OOD** (chủ yếu ở classification) và **VLM làm verifier**.
- Kết đoạn: điểm mới *phòng thủ được*. Chỉnh theo kết quả tra cứu ở mục 8.

### 3. Explanation Metrics Score the Box (khoảng 1 trang; gate B)
- 3.1 Setup: các metric, composite, detector box làm pseudo-GT; công thức ngắn.
- 3.2 Control: `bbox_prior`, `center_prior`, `random`, pixel-shuffled; `*_raw` (bỏ điều kiện box).
- 3.3 Phát hiện:
  - (i) control so với method (**Bảng 1**);
  - (ii) EBPG và Gini dự đoán được từ hình học box (bảng R², mở rộng bảng ở 2.2);
  - (iii) OA toàn ảnh so với OA box-relative theo size (**Hình 2**);
  - (iv) xếp hạng theo box của detector so với GT box;
  - (v) headroom theo từng mục tiêu (+0.003 / +0.004 dưới composite, lớn hơn dưới OA).
- Thông điệp: *không thể dùng các metric này để chọn hay đánh giá explanation; cần một endpoint bên ngoài.*
- File: `results/analysis/metric_validity_coco_test/{methods,by_size,size_trend,conditioning,headroom}_*.csv`.

### 4. Explanations as Evidence (khoảng 1.6 trang)
- 4.1 Bài toán: với mỗi detection d, có y = 1[TP ở IoU ≥ 0.5, ghép kiểu COCO]; các loại FP; chỉ dùng GT làm endpoint, không bao giờ làm feature.
- 4.2 Các họ evidence (bảng feature gọn; danh sách đầy đủ để ở supplement):
  - output: MetaDetect, gồm pre-NMS;
  - gradient;
  - TTA;
  - map structure;
  - curve shape (toàn ảnh và box-relative);
  - cross-explainer agreement;
  - VLM: P(yes) cho crop box và crop vùng salient, không đưa confidence vào prompt, đọc logit Yes/No.
- 4.3 Placebo và phân rã: cùng evaluator nhưng map không dùng model hoặc map bị xáo trộn. Định nghĩa ba vế nội dung / probing / cấu trúc.
- 4.4 Meta-classifier: GBM và logreg, CV 5 fold nhóm theo ảnh, paired image bootstrap.
- 4.5 Acquisition theo budget: tầng rẻ (output + gradient + TTA) và tầng đắt (explanation); chính sách VOI học được / theo độ bất định / ngẫu nhiên.
- 4.6 Dùng kết quả: CRC abstention (rủi ro FP được chấp nhận theo ảnh ≤ α) và AP re-scoring.
- **Hình 3:** sơ đồ pipeline (tái dùng khung v1 nhưng đổi vai trò các module, bảng B1).

### 5. Experimental Protocol (khoảng 0.6 trang)
- Detector: YOLOX-S, Faster R-CNN R50-FPN-v2; ngưỡng conf và NMS.
- Dữ liệu: COCO val2017 chia dev 300 / calib 1000 / test 2000 ảnh, **tách rời với ảnh oracle của v1** (`manifests/coco_val2017_v2.json`); BDD100K val với subgroup ngày/đêm và thời tiết.
- **Đăng ký trước:** gate, ngưỡng và lite config được commit trước test (dẫn SHA của commit); test không bao giờ dùng để chọn gì. Đây là một điểm cộng lớn với reviewer, nên viết rõ.
- Compute: GPU-h thực đo (`image_time`), số forward mỗi detection.

### 6. Results (khoảng 2.3 trang)
- 6.1 **Verification chính (Bảng 2):** AUROC / AUPRC / FPR@95 / ECE cho S0, S1, S2, S2+TTA, S2+VLM_box, S3_map_only, S3_curves_only, S3, P_controls, P_shuffled, S4, cho cả hai detector. Kèm **Hình 4**: forest plot ΔAUROC với CI (`fig_g1_forest`).
- 6.2 **Tín hiệu đến từ đâu:** phân rã nội dung / probing / cấu trúc; single-method rows (`S3_single_*`); **Hình 5**: AUROC theo loại FP.
- 6.3 **VLM:** crop box so với crop vùng salient; kiểm tra answer mass.
- 6.4 **Acquisition (Hình 6):** AUROC theo giây/detection (`fig_g2_acquisition`); % gain giữ lại ở 50%.
- 6.5 **Dùng kết quả:** CRC coverage và risk theo α (`fig_abstention`); TP coverage theo subgroup (size, độ hiếm của lớp); ΔAP50 / ΔAP75 với CI (**Bảng 3**).
- 6.6 **Chuyển miền BDD100K:** AUROC theo subgroup ngày/đêm và thời tiết (`fig_subgroup_auroc`).
- 6.7 (ngắn) **So sánh v1:** hàng selector / cascade chỉ để tham chiếu (`results/v2_legacy/`), đưa vào supplement nếu thiếu chỗ.

### 7. Discussion and Limitations (khoảng 0.4 trang)
- Tính vòng tròn chỉ giảm chứ không mất: feature và detection đến từ cùng một detector.
- Chi phí: khoảng 1.000 forward mỗi detection là công cụ để audit, không phải monitor thời gian thực; acquisition là cách giảm chi phí.
- Nhãn TP ở IoU 0.5 nên lỗi localization khó phát hiện; không tuyên bố đóng góp ở phần đó.
- Gradient-lite (nếu chưa nâng baseline); chỉ có 2 detector CNN; VLM 7B lượng tử hóa int4 / fp16.
- Mô hình evidence được giám sát bằng GT khi huấn luyện offline và chỉ không cần nhãn khi suy luận. Dùng đúng câu chữ theo AGENTS.md.

### 8. Conclusion (khoảng 0.2 trang)

### Supplementary
- Danh sách feature đầy đủ; siêu tham số của method (D-CLOSE 200 so với 2.000 mask, cấu hình LIME); tỉ lệ fallback.
- Kết quả theo từng lớp; toàn bộ bảng subgroup BDD; chi tiết CRC.
- Bảng tái lập: SHA, manifest, config đã đóng băng, GPU-h; checklist B7.
- Hàng v1 và phân tích lý do v1 thất bại (A3), viết lại ngắn gọn.

### Ngân sách trang

| Mục | Trang |
|---|---:|
| 1 Intro + Hình 1 | 1.0 |
| 2 Related | 0.6 |
| 3 Metric audit | 1.0 |
| 4 Method | 1.6 |
| 5 Protocol | 0.6 |
| 6 Results | 2.3 |
| 7–8 | 0.6 |
| **Tổng** | **7.7** |

---

## 7. Dàn bài dự phòng (kịch bản C: G1 fail)

**Tên bài:** *When a Gaussian Wins: Auditing Annotation-Free Explanation Metrics for Object Detection*, hoặc *Self-Referential Metrics Cannot Select Explanations for Object Detection*.

**Định dạng:** XAI4CV. Có hai lựa chọn: bản 4 trang không archival (giữ quyền nộp lại bản mở rộng ở main), hoặc bản 8 trang archival.

1. **Introduction:** việc chọn XAI thích ứng (AutoXAI) và đánh giá không cần nhãn đang được dùng rộng rãi. Câu hỏi: các metric đó có đo đúng thứ chúng tuyên bố không?
2. **Setup:** 4 method, 2 detector, metric, composite; các control; dữ liệu tách rời, đăng ký trước.
3. **Phát hiện 1, không có headroom:** oracle theo từng detection chỉ hơn always-G-CAME +0.003 / +0.004; per-detection selection không thể thắng. Kèm headroom theo từng mục tiêu. Dùng số A2 cộng bản chạy lại trên test v2.
4. **Phát hiện 2, hình học box:** control ≥ method; bảng R² của EBPG và Gini theo hình học box; so sánh `*_raw` với phiên bản có điều kiện box.
5. **Phát hiện 3, artifact của protocol OA:** bước perturbation tính theo phần trăm toàn ảnh xóa hết vật thể nhỏ ngay bước 1; so sánh với box-relative.
6. **Phát hiện 4, pseudo-GT so với GT box:** xếp hạng có ổn định không.
7. **Phát hiện 5, explanation evidence không vượt output + gradient:** kết quả âm có CI chặt, cho biết giới hạn trên của lượng thông tin.
8. **Khuyến nghị:** checklist cho người đánh giá XAI detection (luôn báo cáo control không dùng model, box-relative OA, endpoint độc lập, đăng ký trước).
9. **Limitations and conclusion.**

Hình / bảng: Bảng 1 (method so với control), Hình OA theo size, bảng R², bảng headroom, forest plot G1 (âm tính), phần câu chuyện v1 (A3) đưa vào supplement.

---

## 8. Điểm mới và related work

Kết quả tra cứu ngày 2026-09-29 trên arXiv, CVF, OpenReview. Chỉ đọc abstract, Semantic Scholar bị rate-limit, nên vẫn có thể sót bài.

### 8.1 Các công trình gần nhất

| Công trình | Làm gì | Khác gì với bài này |
|---|---|---|
| Spagnolo et al., MICCAI-W 2024 (arXiv:2409.03772) | 93 đặc trưng radiomic trích từ map SmoothGrad/Grad-CAM++ của từng instance, dùng logreg phân loại TP/FP tổn thương MS | **Tiền lệ gần nhất.** Dữ liệu y tế 3D, một explainer, không có baseline output/gradient, không có curve, agreement, VLM, budget hay conformal. |
| MetaDetect (arXiv:2010.01695); Riedlinger WACV 2023 (arXiv:2107.04517); LMD (arXiv:2306.07835) | Định nghĩa protocol meta-classification TP/FP từ output và gradient | Là baseline bắt buộc. Riedlinger đã chứng minh gradient bổ sung thông tin trực giao với output, nên bài này phải vượt *cả hai*. |
| Mohseni et al. 2019 (arXiv:1905.07679) | Dự đoán lỗi từ saliency cho bài toán lái (steering) | Không phải detection. |
| GAIA, ML-LOO, ABC, Hornauer & Belagiannis | Thống kê attribution để phát hiện OOD, adversarial, hoặc ước lượng confidence | Chỉ làm cho classification. |
| ODAM (ICLR 2023) | ODAM-NMS dùng độ giống nhau của explanation để loại box trùng | Đã có tiền lệ "dùng explanation để hậu xử lý detection", nhưng chỉ cho de-duplication, không phải TP/FP. |
| DetRefiner (arXiv:2605.10190), VLM verifier (arXiv:2509.09958) | Re-score detection bằng feature DINOv3 hoặc VLM trên crop | Không dùng explanation. Là baseline cho phần VLM. |
| Pointing game + center baseline (Zhang 2016), Fake-CAM (Poppi CVPRW 2021), dummy map (Stassin 2023), ROAD, Blücher TMLR 2024, LATEC, MetaQuantus | Map tầm thường hoặc cố tình sai vẫn đạt điểm tốt; metric removal bị rò rỉ; metric bất đồng | Đều làm cho classification. Với detection chỉ có Padmanabhan 2023 (randomization test). **Chưa có bài nào cho thấy Gaussian trên box ngang ngửa method thật trên EBPG/OA/Gini, và chưa có protocol box-relative.** |

### 8.2 Những gì được và không được tuyên bố

- **Được:**
  - Nghiên cứu có kiểm soát đầu tiên *trên detection 2D ảnh tự nhiên*: explanation evidence có thêm tín hiệu TP/FP so với meta-classifier output + gradient hay không, có gate ΔAUROC và control prior/shuffle. Kết quả âm được kiểm soát tốt cũng công bố được.
  - Chọn detection nào để explain theo budget, kết hợp CRC abstention. Không tìm thấy công trình nào trước đó.
  - Audit metric riêng cho detection: kết quả bbox prior và protocol box-relative. Trình bày như **phần mở rộng** các phát hiện đã có ở classification (Zhang 2016, Poppi 2021, Stassin 2023, ROAD).
- **Không được:**
  - "Đầu tiên dùng explanation để phát hiện lỗi". Spagnolo 2024 và các bài classification đã làm.
  - "Mô hình evidence không cần nhãn". Nó được giám sát bằng GT khi huấn luyện offline, chỉ không cần nhãn khi suy luận.

### 8.3 Lưu ý nhạy cảm

- ODExAI, D-CLOSE và G-CAME theo kết quả tra cứu đều đến từ AELab (UNB), cùng nhóm với bài này. Audit metric vì vậy phê bình chính những metric nhóm đã báo cáo.
  - **Double-blind:** trích dẫn ở ngôi thứ ba ("Nguyen et al. propose…"), không viết "our prior work".
  - **Cách viết:** giọng xây dựng. Metric vẫn hữu ích cho việc so sánh *giữa các method có điều kiện box như nhau*, nhưng không phù hợp để *chọn* hay *xác thực* explanation.
- Trọng số Gaussian ở tâm của G-CAME tự bản thân là một box prior, và box của detector là input cho mọi explainer. Phải nói rõ ngay ở mục 3 rằng EBPG theo box của detector có tính vòng tròn một phần. Các biến thể `*_raw` là câu trả lời, nên đặt chúng vào bảng chính.

### 8.4 Baseline reviewer sẽ đòi (bổ sung cho mục 5)

- Confidence; MetaDetect *có* thống kê pre-NMS; gradient kiểu Riedlinger; và tổ hợp của hai nhóm này.
- MC-dropout, ensemble, **tính nhất quán dưới TTA**; hiệu chỉnh hậu kỳ Platt / isotonic (setup của Kuzucu ECCV 2024).
- Feature nội tại RoI / kiểu SAFE (Wilson ICCV 2023) đưa qua MLP.
- Re-scorer trên crop kiểu DetRefiner (DINO hoặc CLIP).
- **VLM trên crop không có saliency**, để tách phần đóng góp của saliency.
- Map prior / shuffle đi qua cùng pipeline feature (đã có).
- So sánh cân compute.
- Metric kết quả: AUROC/AUPRC, AP sau re-score, **LaECE**, đường risk–coverage, conformal coverage.
- Riêng cho audit: removal debias kiểu ROAD, nhiều baseline perturbation (blur, mean fill, inpainting), sanity check randomization model.

Ưu tiên nếu thời gian có hạn: TTA, pre-NMS, VLM crop không saliency, Platt/isotonic, rồi thêm một baseline perturbation cho audit (mean fill). DINO re-scorer và SAFE chỉ cần nếu nhắm main conference.

---

## 9. Timeline tới deadline

Lịch venue (theo tra cứu; mục có dấu ~ là ước tính từ năm 2026):

| Venue | Deadline | Khả thi? |
|---|---|---|
| CVPR 2027 main | Đăng ký 10/11/2026, nộp 16/11/2026 | **Không.** Còn 7 tuần mà pilot chưa chạy. |
| WACV 2027 workshops | ~đầu–giữa 10/2026 | Không kịp. |
| **CVPR 2027 workshops**: SAIAD, XAI4CV (8 trang archival; XAI4CV còn có track 4 trang không archival) | ~cuối 02 – đầu 03/2027 (năm 2026: SAIAD 02/03, XAI4CV 27/02) | **Mục tiêu chính.** |
| ICCV 2027 main (Hồng Kông, 10/2027) | ~đầu 03/2027, chưa công bố | Chỉ khi rơi vào kịch bản A và đã có baseline mạnh cùng detector thứ ba. |
| BMVC 2027 | ~05/2027 | Phương án nộp lại. |

Kế hoạch ngược từ mốc khoảng 25/02/2027:

| Thời gian | Việc | Mốc quyết định |
|---|---|---|
| 29/09 – 12/10 | Pilot G1 trên CPU với dữ liệu v1 (nếu có annotation); sửa phân tích 5.1; thêm TTA và pre-NMS cho YOLOX; env check trên cluster | — |
| 13/10 – 20/10 | Pilot dev YOLOX (1–3 GPU-h) | **D1 hướng G1**: đi tiếp theo A/B hay chuyển sang C |
| 20/10 – 10/11 | Nâng baseline cho FRCNN, pilot FRCNN, calib τ, **đóng băng** config, gate và feature set | Commit đóng băng |
| 10/11 – 15/12 | Test COCO (FRCNN tốn thời gian nhất), BDD100K | — |
| 15/12 – 15/01 | Phân tích, hình, hàng v1; tra cứu related work đầy đủ (đọc full text 5 bài gần nhất) | Chọn kịch bản A, B hoặc C |
| 15/01 – 20/02 | Viết bài; review nội bộ (supervisor); supplementary; chuẩn bị release code, manifest, config đã đóng băng | — |
| ~25/02/2027 | Nộp SAIAD hoặc XAI4CV | — |
| ~03/2027 | (tùy chọn) Nộp ICCV 2027 nếu kịch bản A | — |

Nếu D1 cho hướng âm tính rõ, hãy chuyển ngay sang kịch bản C. Khi đó chỉ cần metric audit trên test với các method và control, tiết kiệm khoảng 60% GPU-h. Phần thời gian còn dư có thể dùng để làm thêm baseline perturbation và sanity check randomization cho audit.

---

## 10. Những phần viết và chuẩn bị được trước khi có kết quả

Nguyên tắc: mọi thứ **không phụ thuộc vào con số** nên xong trước khi chạy test. Khoảng một nửa nội dung 8 trang thuộc loại này (Related Work, Method, Protocol, phần setup của audit, Limitations, phần lớn Intro). Cách chạy trên cluster xem ở `cluster/RUNBOOK_VI.md`.

### 10.1 Viết được ngay: không phụ thuộc kết quả

| # | Phần | Nguồn nội dung | Công sức | Ghi chú |
|---|---|---|---|---|
| W1 | **§4 Method** (khoảng 1.6 trang) | `src/evidence.py`, `src/gt_matching.py`, `scripts/analyze_evidence.py`, `src/vlm_judge.py` | 1–2 ngày | Viết được đầy đủ công thức: gán nhãn TP/FP và loại FP; các họ feature (map, curve, agreement, gradient, output, VLM với P(yes) = P(Yes)/(P(Yes)+P(No))); phân rã placebo; CV nhóm theo ảnh; gain VOI theo log-loss; ngưỡng CRC (n/(n+1))·R̂ + 1/(n+1) ≤ α; re-score AP. Phần giá trị nhất nên viết trước. |
| W2 | **§3.1–3.2**: định nghĩa metric, control, protocol box-relative | `src/evaluator.py`, `src/xai_methods/controls.py` | nửa ngày | Chỉ phần 3.3 (các phát hiện) phải chờ số. Ghi rõ tính vòng tròn của EBPG theo box detector (mục 8.3). |
| W3 | **§5 Protocol** | config, manifest, B4 | nửa ngày | Detector, ngưỡng, split, đăng ký trước, thống kê (5 fold, bootstrap 2000), siêu tham số (LIME 500, D-CLOSE 200, σ G-CAME 0.35, fill 114, baseline blur 51, 10 bước). Chỉ GPU-h phải điền sau. |
| W4 | **§7 Limitations** | mục 6 của dàn bài | vài giờ | Gần như không phụ thuộc kết quả. |
| W5 | **Supplementary A**: bảng feature đầy đủ, prompt VLM nguyên văn, siêu tham số explainer, checklist tái lập | code và config | 1 ngày | Nên sinh bảng feature tự động từ tên cột (`out__`, `grad__`, `mf__`, `cf__`, `cfb__`, `ag__`, `vlm__`) để không lệch khỏi code. |
| W6 | **Supplementary B**: tóm tắt vì sao v1 thất bại (A3) | `RESULTS_AND_PLAN.md` Part A | nửa ngày | Chỉ dùng các câu đã sửa ở A4. |
| W7 | **Tài liệu đăng ký trước** (`PREREGISTRATION.md`, commit và tag trước test) | B1, `config/evidence_config_lite.yaml` | nửa ngày | Giả thuyết, endpoint chính, gate, feature set, các cặp so sánh, luật loại dữ liệu, cách xử lý nhiều phép so sánh. Dẫn SHA của tag trong §5. Rất có giá trị khi phản biện. |

### 10.2 Chuẩn bị hạ tầng cho số liệu và hình

| # | Việc | Vì sao nên làm trước | Công sức |
|---|---|---|---|
| P1 | **Script xuất số liệu vào LaTeX** (`scripts/export_paper_numbers.py`): đọc `summary.json` và các CSV, ghi `paper/numbers.tex` gồm các `\newcommand` (ví dụ `\GoneDeltaYolox`) | Thay mọi `\res{}` bằng macro; số trong bài không bao giờ lệch khỏi kết quả; chạy lại một lệnh là cập nhật cả bài. Thử được ngay bằng dữ liệu smoke hoặc `tests/toy_models.py`. | 1 ngày |
| P2 | **Khung bảng LaTeX** cho Bảng 1 (audit), Bảng 2 (G1), Bảng 3 (AP/CRC), các ô dùng macro của P1 | Chốt bố cục và số trang sớm | nửa ngày |
| P3 | **Hình 3: sơ đồ pipeline** (vector hoặc TikZ) | Hoàn toàn không phụ thuộc kết quả | nửa ngày |
| P4 | **Bố cục Hình 1 (teaser)** và kiểu hình chung | `src/v2_figures.py` đã có style; chỉ chờ chọn ví dụ | vài giờ |
| P5 | Chạy thử notebook 08 trên dữ liệu smoke hoặc pilot | Phát hiện lỗi hình trước khi có dữ liệu test | vài giờ |

### 10.3 Code phải xong trước mốc đóng băng (không phải viết bài, nhưng quyết định bài viết được gì)

- **Trước pilot:** baseline mạnh ở mục 5.2 (TTA, pre-NMS, gradient kiểu Riedlinger).
- **Trước đóng băng:** bổ sung phân tích ở mục 5.1:
  - so sánh S4 với S2+VLM_box;
  - các set `S3_map_only` / `S3_curves_only`;
  - AUROC theo loại FP;
  - hồi quy IoU;
  - S3-compact.
- **Tùy chọn:** pilot G1 trên CPU với dữ liệu v1. Chạy được trên login node ngay sau khi `download_data.sh` tải xong annotation.

### 10.4 Việc hành chính (nên làm sớm, vì có độ trễ)

- **Tài khoản OpenReview cho mọi đồng tác giả.** Hồ sơ mới dùng email không thuộc tổ chức có thể mất đến khoảng 2 tuần để được duyệt.
- **Chọn track trước khi viết:** XAI4CV 8 trang archival hay 4 trang non-archival.
  - Nếu muốn nộp bản mở rộng lên ICCV 2027 hoặc main conference sau đó: bài workshop archival dài hơn 4 trang thường bị coi là đã xuất bản theo quy định dual-submission, nên hãy chọn bản non-archival.
  - Kiểm tra lại CFP khi được công bố.
- **Author kit:** viết nháp bằng kit CVPR 2026, rồi chuyển sang kit 2027 khi có. Kiểm tra giới hạn trang của SAIAD và XAI4CV.
- **Repo ẩn danh để nộp kèm supplementary** (ví dụ anonymous.4open.science): chuẩn bị README v2 và license.
- **Đọc toàn văn** 5 bài gần nhất (Spagnolo 2024, MetaDetect, Riedlinger 2023, Padmanabhan 2023, Stassin 2023) để làm sắc câu phân biệt ở Related Work. Kiểm tra câu về Fake-CAM còn đánh dấu CHECK.

### 10.5 Phải chờ kết quả

- Số liệu ở §3.3, §6, abstract và đoạn "Our findings" của Intro.
- Chọn kịch bản A, B hoặc C, kéo theo tên bài và cách đặt vấn đề.
- Các ví dụ cho teaser, phần thảo luận và kết luận.

### 10.6 Thứ tự đề xuất (song song với việc chuẩn bị cluster)

1. **Tuần 1:** code 5.2 và 5.1 (vì chặn pilot và đóng băng); W7 đăng ký trước; tạo tài khoản OpenReview.
2. **Tuần 2** (trong lúc chạy smoke / pilot): W1 Method, W2, W3.
3. **Tuần 3** (trong lúc chạy calib / test): P1 script xuất số, P2 bảng, P3 sơ đồ, W5 supplementary.
4. **Trong lúc chạy BDD / legacy:** W4, W6, đọc toàn văn related work, P5.
5. **Khi có kết quả test:** chọn kịch bản, viết §3.3 và §6, rồi abstract, đoạn findings của Intro và kết luận.
