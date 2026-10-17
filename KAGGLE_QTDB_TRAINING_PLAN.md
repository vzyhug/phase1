# Kế hoạch triển khai và huấn luyện QTDB trên Kaggle

## 1. Phạm vi đúng

Huấn luyện và đánh giá Conditional DDPM khử nhiễu ECG theo giao thức QTDB hiện có trong repo:

- ECG sạch: **QT Database (QTDB)**;
- đầu vào/đầu ra: **một kênh**, cửa sổ 512 mẫu;
- nhiễu: MIT-BIH Noise Stress Test Database;
- hai phiên bản nhiễu: `nv1` và `nv2` (`noise_type1`, `noise_type2`);
- mô hình: 1D U-Net + HNF + FiLM/Bridge + self-attention;
- loss miền thời gian kết hợp miền tần số;
- inference bằng DDIM;
- so sánh MyDDPM với các baseline DAE/FGDAE trên cùng dữ liệu và metric.

Notebook training chính:

```text
notebooks/kaggle_train_qtdb_1ch_unet_multidomain.ipynb
```

Không dùng `notebooks/kaggle_training.ipynb` làm notebook chính vì notebook đó thuộc pipeline LUDB 12 đạo trình.

## 2. Dữ liệu chuẩn

Theo `results/dataset_audit (1).csv`:

| Version | Train segments | Test segments | Test shape | Dtype hiện có |
|---|---:|---:|---|---|
| `nv1` | 91,062 | 15,535 | `(15535, 512, 1)` | `float64` |
| `nv2` | 91,062 | 15,535 | `(15535, 512, 1)` | `float64` |

Hai version có cùng clean-test hash trong audit hiện tại. Phải giữ nguyên đặc điểm này để so sánh paired giữa `nv1` và `nv2`.

Loader nên chuyển từng batch sang `float32` để giảm RAM/VRAM, nhưng không sửa file nguồn trước khi ghi hash và audit.

## 3. Kiến trúc triển khai Kaggle

Tách thành ba notebook/version:

1. **QTDB data audit/build — CPU**
   - kiểm tra `noise_type1` và `noise_type2`;
   - xác nhận số mẫu, shape, dtype, finite values và clean hash;
   - tạo manifest train/validation/test;
   - xuất Kaggle Dataset bất biến cho training.
2. **QTDB training — GPU**
   - chạy notebook QTDB 1-channel;
   - chọn `nv1` hoặc `nv2` bằng config, không trộn hai version;
   - chạy smoke test, pilot, rồi full training;
   - lưu checkpoint để resume qua nhiều phiên.
3. **QTDB final evaluation/benchmark — GPU**
   - đánh giá best checkpoint trên full test 15,535 segments;
   - xuất metric per-segment, summary, latency và statistical tests.

Kaggle cho phép lưu tối đa 20 GB trong `/kaggle/working`; output của notebook trước có thể gắn lại làm Input. Phiên CPU/GPU có giới hạn 12 giờ, nên full training phải hỗ trợ resume. Tài liệu: <https://www.kaggle.com/docs/notebooks>.

## 4. Gate 0 — khóa dữ liệu và ngăn leakage

Không chạy full training trước khi hoàn thành gate này.

### 4.1. Giữ đúng giao thức `nv1/nv2`

- Không coi `nv1` và `nv2` là hai artifact khác nhau.
- Đây là hai phiên bản chia nguồn nhiễu của benchmark; mỗi version cần checkpoint riêng.
- Không trộn train `nv1` với train `nv2`.
- Không dùng test của version nào để chọn checkpoint hoặc hyperparameter.
- Giữ cùng clean test set giữa hai version.

### 4.2. Chia train/validation/test

- Giữ nguyên test set 15,535 segments.
- Tạo validation từ 91,062 train segments.
- Ưu tiên chia theo QTDB record/group, không chia ngẫu nhiên từng cửa sổ.
- Không để các cửa sổ chồng lấp của cùng record xuất hiện ở cả train và validation.
- Không để cùng đoạn/kênh nhiễu xuất hiện ở train và test của một noise version.
- Nếu dữ liệu legacy không chứa record ID, phục hồi manifest từ pipeline tạo QTDB trước khi full run.

### 4.3. Manifest bắt buộc

Mỗi sample nên truy vết được:

```text
sample_id
qtdb_record
window_start
signal_channel
noise_version
noise_record_or_channel
noise_start
noise_level_or_input_snr
split
seed
```

Nếu không phục hồi được trường nào từ dữ liệu legacy, ghi rõ khoảng trống đó trong audit; không tuyên bố đã kiểm chứng leakage cho trường chưa có.

### 4.4. Kiểm tra bắt buộc

- tensor cho model có shape `(N, 1, 512)`;
- clean/noisy cùng số mẫu và đúng alignment;
- không NaN/Inf;
- thống kê biên độ hợp lý sau normalization;
- train/validation/test group rời nhau;
- full test đúng 15,535 segments cho cả hai version;
- clean-test hash của `nv1` và `nv2` giống nhau;
- xem trực quan các cặp clean/noisy ngẫu nhiên.

**Đầu ra:** Kaggle Dataset QTDB v1 gồm dữ liệu, manifest, hashes, preprocessing config và audit report. Notebook train chỉ đọc dataset này, không regenerate dữ liệu.

## 5. Gate 1 — smoke test

Chạy `nv1` trước:

- 512–2,048 train segments;
- 128–256 validation segments;
- 2–3 epoch;
- batch size 32, giảm còn 16 nếu thiếu VRAM;
- DDIM 15 steps trên validation subset nhỏ, cố định.

Tiêu chí đạt:

- input/output đúng `(B, 1, 512)`;
- loss hữu hạn và giảm;
- overfit được một batch nhỏ;
- notebook chạy sạch từ đầu đến cuối;
- tạo đủ `last_resume`, `best_resume`, `model.pth`, log và config;
- resume khôi phục đúng epoch, optimizer, scheduler, EMA và RNG;
- checkpoint inference nạp độc lập và sinh đúng shape;
- peak VRAM còn biên an toàn khoảng 10–15%.

Sau khi `nv1` đạt, lặp lại smoke test cho `nv2`.

## 6. Gate 2 — pilot toàn dữ liệu

Chạy 10–20 epoch cho mỗi version:

- đo giây/epoch, peak VRAM/RAM và kích thước checkpoint;
- log train/validation total loss, time loss, frequency loss và learning rate;
- tính PRD, cosine và SNR trên validation subset cố định;
- xác nhận validation không chạm test set;
- thử resume giữa hai Kaggle version.

Tính số epoch mỗi phiên bằng số đo thực tế:

```text
epochs_per_session = floor((10.5 giờ - overhead) / seconds_per_epoch)
number_of_sessions = ceil(target_epochs / epochs_per_session)
```

Chừa khoảng 1.5 giờ cho validation, lưu checkpoint và đóng gói output. Nếu bật AMP, phải đối chiếu một run FP32/AMP ngắn và chỉ giữ AMP khi loss, gradient và output đều hữu hạn.

## 7. Full training

### 7.1. Ma trận bắt buộc

| Run | Train | Test chính | Mục đích |
|---|---|---|---|
| `qtdb_ddpm_nv1` | QTDB noise version 1 | QTDB test `nv1` | benchmark đúng miền |
| `qtdb_ddpm_nv2` | QTDB noise version 2 | QTDB test `nv2` | benchmark đúng miền |

Hai run dùng cùng architecture, seed policy và hyperparameters; chỉ khác noise version/split.

### 7.2. Quy tắc training

- seed và validation protocol cố định;
- số epoch tối đa được khóa trong config, có early stopping đủ kiên nhẫn;
- chọn best checkpoint bằng validation loss định nghĩa trước;
- dùng EMA weights cho final inference nếu pilot xác nhận tốt hơn raw weights;
- checkpoint nguyên tử sau mỗi epoch;
- log epoch time, learning rate, GPU memory và validation sample count;
- experiment name luôn chứa `nv1` hoặc `nv2` để tránh ghi đè.

Mỗi phiên Kaggle:

1. gắn dataset đã pin và output phiên trước;
2. kiểm tra data hash, config hash và checkpoint version;
3. resume từ `last_resume.pth`;
4. train đến mốc epoch an toàn;
5. load-test `last_resume.pth`, `best_resume.pth` và `model.pth`;
6. publish output thành version mới;
7. chỉ bỏ version cũ sau khi version mới đã được kiểm tra.

### 7.3. Run xác nhận tùy chọn

| Thiết kế | Số checkpoint |
|---|---:|
| `nv1`, `nv2` × 1 seed | 2 |
| `nv1`, `nv2` × 2 seed | 4 |

Hoàn thành hai checkpoint chuẩn trước khi mở rộng seed hoặc ablation.

## 8. Final evaluation

### 8.1. Notebook

```text
notebooks/kaggle_qtdb_rule_evaluation.ipynb
notebooks/kaggle_qtdb_rule_benchmark.ipynb
```

Đánh giá mỗi checkpoint trên đúng noise version tương ứng. Cross-version (`train nv1 → test nv2`, `train nv2 → test nv1`) là thí nghiệm generalization riêng, không gộp vào benchmark chính.

### 8.2. Metric

- SSD;
- MAD;
- PRD theo định nghĩa paper đã khóa;
- RMSE và MAE;
- cosine similarity;
- SNR hoặc SNR improvement theo cùng một định nghĩa;
- số tham số;
- inference time, segments/second và peak VRAM.

Báo cáo mean, standard deviation, median và confidence interval. Nếu có record ID, aggregate hoặc bootstrap theo record vì các segment có thể chồng lấp và không độc lập.

### 8.3. Sampling

- benchmark chính: DDIM 50 steps, `eta=0`, 1-shot để khớp kết quả hiện có;
- DDIM 15 steps là thí nghiệm speed/quality riêng;
- chỉ chạy 3/5/10-shot sau khi 1-shot đúng;
- ghi rõ steps, eta, shots, seed và batch size.

### 8.4. Baseline

So sánh với noisy input và các model:

- Vanilla DAE, CNN-DAE, DRNN, FCN-DAE, DeepFilter;
- ACDAE, CBAM-DAE, TCDAE;
- FGDAE `q=1..4`.

Tất cả model phải dùng cùng 15,535 test segments và cùng metric implementation. Kết quả hiện có ghi nhận MyDDPM khoảng 18 segments/s với DDIM 50 steps, chậm hơn nhiều so với DAE, nên latency là tiêu chí bắt buộc.

## 9. Output mỗi experiment

```text
/kaggle/working/qtdb_experiments/<experiment_id>/
├── config.json
├── environment.txt
├── git_commit.txt
├── data_manifest.json
├── split_manifest.json
├── audit_report.json
├── training_log.csv
├── training_curves.png
├── last_resume.pth
├── best_resume.pth
├── model.pth
├── final.pth
└── run_summary.json
```

`model.pth` là weights tốt nhất dùng cho inference. `last_resume.pth` chứa trạng thái để tiếp tục training.

## 10. Lịch triển khai

| Mốc | Công việc | Điều kiện hoàn tất |
|---|---|---|
| M0 | Audit QTDB `nv1/nv2`, phục hồi record/group manifest | kiểm tra leakage đạt |
| M1 | Đóng gói Kaggle Dataset QTDB v1 | hash + manifest + audit report |
| M2 | Smoke test `nv1`, sau đó `nv2` | train/resume/inference đạt |
| M3 | Pilot 10–20 epoch mỗi version | có throughput, VRAM, lịch session |
| M4 | Full train `qtdb_ddpm_nv1` | best checkpoint ổn định |
| M5 | Full train `qtdb_ddpm_nv2` | best checkpoint ổn định |
| M6 | Full-test 15,535 segments | metrics + latency + per-segment rows |
| M7 | Joint benchmark và statistical tests | bảng và biểu đồ cuối tái lập được |

## 11. Tiêu chí hoàn thành

- `nv1` và `nv2` có checkpoint riêng;
- model dùng đúng một kênh, 512 mẫu;
- validation tách theo record/group khi metadata cho phép;
- resume đã được thử qua hai phiên Kaggle;
- best checkpoint chỉ được chọn bằng validation;
- mỗi checkpoint chạy full test 15,535 segments tương ứng;
- MyDDPM và baseline dùng cùng dữ liệu, metric và protocol;
- kết quả truy ngược được tới dataset version, hash, commit, config và checkpoint;
- output cuối được sao lưu ngoài session tạm.

## 12. Việc cần làm ngay

1. Dùng notebook `kaggle_train_qtdb_1ch_unet_multidomain.ipynb` làm training entry point.
2. Audit nguồn tạo `noise_type1/noise_type2` và phục hồi record/group manifest.
3. Chuyển tensor sang `float32` trong loader sau khi audit file nguồn.
4. Chạy smoke test `nv1`, kiểm tra resume, rồi lặp lại cho `nv2`.
5. Chạy pilot để quyết định batch size và số epoch mỗi phiên.

