# Kế hoạch cũ (LUDB) — không sử dụng

> Kế hoạch đúng cho QTDB đã được chuyển sang `KAGGLE_QTDB_TRAINING_PLAN.md`.

# Kế hoạch triển khai và huấn luyện trên Kaggle

## 1. Mục tiêu

Huấn luyện và đánh giá mô hình Conditional DDPM khử nhiễu ECG 12 đạo trình trên LUDB với ba artifact của MIT-BIH NST (`bw`, `em`, `ma`) theo một quy trình:

- tái lập được từ dữ liệu, mã nguồn và seed;
- không rò rỉ record ECG hoặc đoạn nhiễu giữa train/validation/test;
- tiếp tục được qua nhiều phiên Kaggle;
- tạo đủ checkpoint, log, manifest và kết quả để viết báo cáo;
- không dùng test set để chọn checkpoint hoặc siêu tham số.

Notebook khởi điểm là `notebooks/kaggle_training.ipynb`. Cấu hình hiện tại dùng U-Net 1D 12 đạo trình, 50 diffusion steps, loss thời gian + STFT, EMA, batch size 32 và mục tiêu 300 epoch.

## 2. Kiến trúc triển khai đề xuất

Tách quy trình Kaggle thành ba notebook/version độc lập:

1. **Data build (CPU, Internet bật)**
   - tải LUDB và MIT-BIH NST;
   - resample, segment và chuẩn hóa;
   - chia record trước khi sinh nhiễu;
   - sinh dữ liệu và lưu metadata;
   - xuất một Kaggle Dataset bất biến dùng làm input cho các lần train.
2. **Training (GPU, ưu tiên Internet tắt)**
   - gắn dataset đã build và snapshot mã nguồn;
   - chạy smoke test, pilot rồi full training;
   - lưu checkpoint mỗi epoch vào `/kaggle/working`;
   - khi hết phiên, publish output và gắn output đó làm input cho phiên kế tiếp.
3. **Final evaluation (GPU)**
   - chỉ nạp checkpoint tốt nhất đã chọn bằng validation;
   - chạy test đúng một lần theo artifact, noise level, split và số shot;
   - xuất bảng metric, latency, hình minh họa và manifest đánh giá.

Kaggle hiện cho phép lưu tối đa 20 GB output dưới `/kaggle/working`; output notebook có thể được gắn lại làm input cho notebook sau. Một phiên CPU/GPU có giới hạn thực thi 12 giờ, nên full training phải được thiết kế để resume qua nhiều version. Tham khảo: <https://www.kaggle.com/docs/notebooks>.

## 3. Gate 0 — sửa và khóa giao thức dữ liệu

Không chạy full training trước khi các điều kiện sau đạt hết.

### 3.1. Split

- Test LUDB cố định: record `180..200`.
- Chia train/validation theo **record**, không chia theo window. Notebook hiện đã làm đúng phần này.
- Sinh và lưu `split_manifest.json` gồm record, số window và số sample của từng split.
- Nếu cần hai fold nghiên cứu, tạo hai manifest cố định; không sinh lại split tùy ý giữa các model.

### 3.2. Chống leakage từ NSTDB

Code hiện tại chọn ngẫu nhiên một patch từ toàn bộ mỗi tín hiệu nhiễu cho mọi sample. Cần sửa trước full run:

- chia phạm vi hoặc channel NSTDB thành train/validation/test trước khi lấy patch;
- cấm khoảng noise source chồng lấp giữa các split;
- lưu cho từng sample: `clean_record`, `window_start`, `artifact`, `noise_channel`, `noise_start`, `delta`, `input_snr_db`, `seed`, `split`;
- kiểm tra tự động rằng không có identity hoặc khoảng noise bị dùng chéo split.

### 3.3. Cân bằng và khả năng tái lập

- Dùng tỷ lệ mixed-noise ban đầu `BW:EM:MA = 1:1:1` theo đúng số mẫu, không chỉ kỳ vọng ngẫu nhiên.
- Giữ cùng phân bố mức nhiễu giữa ba artifact.
- Bổ sung các mức input SNR cố định để so sánh công bằng, ví dụ `-5, 0, 5, 10 dB`; có thể giữ `delta 0.2–2.0` như một thí nghiệm robustness riêng.
- Ghi `git commit`, phiên bản package, hash file dữ liệu và toàn bộ config vào manifest.
- Pin dependency quan trọng thay vì dùng toàn bộ khoảng phiên bản mở trong `requirements.txt`.

### 3.4. Kiểm tra dữ liệu bắt buộc

- shape `(N, 512, 12)`, dtype `float32`, không NaN/Inf;
- 12 lead đúng thứ tự và biên độ sau chuẩn hóa hợp lệ;
- record sets đôi một rời nhau;
- noise ranges đôi một rời nhau giữa split;
- số mẫu và tỷ lệ artifact/SNR đúng thiết kế;
- clean/noisy/metadata khớp từng hàng;
- vẽ ngẫu nhiên ít nhất 10 mẫu cho mỗi artifact và mỗi mức SNR.

**Đầu ra Gate 0:** Kaggle Dataset phiên bản `v1` gồm dữ liệu, metadata, split manifest, config và audit report. Sau khi tạo dataset này, training notebook không regenerate dữ liệu nữa.

## 4. Gate 1 — smoke test

Chạy trên GPU với cấu hình nhỏ:

- `MAX_TRAIN_SAMPLES = 512–2048`;
- validation `128–256` mẫu;
- `EPOCHS = 2–3`;
- batch size bắt đầu `16`, sau đó thử `32`;
- validation sampling dùng DDIM 15 steps trên tối đa 32 mẫu.

Tiêu chí đạt:

- notebook chạy sạch từ đầu tới cuối trên một Kaggle version;
- loss hữu hạn và giảm qua vài epoch;
- model forward/backward đúng `(B, 12, 512)`;
- `last_resume`, `best_resume`, `model.pth`, CSV log và config đều được tạo;
- có thể resume và bắt đầu đúng ở epoch kế tiếp;
- checkpoint inference nạp được độc lập;
- peak VRAM còn biên an toàn tối thiểu khoảng 10–15%;
- một batch overfit nhỏ được để xác nhận pipeline học được.

Nếu bật AMP để tăng tốc, phải so sánh một run ngắn FP32 và AMP; chỉ giữ AMP khi loss và output đều hữu hạn, không sai lệch bất thường.

## 5. Gate 2 — pilot toàn dữ liệu

Chạy `10–20` epoch trên toàn bộ train set để đo thực tế:

- thời gian preprocessing (nếu còn);
- giây/epoch;
- peak VRAM/RAM;
- train/validation total, time loss và frequency loss;
- SNR improvement và cosine trên validation;
- kích thước checkpoint/output.

Từ số đo pilot, tính:

```text
epochs_per_session = floor((10.5 giờ - overhead) / seconds_per_epoch)
number_of_sessions = ceil(target_epochs / epochs_per_session)
```

Dùng 10.5 giờ thay vì 12 giờ để còn thời gian validation, vẽ biểu đồ và ghi checkpoint. Không ước lượng cứng thời gian trước pilot vì loại GPU Kaggle và throughput có thể thay đổi.

Tiêu chí đạt:

- không OOM và không có non-finite loss;
- resume tái lập đúng learning rate, epoch, optimizer, scheduler, EMA và RNG;
- train/validation loss không phân kỳ;
- validation SNR improvement không suy giảm kéo dài;
- tổng output nhỏ hơn giới hạn `/kaggle/working`.

## 6. Full training

### 6.1. Run ưu tiên (MVP nghiên cứu)

Huấn luyện một **mixed-noise 12-lead model** trước:

- seed 42;
- split manifest cố định;
- `BASE_FEATS=80`, `T=50`, `batch_size` theo pilot;
- Adam `1e-4`, gradient clip `1.0`, EMA `0.999`;
- tối đa 300 epoch, nhưng cho phép dừng sớm dựa trên validation nếu không cải thiện đủ lâu;
- chọn checkpoint bằng `val_total` đã xác định trước, không bằng test metric;
- log thêm learning rate, epoch time, GPU memory và validation sample count.

Mỗi phiên:

1. gắn output phiên trước làm Input;
2. kiểm tra hash config/split trước khi resume;
3. train đến mốc epoch của phiên;
4. lưu checkpoint nguyên tử mỗi epoch;
5. đảm bảo `last_resume` và `best_resume` tồn tại;
6. publish output thành một version mới;
7. chỉ xóa version cũ sau khi version mới đã kiểm tra tải/nạp thành công.

### 6.2. Run xác nhận

Sau khi MVP ổn định:

- chạy thêm seed hoặc fold thứ hai cho mixed model;
- báo cáo mean ± std giữa hai run;
- chỉ sau đó mới mở rộng sang model chuyên biệt `BW`, `EM`, `MA` nếu câu hỏi nghiên cứu cần.

Ma trận đầy đủ tùy chọn:

| Nhóm | Số checkpoint |
|---|---:|
| Mixed noise × 2 fold/seed | 2 |
| BW/EM/MA chuyên biệt × 2 fold | 6 |
| Tổng | 8 |

Không nên bắt đầu cả 8 checkpoint cùng lúc. Hoàn tất mixed MVP và final evaluation trước để tránh tiêu GPU cho một pipeline chưa được khóa.

## 7. Final evaluation

Test set chỉ được mở sau khi config, checkpoint selection và protocol đã khóa.

### 7.1. Báo cáo bắt buộc

- metric theo từng artifact: SSD, MAD, PRD chuẩn, RMSE, MAE, cosine, SNR-in, SNR-out, SNR improvement;
- mean, standard deviation, median và khoảng tin cậy bootstrap theo **record**, không coi các window chồng lấp là độc lập;
- kết quả theo input SNR/noise level;
- macro average giữa `BW`, `EM`, `MA`;
- kết quả riêng từng fold/seed rồi mới tổng hợp;
- throughput, latency, số tham số và peak VRAM.

### 7.2. Sampling

- Gate đầu: `1-shot`, DDIM 15 hoặc 50 steps;
- chỉ chạy `3/5/10-shot` sau khi 1-shot đã xác nhận đúng;
- ghi rõ `steps`, `eta`, `shots`, seed và thời gian suy luận;
- so sánh chất lượng–latency thay vì chỉ báo cáo chất lượng.

### 7.3. So sánh

- so với noisy input (no-op baseline);
- so với các baseline FGDAE/DAE hiện có bằng cùng test set và cùng định nghĩa metric;
- thực hiện Wilcoxon hoặc bootstrap paired theo record khi tuyên bố khác biệt;
- không trộn kết quả 1-channel cũ với kết quả 12-lead mới nếu chưa chuẩn hóa protocol.

## 8. Cấu trúc output mỗi run

```text
/kaggle/working/experiments/<experiment_id>/
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

`model.pth` là weights EMA tốt nhất dùng cho inference. `last_resume.pth` chứa optimizer/scheduler/RNG để tiếp tục training; không dùng nó như artifact triển khai cuối.

## 9. Lịch triển khai đề xuất

| Mốc | Công việc | Điều kiện hoàn tất |
|---|---|---|
| M0 | Sửa leakage, metadata, metric và pin config | unit/data audit pass |
| M1 | Build Kaggle Dataset v1 | manifest + hashes + audit report |
| M2 | Smoke test 2–3 epoch | run-all và resume pass |
| M3 | Pilot 10–20 epoch | có throughput/VRAM và lịch session |
| M4 | Full mixed-model training | best checkpoint ổn định |
| M5 | Final test 1-shot | bảng metric theo artifact/SNR |
| M6 | Replicate fold/seed 2 | mean ± std và paired statistics |
| M7 | Specialized models/ablation tùy chọn | trả lời câu hỏi nghiên cứu bổ sung |

## 10. Tiêu chí hoàn thành

Quy trình chỉ được xem là hoàn tất khi:

- một Kaggle version sạch có thể tái tạo training từ input đã pin;
- không có leakage record/noise theo audit;
- checkpoint resume đã được thử thực tế qua hai phiên;
- best checkpoint được chọn hoàn toàn bằng validation;
- test report chứa metric theo artifact, SNR và record-level uncertainty;
- mọi kết quả trỏ được về data version, code commit, config và checkpoint tương ứng;
- output cuối được tải xuống hoặc sao lưu sang Kaggle Dataset/Model ngoài notebook output.

## 11. Việc cần làm ngay

1. Sửa synthesizer để nhận `split` và lưu metadata nguồn nhiễu.
2. Tạo notebook/data-build riêng và xuất Kaggle Dataset v1.
3. Chuyển training notebook sang đọc dataset đã build thay vì tải và regenerate mỗi phiên.
4. Thêm smoke-test mode, AMP tùy chọn, epoch timing, VRAM logging và early stopping.
5. Chạy smoke test, sau đó pilot; chỉ chốt số epoch/session sau khi có số đo.
