# Hệ thống nghiên cứu khử nhiễu ECG một đạo trình trên QTDB

## 1. Tổng quan

Nghiên cứu xây dựng hệ thống **Conditional Denoising Diffusion Probabilistic Model (Conditional DDPM)** để khử nhiễu ECG một đạo trình từ **QT Database (QTDB)**. QTDB là nguồn ECG duy nhất trong phạm vi train, validation, test và báo cáo kết quả. Mục tiêu là phục hồi tín hiệu sạch \(x_0\) từ tín hiệu quan sát bị nhiễu \(y\), đồng thời bảo toàn hình thái ECG trên cả miền thời gian và miền tần số.

Thay vì hồi quy trực tiếp \(y\rightarrow x_0\), mô hình nhận trạng thái đã khuếch tán \(x_t\), ECG nhiễu điều kiện \(y\) và mức diffusion \(\sqrt{\bar\alpha_t}\), sau đó dự đoán nhiễu Gaussian đã được thêm vào tín hiệu sạch. Quá trình đảo diffusion dùng dự đoán này để tái tạo ECG.

Các thành phần chính của nghiên cứu:

- common data ECG một đạo trình từ QTDB;
- benchmark chính QTDB–BW dài 3600 điểm ở 360 Hz;
- 1D U-Net với HNF đa tỉ lệ, FiLM và self-attention;
- loss kết hợp miền thời gian và miền tần số;
- suy luận bằng DDPM hoặc DDIM;
- đánh giá bằng các metric sai số, tương đồng hình thái và SNR.

## 2. Bài toán diffusion có điều kiện

Tín hiệu quan sát được tạo bởi:

\[
y=x_0+\alpha n,
\]

trong đó \(x_0\) là ECG sạch, \(n\) là artifact và \(\alpha\) điều khiển cường độ nhiễu.

Forward diffusion tạo tín hiệu ở bước \(t\):

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

Mạng học hàm:

\[
\hat\epsilon=\epsilon_\theta(x_t,y,\sqrt{\bar\alpha_t}),
\]

rồi dùng \(\hat\epsilon\) để ước lượng \(\hat x_0\) và thực hiện quá trình đảo từ nhiễu Gaussian về ECG đã khử nhiễu.

## 3. Dữ liệu và tiền xử lý

### 3.1. Common data từ QTDB

Benchmark chính dùng một common dataset đã được khóa và dùng chung cho cả năm mô hình. Notebook của từng mô hình không tự tải lại QTDB, không tự chia tập và không tự sinh nhiễu. Contract chính:

```text
clean_train/noisy_train: [N_train, 1, 3600]
clean_val/noisy_val:     [N_val,   1, 3600]
clean_test/noisy_test:   [N_test,  1, 3600]
sample_id_test:          [N_test]
```

Tín hiệu có một đạo trình, tần số 360 Hz và chiều dài 3600 điểm. Train chỉ dùng train split; validation dùng để chọn checkpoint; test không được dùng để chọn epoch hoặc siêu tham số.

Mỗi mẫu được chuẩn hóa bằng scale chỉ tính từ noisy input và dùng chung cho clean target:

\[
s_i=\max|y_i|,\qquad y_i'=y_i/s_i,\qquad x_i'=x_i/s_i.
\]

Prediction được nhân lại với \(s_i\) trước khi benchmark.

### 3.2. Nguồn nhiễu

Các nguồn artifact nghiên cứu gồm:

- `bw`: baseline wander;
- `em`: electrode motion;
- `ma`: muscle artifact.

Benchmark chính 3600 sử dụng **baseline wander (BW)** đã được tạo sẵn trong common data. Bốn mô hình đối chứng và Conditional DDPM phải nhận đúng cùng các cặp clean–noisy và cùng `sample_id`; không mô hình nào được tự tạo một phiên bản BW khác.

BW, EM và MA cũng được dùng trong nhánh thí nghiệm QTDB 512 điểm để nghiên cứu model chuyên biệt, zero-shot RMN và generalization chéo nhiễu. Nhánh 512 là thí nghiệm phụ và không được trộn số liệu trực tiếp với benchmark 3600.

## 4. Kiến trúc mô hình

### 4.1. 1D U-Net chính

Đầu vào mạng là phép nối theo chiều channel giữa:

- \(x_t\): trạng thái diffusion, 1 kênh;
- \(y\): ECG quan sát bị nhiễu, 1 kênh.

Do đó mô hình dùng `in_channels=2`, `out_channels=1`. Với `base_channels=80`, encoder sử dụng số đặc trưng `80 → 160 → 320 → 640`. Ba lần downsampling biến chiều dài `3600 → 1800 → 900 → 450`; ba lần upsampling cùng skip connection trả output về đúng 3600 điểm.

### 4.2. HNFBlock

HNFBlock trích xuất đặc trưng đa tỉ lệ qua bốn Conv1D song song có kernel 3, 5, 9 và 15. Các đặc trưng được nối lại, trộn bằng Conv1D 1×1, sau đó:

- một nửa số channel được `InstanceNorm1d`;
- nửa còn lại giữ nguyên thông tin biên độ;
- hai phần được ghép lại, kích hoạt ReLU và cộng residual.

Thiết kế này nhằm kết hợp độ ổn định của normalization với khả năng giữ đặc trưng hình thái ECG.

### 4.3. BridgeBlock và FiLM

Mức diffusion \(\sqrt{\bar\alpha_t}\) được mã hóa bằng sinusoidal embedding 128 chiều. Một mạng tuyến tính sinh hai vector `scale` và `shift`, rồi điều chế feature map:

\[
h'=h(1+scale)+shift.
\]

BridgeBlock được đặt ở từng mức encoder, giúp mạng biết mức nhiễu hiện tại của quá trình diffusion.

### 4.4. Self-attention

Tại bottleneck, mô hình dùng self-attention 1D với bốn head. Cơ chế này cho phép mỗi vị trí thời gian kết hợp thông tin từ cửa sổ ECG 10 giây, hỗ trợ học quan hệ dài hạn giữa các thành phần P–QRS–T.

### 4.5. Mô hình thay thế

Repo còn cung cấp `ConditionalModel` để đối chứng. Mô hình này dùng hai stream Conv1D riêng cho \(x_t\) và \(y\), năm HNFBlock với dilation `1, 2, 4, 2, 1`, cùng các bridge điều kiện hóa theo timestep. Có thể chọn mô hình khi huấn luyện bằng `--model 1` hoặc `--model 2`.

## 5. Huấn luyện

Cấu hình trong `configs/base.yaml`:

| Thành phần | Giá trị |
|---|---:|
| Epoch | 400 |
| Physical batch size mặc định | 4 |
| Learning rate | `1e-4` |
| Feature cơ sở | 80 |
| Diffusion steps | 50 |
| `beta_start` | `0.0001` |
| `beta_end` | `0.5` |
| Beta schedule | Quadratic |

### 5.1. Loss đa miền

Loss tổng gồm hai thành phần:

\[
\mathcal L=\lambda_{time}\mathcal L_{L1}
+\lambda_{freq}\mathcal L_{STFT},
\]

với mặc định \(\lambda_{time}=1\) và \(\lambda_{freq}=0.1\).

- **Time-domain loss:** L1 giữa nhiễu Gaussian thật \(\epsilon\) và nhiễu dự đoán \(\hat\epsilon\).
- **Frequency-domain loss:** MSE giữa biên độ STFT của ECG sạch và ECG được ước lượng từ \(\hat\epsilon\), với `n_fft=128`, `hop_length=64`.

Loss thời gian buộc mạng dự đoán đúng noise, còn loss tần số hỗ trợ giữ cấu trúc phổ của ECG.

### 5.2. Tối ưu và validation

- Optimizer: Adam.
- Gradient clipping: norm tối đa 1.0.
- EMA: cập nhật shadow weights với \(\mu=0.9\).
- Validation chạy mỗi epoch và chỉ validation loss được dùng để chọn checkpoint; test metric không tham gia chọn model.
- `last_proposed_conditional_ddpm_3600.pth` dùng để resume; `best_proposed_conditional_ddpm_3600.pth` lưu epoch có validation loss thấp nhất.
- EMA weights, optimizer, scheduler, AMP scaler, RNG state, source hash và common-data fingerprint đều được lưu trong checkpoint.
- Log được lưu tại `training_proposed_conditional_ddpm_3600.csv`.
- Loss L1 của mã nguồn dùng `reduction='sum'`, nên raw loss tỷ lệ với batch size và chiều dài 3600; không được so trị tuyệt đối với loss của thí nghiệm 512 điểm.

## 6. Suy luận

Hệ thống hỗ trợ hai phương pháp:

- **DDPM:** chạy toàn bộ quá trình đảo có tính ngẫu nhiên;
- **DDIM:** lấy mẫu theo chuỗi timestep rút gọn để giảm thời gian suy luận.

Luồng inference mặc định dùng DDIM 50 bước, `eta=0`. Quá trình bắt đầu từ Gaussian noise có cùng kích thước với ECG đầu vào, sau đó lặp dự đoán \(\epsilon\) và cập nhật \(x_t\) cho đến khi thu được \(\hat x_0\).

Kết quả chính của benchmark dùng **DDIM 1-shot**, 50 bước, `eta=0`, seed 2024 và EMA weights. Multi-shot chỉ được xem là thí nghiệm khác và không được ghi đè output 1-shot.

Notebook `kaggle_proposed_conditional_ddpm_3600.ipynb` xuất file đóng góp của mô hình:

```text
proposed_seed2024.npz
```

gồm `sample_id`, `prediction`, `latency_ms` và `normalization_scale`. Prediction có shape `[N_test,1,3600]` và đã được inverse-normalize.

## 7. Đánh giá

Các metric đã cài đặt:

| Metric | Ý nghĩa | Tốt hơn khi |
|---|---|---|
| SSD | Tổng bình phương sai khác | Thấp |
| MAD | Sai khác tuyệt đối lớn nhất | Thấp |
| PRD | Phần trăm sai khác RMS | Thấp |
| Cosine similarity | Tương đồng hình dạng tín hiệu | Gần 1 |
| RMSE | Sai số RMS | Thấp |
| SNR-out | Chất lượng tín hiệu phục hồi | Cao |
| SNR improvement | `SNR-out − SNR-in` | Dương và cao |

Notebook huấn luyện/export không tính metric. Benchmark chung nhận năm file prediction, căn theo `sample_id`, rồi tính lại metric trên đúng cùng `clean_test/noisy_test`. Checkpoint `.pth` không phải đầu vào trực tiếp của bảng benchmark.

Các kết quả QTDB 512 điểm trong `src/results/` hoặc benchmark BW/EM/MA là bằng chứng cho nhánh thí nghiệm phụ; chúng không thay thế kết quả benchmark 5 mô hình trên common data 3600.

## 8. Luồng hoạt động

```text
Common QTDB–BW: [N,1,3600], 360 Hz
                   │
        train / validation / test cố định
                   │
        noisy-derived shared normalization
                   │
        Conditional DDPM + HNF U-Net
                   │
          train/resume đến 400 epoch
                   │
       validation-best EMA checkpoint
                   │
           DDIM 50-step, 1-shot
                   │
          proposed_seed2024.npz
                   │
      benchmark chung với 4 mô hình khác
```

Notebook chính:

```text
notebooks/kaggle_proposed_conditional_ddpm_3600.ipynb
```

## 9. Hiện trạng và hạn chế cần xử lý

1. **Một đạo trình:** kết quả không được diễn giải như mô hình ECG đa đạo trình.
2. **Nhiễu tổng hợp:** hiệu năng QTDB–BW không tự động đại diện cho mọi nhiễu thực tế, thiết bị hoặc bệnh viện.
3. **Dữ liệu chung là nguồn chuẩn:** mọi mô hình phải dùng đúng common NPZ/manifest; tự tạo split hoặc nhiễu riêng làm mất tính công bằng.
4. **Không test leakage:** checkpoint chỉ được chọn bằng validation loss; test chỉ được mở sau khi khóa model.
5. **Loss khác thang đo:** L1 dùng `reduction='sum'`, còn STFT dùng mean. Raw loss không phải MAE và không dùng để so giữa chiều dài/batch size khác nhau.
6. **Alpha trong STFT loss:** cần tiếp tục kiểm toán tính nhất quán giữa alpha liên tục dùng tạo \(x_t\) và alpha dùng khôi phục \(\hat x_0\).
7. **Metric thống nhất:** PRD, SNR và mọi metric phải được tính lại trong notebook benchmark chung, không lấy trực tiếp từ log riêng của từng mô hình.
8. **Tái lập:** phải lưu data fingerprint, manifest, source hash, checkpoint hash, seed, cấu hình sampler và thứ tự sample ID.
9. **Nhiều seed:** một lần train với seed 2024 chưa phản ánh bất định giữa nhiều lần huấn luyện.
10. **Hai nhánh khác nhau:** kết quả QTDB 512 BW/EM/MA phải báo cáo riêng, không gộp với benchmark 3600.

## 10. Kết luận

Nghiên cứu chính là hệ thống khử nhiễu **ECG QTDB một đạo trình** bằng Conditional DDPM. Benchmark chính sử dụng common QTDB–BW data dài 3600 điểm ở 360 Hz để bảo đảm năm mô hình học và được chấm trên cùng dữ liệu. Đóng góp kỹ thuật nằm ở 1D U-Net kết hợp HNF đa tỉ lệ, FiLM theo mức diffusion, self-attention, loss thời gian–tần số, EMA và DDIM.

Kết luận thực nghiệm cuối cùng chỉ được đưa ra sau khi `proposed_seed2024.npz` và bốn prediction file còn lại được đánh giá bằng cùng notebook benchmark, cùng `sample_id` và cùng định nghĩa metric. Nhánh QTDB 512 điểm cho BW/EM/MA là thí nghiệm bổ sung và phải được báo cáo riêng.
