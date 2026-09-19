# Hệ thống nghiên cứu khử nhiễu ECG 12 đạo trình trên LUDB

## 1. Tổng quan

Nghiên cứu xây dựng hệ thống **Conditional Denoising Diffusion Probabilistic Model (Conditional DDPM)** để khử nhiễu ECG 12 đạo trình từ cơ sở dữ liệu **LUDB**. Mục tiêu là phục hồi tín hiệu sạch \(x_0\) từ tín hiệu quan sát bị nhiễu \(y\), đồng thời bảo toàn hình thái sóng ECG trên cả miền thời gian và miền tần số.

Thay vì hồi quy trực tiếp \(y\rightarrow x_0\), mô hình nhận trạng thái đã khuếch tán \(x_t\), ECG nhiễu điều kiện \(y\) và mức diffusion \(\sqrt{\bar\alpha_t}\), sau đó dự đoán nhiễu Gaussian đã được thêm vào tín hiệu sạch. Quá trình đảo diffusion dùng dự đoán này để tái tạo ECG.

Các thành phần chính của nghiên cứu:

- dữ liệu ECG 12 đạo trình từ LUDB;
- tổng hợp ba loại artifact từ MIT-BIH Noise Stress Test Database;
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

### 3.1. ECG sạch từ LUDB

- Đọc các record LUDB bằng thư viện `wfdb`.
- Giữ toàn bộ **12 đạo trình**.
- Resample tín hiệu về **500 Hz** nếu tần số gốc khác 500 Hz.
- Cắt mỗi record thành các cửa sổ dài **512 mẫu**, tương đương khoảng 1,024 giây.
- Dùng stride 32 cho record train và stride 128 cho record test.
- Chuẩn hóa từng đạo trình trong mỗi cửa sổ bằng biên độ tuyệt đối cực đại:

\[
x_{norm}=\frac{x}{\max |x|}.
\]

### 3.2. Nguồn nhiễu

Artifact được lấy từ **MIT-BIH Noise Stress Test Database**, gồm:

- `bw`: baseline wander;
- `em`: electrode motion;
- `ma`: muscle artifact.

Các tín hiệu nhiễu được resample về 500 Hz để đồng bộ với LUDB. Với mỗi cửa sổ ECG sạch, hệ thống:

1. chọn ngẫu nhiên một loại artifact;
2. lấy một noise patch liên tiếp dài 512 mẫu;
3. cân bằng công suất noise theo công suất của ECG sạch;
4. nhân thêm hệ số \(\delta\sim U(0.2,2.0)\);
5. cộng noise vào ECG sạch;
6. tạo ba phiên bản nhiễu cho mỗi cửa sổ để tăng dữ liệu.

Công thức scale hiện tại:

\[
s=\sqrt{\frac{P_{clean}}{P_{noise}}},
\qquad y=x_0+\delta s n.
\]

Một noise patch được broadcast cho cả 12 đạo trình. Pipeline lưu `clean.npy`, `noisy.npy` và `deltas.npy` trong `data/synthesis/`.

## 4. Kiến trúc mô hình

### 4.1. 1D U-Net chính

Đầu vào mạng là phép nối theo chiều channel giữa:

- \(x_t\): trạng thái diffusion, 12 kênh;
- \(y\): ECG quan sát bị nhiễu, 12 kênh.

Do đó đầu vào có 24 kênh và đầu ra có 12 kênh dự đoán nhiễu. Với cấu hình mặc định `base_channels=80`, encoder sử dụng số đặc trưng `80 → 160 → 320 → 640`. Mạng có ba lần downsampling, ba lần upsampling và các skip connection giữa encoder–decoder.

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

Tại bottleneck, mô hình dùng self-attention 1D với bốn head. Cơ chế này cho phép mỗi vị trí thời gian kết hợp thông tin từ toàn bộ cửa sổ, hỗ trợ học quan hệ dài hạn giữa các thành phần P–QRS–T.

### 4.5. Mô hình thay thế

Repo còn cung cấp `ConditionalModel` để đối chứng. Mô hình này dùng hai stream Conv1D riêng cho \(x_t\) và \(y\), năm HNFBlock với dilation `1, 2, 4, 2, 1`, cùng các bridge điều kiện hóa theo timestep. Có thể chọn mô hình khi huấn luyện bằng `--model 1` hoặc `--model 2`.

## 5. Huấn luyện

Cấu hình trong `configs/base.yaml`:

| Thành phần | Giá trị |
|---|---:|
| Epoch | 400 |
| Batch size | 96 |
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
- Validation mặc định chạy mỗi epoch và dùng DDIM 15 bước để tính RMSE, SNR, PRD và cosine similarity.
- Checkpoint có validation loss thấp nhất được lưu thành `checkpoints/model.pth`; epoch cuối được lưu thành `final.pth`.
- Log huấn luyện dự kiến được lưu tại `checkpoints/training_log.csv`.

## 6. Suy luận

Hệ thống hỗ trợ hai phương pháp:

- **DDPM:** chạy toàn bộ quá trình đảo có tính ngẫu nhiên;
- **DDIM:** lấy mẫu theo chuỗi timestep rút gọn để giảm thời gian suy luận.

Luồng inference mặc định dùng DDIM 50 bước, `eta=0`. Quá trình bắt đầu từ Gaussian noise có cùng kích thước với ECG đầu vào, sau đó lặp dự đoán \(\epsilon\) và cập nhật \(x_t\) cho đến khi thu được \(\hat x_0\).

Hệ thống cũng hỗ trợ **multi-shot**: chạy nhiều lần từ các Gaussian khởi tạo khác nhau rồi lấy trung bình. Cách này có thể giảm phương sai đầu ra nhưng làm chi phí suy luận tăng gần tuyến tính theo số shot.

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

`evaluate_and_plot.py` còn triển khai Butterworth, median filter và wavelet để so sánh với mô hình diffusion, đồng thời vẽ cả 12 đạo trình. Tuy nhiên, script hiện chỉ đánh giá mẫu đầu tiên và có fallback sang sóng sin nhân tạo khi thiếu dữ liệu; vì vậy chưa thể xem đây là benchmark LUDB chính thức.

Repo hiện **chưa có bảng kết quả đánh giá LUDB đầy đủ**. Các CSV trong `src/results/` không thuộc thí nghiệm LUDB chính nên không được sử dụng trong tài liệu này.

## 8. Luồng hoạt động

```text
LUDB raw
   │
   v
Resample 500 Hz ──> Segment 512 ──> Normalize
                                          │
NSTDB bw/em/ma ──> chọn patch + scale ─────┤
                                          v
                                  clean/noisy pairs
                                          │
                         Conditional DDPM + 1D U-Net
                                          │
                              DDPM/DDIM reverse process
                                          v
                         ECG 12 đạo trình đã khử nhiễu
                                          │
                         metrics + biểu đồ so sánh
```

Các lệnh chính:

```powershell
python run_workflow.py --mode preprocess
python run_workflow.py --mode train --model 1
python run_workflow.py --mode infer
python run_workflow.py --mode eval
```

## 9. Hiện trạng và hạn chế cần xử lý

1. **Split dữ liệu chưa nhất quán:** `preprocess` đánh dấu các record LUDB 180–200 làm test, nhưng `train_model` gọi lại `Data_Preparation` không truyền danh sách này. Hàm sau đó dùng danh sách tên record của nguồn dữ liệu khác, có thể khiến test LUDB rỗng và đưa record dự kiến làm test vào train.
2. **Nguy cơ leakage train/validation:** các cửa sổ chồng lấp được tạo trước, rồi mới chia train/validation ngẫu nhiên. Hai tập có thể chứa các đoạn gần như giống nhau từ cùng record.
3. **Tham số `n_type` không có tác dụng:** `Data_Preparation` nhận tham số này nhưng bộ tổng hợp hiện chọn ngẫu nhiên `bw`, `em`, `ma` và không dùng `n_type`.
4. **Scheduler không giảm learning rate:** `StepLR` đặt `gamma=1.0`, nên learning rate giữ nguyên dù tài liệu cũ mô tả có decay.
5. **EMA chưa tham gia inference:** shadow weights được cập nhật nhưng không được áp vào model validation, không được lưu riêng và không được nạp khi suy luận.
6. **Hai loss lệch thang đo:** L1 dùng `reduction='sum'`, còn STFT dùng mean; hệ số 0.1 chưa thể hiện rõ đóng góp thực tế của miền tần số.
7. **Alpha trong STFT loss chưa hoàn toàn nhất quán:** \(x_t\) dùng mức alpha liên tục được lấy mẫu trong khoảng timestep, nhưng \(\hat x_0\) lại dùng alpha rời rạc theo chỉ số \(t\).
8. **Noise đa đạo trình còn đơn giản:** cùng một noise patch được cộng vào cả 12 lead và scale theo công suất chung; điều này chưa mô phỏng đầy đủ sự khác biệt artifact giữa các điện cực.
9. **Thiếu khả năng tái lập:** chưa lưu manifest gồm record, split, vị trí/loại noise, delta, seed, phiên bản preprocessing và code commit.
10. **Định nghĩa PRD chưa chuẩn:** mẫu số hiện phụ thuộc tín hiệu dự đoán thay vì năng lượng tín hiệu sạch quanh giá trị trung bình, nên khó so sánh với nghiên cứu khác.
11. **Chưa có benchmark LUDB hoàn chỉnh:** thiếu kết quả trên toàn test set, baseline công bằng, phân tầng theo loại/cường độ nhiễu, latency, nhiều seed và thống kê theo record.
12. **Repo chưa tự chứa dữ liệu:** các thư mục dữ liệu đang trống và bị `.gitignore`; muốn chạy lại cần đặt LUDB và NSTDB đúng cấu trúc.

## 10. Kết luận

Nghiên cứu chính là hệ thống khử nhiễu **ECG LUDB 12 đạo trình** bằng Conditional DDPM. Đóng góp kỹ thuật nằm ở 1D U-Net kết hợp HNF đa tỉ lệ, FiLM theo mức diffusion, self-attention và loss thời gian–tần số; DDIM được dùng để giảm chi phí suy luận.

Pipeline đã có đủ các khối từ tiền xử lý đến inference, nhưng chưa sẵn sàng để đưa ra kết luận thực nghiệm đáng tin cậy. Ưu tiên tiếp theo là sửa split theo record, loại bỏ leakage, chuẩn hóa loss/EMA/scheduler/PRD, lưu manifest dữ liệu và thực hiện benchmark LUDB đầy đủ trên ba loại artifact.
