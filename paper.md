# Quy trình huấn luyện và đánh giá DDPM khử nhiễu ECG

Tài liệu này mô tả pipeline trong repository **Score-based-ECG-Denoising** theo dạng có thể tái sử dụng cho nghiên cứu khử nhiễu tín hiệu sinh học. Code hiện tại xử lý **baseline wander (BW)**; các nhiễu electrode motion và muscle artifact của NSTDB chưa được đưa vào pipeline.

## 1. Bài toán

Với một đoạn ECG sạch \(x\in\mathbb{R}^{512}\), tạo ECG nhiễu:

\[
y=x+\alpha n,
\]

trong đó \(n\) là baseline-wander dài 512 mẫu. Mục tiêu là phục hồi \(\hat{x}\) từ \(y\).

Mô hình không hồi quy trực tiếp \(y\rightarrow x\). Nó học dự đoán nhiễu Gaussian được thêm vào ECG sạch ở một mức diffusion ngẫu nhiên; ECG nhiễu \(y\) là điều kiện cho mô hình trong cả train lẫn suy luận.

```text
x (ECG sạch) -- forward diffusion --> x_t --┐
                                             ├-- mạng dự đoán epsilon
y (ECG có BW) -------------------------------┘

y (ECG có BW) + Gaussian ban đầu -- reverse diffusion --> x_hat
```

## 2. Chuẩn bị dữ liệu

### 2.1 ECG sạch

Nguồn ECG sạch là QT Database. `Data_Preparation/Prepare_QTDatabase.py` thực hiện:

1. Đọc ECG cùng annotation `pu1`.
2. Xác định điểm bắt đầu sóng P, lùi thêm 40 ms, và cắt đoạn giữa hai P-start liên tiếp.
3. Chỉ giữ đoạn không có hơn một R-peak ở giữa.
4. Resample từng đoạn về 360 Hz; dùng padding phản xạ trước resample để hạn chế artefact biên.
5. Loại đoạn dài hơn 496 mẫu; trừ baseline bằng trung bình hai đầu đoạn; zero-pad vào vector 512 mẫu từ vị trí 16.

Test được tách theo record, không theo beat. Các record test là `sel123`, `sel233`, `sel302`, `sel307`, `sel820`, `sel853`, `sel16420`, `sel16795`, `sele0106`, `sele0121`, `sel32`, `sel49`, `sel14046`, `sel15814`; record còn lại cung cấp train/validation.

Đối với nghiên cứu mới, phải tách train/validation/test theo patient hoặc record **trước** khi tạo sample và thêm nhiễu. Công bố danh sách record mỗi tập; không để các cửa sổ chồng lấp từ cùng tín hiệu xuất hiện ở nhiều tập.

### 2.2 Nguồn noise và protocol

Baseline wander lấy từ record `bw` của MIT-BIH Noise Stress Test Database. Có hai protocol độc lập:

| Protocol | Nhiễu train | Nhiễu test |
|---|---|---|
| `n_type=1` | nửa đầu channel 1 | nửa sau channel 2 |
| `n_type=2` | nửa đầu channel 2 | nửa sau channel 1 |

Mỗi protocol tạo một checkpoint riêng. Việc tách cả channel lẫn đoạn noise giúp tránh train và test trên cùng nguồn nhiễu. Với nghiên cứu mới, có thể thay bằng K-fold theo record noise, nhiều loại noise, hoặc test chéo database, nhưng không được dùng lại đúng noise segment đã thấy trong train.

### 2.3 Cách thêm noise

Với mỗi beat \(x\), lấy một đoạn BW liên tiếp \(n\) dài 512 mẫu. Code chọn:

\[
r\sim\operatorname{Uniform}\{0.20,0.21,\ldots,1.99\}.
\]

Gọi \(A_x=\max(x)-\min(x)\), \(A_n=\max(n)-\min(n)\), rồi đặt:

\[
\alpha=r\frac{A_x}{A_n}.
\]

Như vậy:

\[
\frac{\operatorname{range}(\alpha n)}{\operatorname{range}(x)}=r.
\]

Tín hiệu nhiễu là \(y=x+\alpha n\). `rnd_test.npy` lưu các giá trị \(r\) của test set để phân tích theo cường độ noise. Code đặt `numpy.random.seed(1234)` ở đầu `Data_Preparation`, nên việc sinh corruption có thể lặp lại khi dữ liệu và thứ tự record không đổi.

Khi tái sử dụng, nên lưu một manifest cho từng sample: ID record/patient, split, window index, noise segment, noise type, \(r\), seed và preprocessing version. Chỉ lưu `rnd_test.npy` là chưa đủ để tái lập toàn bộ dữ liệu.

## 3. Chia dữ liệu và định dạng tensor

Sau khi tạo cặp `(clean, noisy)`:

- Các beat từ record train được chia 70% train và 30% validation bằng `train_test_split`.
- Test là các record giữ riêng từ đầu.
- Dữ liệu `(N, 512, 1)` được đổi thành `(N, 1, 512)` cho Conv1D.
- Dataset giữ thứ tự `TensorDataset(clean, noisy)`.

Code hiện không gán `random_state` cho train/validation split. Để tái lập, hãy đặt seed cho Python, NumPy, PyTorch CPU/CUDA và dùng `random_state` cố định.

## 4. Kiến trúc conditional diffusion

`ConditionalModel` gồm hai stream Conv1D song song cho \(x_t\) và ECG điều kiện \(y\). Mỗi stream có convolution đầu vào và các HNF block với dilation `1, 2, 4, 2, 1`. Mỗi HNF block dùng kernel đa tỉ lệ 3, 5, 9, 15, residual connection và LeakyReLU. Mức noise diffusion được biểu diễn bằng sinusoidal positional encoding, sau đó đưa vào các bridge qua Feature-Wise Affine modulation. Conv1D cuối cho ra một kênh dự đoán \(\hat{\epsilon}\).

Cấu hình mặc định dùng 80 feature channels. Khi áp dụng cho tín hiệu khác, cần kiểm tra cửa sổ đầu vào và receptive field có bao phủ đặc trưng cần giữ lại hay không.

## 5. Huấn luyện conditional DDPM

### 5.1 Forward diffusion

Cấu hình mặc định:

| Tham số | Giá trị |
|---|---:|
| Số bước diffusion | 50 |
| `beta_start` | 0.0001 |
| `beta_end` | 0.5 |
| Beta schedule | Quadratic |
| Feature channels | 80 |

Đặt \(\alpha_t=1-\beta_t\), \(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\). Với \(\epsilon\sim\mathcal{N}(0,I)\):

\[
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon.
\]

Mỗi batch chọn một bước \(t\in[1,50]\); từng sample nhận một mức \(\sqrt{\bar\alpha}\) liên tục trong khoảng của bước đó.

### 5.2 Loss

Mạng nhận \((x_t,y,\sqrt{\bar\alpha})\), dự đoán \(\hat\epsilon_\theta\), và tối ưu:

\[
\mathcal{L}=\sum_{b,c,i}|\epsilon_{b,c,i}-\hat\epsilon_{\theta}(x_t,y,\sqrt{\bar\alpha})_{b,c,i}|.
\]

Đây là L1 `sum`, nên con số loss phụ thuộc batch size và window length. Nếu cần so sánh giữa thí nghiệm khác nhau, hãy dùng hoặc báo cáo thêm loss trung bình trên phần tử.

### 5.3 Tối ưu và checkpoint

| Thành phần | Giá trị |
|---|---:|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Batch size | 96 |
| Epoch | 400 |
| Gradient clipping | norm tối đa 1.0 |
| Scheduler | StepLR: mỗi 150 epoch, LR × 0.1 |
| Validation | mỗi epoch |

`model.pth` là checkpoint có validation loss nhỏ nhất. `final.pth` là checkpoint ở epoch cuối và không nhất thiết tốt nhất. Chỉ dùng validation để chọn checkpoint/hyperparameter; test chỉ dùng sau cùng.

```python
for epoch in range(num_epochs):
    for clean, noisy in train_loader:
        t = sample_diffusion_step()
        epsilon = randn_like(clean)
        alpha_bar = sample_continuous_alpha_bar(t, len(clean))
        x_t = sqrt(alpha_bar) * clean + sqrt(1 - alpha_bar) * epsilon
        epsilon_hat = model(x_t, noisy, sqrt(alpha_bar))
        loss = l1_sum(epsilon, epsilon_hat)
        optimize_with_gradient_clipping(loss)
    save_checkpoint_if_validation_loss_improves()
```

## 6. Suy luận và multi-shot

Để khử nhiễu ECG \(y\), reverse diffusion khởi tạo \(x_T\sim\mathcal{N}(0,I)\), rồi chạy ngược 50 bước. Ở mỗi bước, model dùng \(y\) làm điều kiện, dự đoán noise, ước lượng \(x_0\), rồi sample posterior để đi đến \(x_{t-1}\). Bước cuối không thêm Gaussian noise.

```python
x = randn_like(noisy)
for t in reversed(range(num_steps)):
    epsilon_hat = model(x, noisy, noise_level(t))
    x0_hat = predict_x0_from_noise(x, t, epsilon_hat)
    mean, log_variance = q_posterior(x0_hat, x, t)
    x = mean + exp(0.5 * log_variance) * randn_like(x) if t > 0 else mean
return x
```

Sampling là ngẫu nhiên. `eval_new.py` so sánh `shots = [1, 3, 5, 10]`. Với K-shot:

\[
\hat{x}_K=\frac{1}{K}\sum_{k=1}^{K}f_\theta^{(k)}(y).
\]

K-shot thường giảm phương sai đầu ra nhưng tăng gần K lần chi phí suy luận. Khi công bố, luôn ghi số step, K-shot, latency/beat, GPU/CPU và seed sampling.

## 7. Đánh giá qua `eval_new.py`

Với mỗi K trong `[1, 3, 5, 10]`, script:

1. Lặp `n_type=1` và `n_type=2`.
2. Load `check_points/noise_type_<n_type>/model.pth`.
3. Tạo lại test set tương ứng bằng `Data_Preparation(n_type)`.
4. Khử nhiễu theo batch 50.
5. Tính metric trên từng beat.
6. Gộp các beat từ hai protocol.
7. In `mean ± std` cho toàn bộ test và từng nhóm cường độ noise.

Đây là thống kê theo beat. Nếu mục tiêu là kết luận ở mức bệnh nhân/record, hãy tính trung bình theo record trước rồi báo cáo thống kê, CI hoặc kiểm định trên record/patient.

### 7.1 Metric

Với ECG sạch \(x\), ECG nhiễu \(y\), ECG phục hồi \(\hat{x}\):

| Metric | Công thức theo code | Tốt hơn khi |
|---|---|---|
| SSD | \(\sum_i(x_i-\hat{x}_i)^2\) | Thấp |
| MAD | \(\max_i|x_i-\hat{x}_i|\) | Thấp |
| PRD | \(100\sqrt{\frac{\sum_i(\hat{x}_i-x_i)^2}{\sum_i(\hat{x}_i-\bar{x})^2}}\) | Thấp |
| Cosine similarity | \(\frac{x^T\hat{x}}{\|x\|_2\|\hat{x}\|_2}\) | Cao, gần 1 |
| SNR-in | \(10\log_{10}\frac{\sum_i x_i^2}{\sum_i(y_i-x_i)^2}\) | Cao hơn = đầu vào sạch hơn |
| SNR-out | \(10\log_{10}\frac{\sum_i x_i^2}{\sum_i(\hat{x}_i-x_i)^2}\) | Cao |
| SNR improvement | \(SNR_{out}-SNR_{in}\) | Cao, dương |

PRD trong code dùng \(\hat{x}-\bar{x}\) ở mẫu số. Công thức PRD phổ biến hơn dùng \(x-\bar{x}\):

\[
PRD_{standard}=100\sqrt{\frac{\sum_i(\hat{x}_i-x_i)^2}{\sum_i(x_i-\bar{x})^2}}.
\]

Không được trộn hai định nghĩa này giữa các baseline. Hãy chọn một công thức và nêu rõ trong bài báo.

### 7.2 Phân tầng theo noise level

Code hiện dùng biên `[0.2, 0.6, 1.0, 1.5, 2.0]`. Đây là tỉ số peak-to-peak của noise sau scale với ECG sạch, không phải SNR dB. Để không đếm một sample ở hai nhóm, dùng:

```text
[0.2, 0.6)
[0.6, 1.0)
[1.0, 1.5)
[1.5, 2.0]
```

Code gốc dùng cả `>=` và `<=`, vì thế các mẫu tại 0.6, 1.0, 1.5 có thể bị tính hai lần.

## 8. Checklist tái sử dụng cho nghiên cứu khác

1. Chốt split patient/record trước preprocessing và thêm noise.
2. Lưu manifest dữ liệu, seed, code commit, config, checkpoint và phiên bản thư viện.
3. Tách độc lập nguồn/đoạn noise giữa train và test.
4. Dùng cùng preprocessing, input, target và test split cho mọi baseline.
5. Chọn checkpoint bằng validation, không bằng test.
6. Gọi `model.eval()` và `torch.no_grad()` khi suy luận.
7. Báo cáo K-shot, diffusion step, latency, mean ± std; tốt hơn nữa là 95% CI và kết quả theo record.
8. Đánh giá theo tổng thể, mức noise, loại artifact và nhiệm vụ downstream (ví dụ R-peak/QRS/morphology) nếu mục tiêu có ý nghĩa lâm sàng.
9. Nếu so sánh cải thiện nhỏ, lặp lại nhiều seed và báo cáo độ bất định giữa lần train/sampling.

## 9. Các lỗi/điểm cần chỉnh trong code gốc

1. `main_exp.py` hard-code `noise_type_1` khi load checkpoint tốt nhất sau train; khi train `--n_type=2` cần thay bằng `args.n_type`.
2. `train_test_split` chưa cố định seed.
3. `eval_new.py` nên gọi `model.eval()` sau khi load checkpoint.
4. Nên dùng `torch.load(output_path, map_location=device)` để hỗ trợ chạy trên thiết bị khác.
5. Sửa bin noise để không overlap ở biên.
6. Chọn và ghi rõ định nghĩa PRD.
7. Lưu ID sample, đầu ra denoised và metric từng sample vào CSV/Parquet thay vì chỉ `print`.

## 10. Lệnh chạy repository hiện tại

```powershell
python -W ignore main_exp.py --n_type=1
python -W ignore main_exp.py --n_type=2
python -W ignore eval_new.py
```

Cần đặt QT Database tại `data/qt-database-1.0.0/`, NSTDB tại `data/mit-bih-noise-stress-test-database-1.0.0/`, và có GPU ở `cuda:0` theo cấu hình hiện tại.
