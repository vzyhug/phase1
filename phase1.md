# Báo cáo Chi tiết Hệ thống: Kiến trúc Mô hình, Phương pháp và Kỹ thuật (Phase 1)

Dự án này là một hệ thống **Denoising Diffusion Probabilistic Model (DDPM) có điều kiện (Conditional DDPM)** kết hợp đa miền (thời gian và tần số) được sử dụng để khử nhiễu (denoising) cho tín hiệu đa kênh (ví dụ: tín hiệu điện tim ECG 12 đạo trình).

Dưới đây là tài liệu tổng hợp và hệ thống hóa chi tiết về cấu trúc dự án, kiến trúc mô hình, phương pháp huấn luyện và kỹ thuật sử dụng.

---

## 1. Kiến trúc Mô hình (Model Architecture)

Hệ thống được thiết kế theo dạng **Conditional Diffusion Model**, trong đó mô hình học quá trình khử nhiễu từ tín hiệu nhiễu hoàn toàn ($x_T$) về tín hiệu sạch ($x_0$), dưới sự điều kiện hóa của tín hiệu quan sát bị nhiễu ($y$).

Mô hình học sâu cốt lõi (Base Model) chịu trách nhiệm dự đoán nhiễu (noise) ở mỗi bước timestep là một kiến trúc **1D U-Net** được tùy chỉnh kết hợp với các khối chú ý và điều chế đặc trưng.

### 1.1. Cấu trúc tổng thể của 1D U-Net (`UNet1D`)
Mô hình `UNet1D` bao gồm 4 cấp độ (levels) Encoder và 4 cấp độ Decoder với cấu trúc đối xứng và có các kết nối tắt (skip connections).
- **Đầu vào (Input)**: 24 channels. Đây là kết quả của việc ghép nối dọc theo trục channel (concatenate) giữa trạng thái nhiễu ở bước $t$ là $x_t$ (12 channels) và tín hiệu quan sát bị nhiễu $y$ (12 channels).
- **Đầu ra (Output)**: 12 channels (tín hiệu dự đoán).
- **Encoder**: 4 levels tăng dần số lượng feature maps (64 $\rightarrow$ 128 $\rightarrow$ 256 $\rightarrow$ 512). Tại mỗi level, luồng dữ liệu đi qua một khối HNFBlock, khối BridgeBlock, và một lớp MaxPool/Conv strided để giảm chiều kích thước không gian (downsampling).
- **Bottleneck**: Tại vị trí hẹp nhất (level 4), mô hình sử dụng một khối **1D Self-Attention** để nắm bắt các phụ thuộc xa (long-range dependencies) trên toàn bộ chuỗi tín hiệu.
- **Decoder**: Thực hiện upsampling thông qua ConvTranspose1d, ghép nối với feature map tương ứng từ Encoder (skip connection), và đi qua khối HNFBlock.

### 1.2. Các Khối Thành Phần Cốt Lõi (Core Blocks)

#### A. Khối HNF (Half Normalized Filter Block - `HNFBlock`)
Khối HNF được thiết kế để trích xuất đặc trưng đa tỷ lệ (multi-scale) và chuẩn hóa đặc trưng một cách hiệu quả.
1. **Multi-scale Convolutions**: Dữ liệu đầu vào đi qua một tập hợp các lớp 1D Convolution song song với các kích thước kernel khác nhau: 3, 5, 9, và 15. Điều này giúp mô hình nắm bắt được các đặc trưng tín hiệu có tần số và chu kỳ khác nhau.
2. **Aggregation**: Kết quả từ các nhánh multi-scale được nối lại (concatenate) và đi qua một lớp Conv1d 1x1 để hòa trộn đặc trưng.
3. **Half-Instance Normalization**: Trái với việc chuẩn hóa toàn bộ channels, mô hình chỉ thực hiện `InstanceNorm1d` trên **một nửa** số lượng channels (half). Nửa còn lại được giữ nguyên, sau đó 2 nửa này được ghép lại. Kỹ thuật này giúp mô hình giữ lại được các thông tin ngữ cảnh không gian/thời gian tĩnh không bị mất đi do chuẩn hóa, vừa tận dụng được khả năng ổn định của normalization.
4. **Residual Connection**: Thêm đầu vào gốc vào đầu ra cuối cùng (cộng residual).

#### B. Khối Bridge (FiLM-based Timestep Injection - `BridgeBlock`)
Trong Diffusion Models, mô hình cần biết nó đang ở bước khử nhiễu thứ $t$ nào. Khối này thực hiện điều kiện hóa mô hình theo thời gian (noise level $\sqrt{\bar{\alpha}}$) sử dụng kỹ thuật **FiLM (Feature-wise Linear Modulation)**.
- **Sinusoidal Embedding**: Biến đổi đại lượng vô hướng (mức độ nhiễu/timestep) thành một vector nhúng (embedding) có độ dài 128 (emb_dim) thông qua các hàm sin, cos.
- **Modulation**: Mạng tuyến tính biến đổi embedding này thành hệ số tỷ lệ (`scale`) và dịch chuyển (`shift`). Đầu ra của HNFBlock được nhân với `(1 + scale)` và cộng với `shift`. Kỹ thuật này hiệu quả hơn việc chỉ đơn thuần cộng vector thời gian vào đầu vào.

#### C. Khối Self-Attention 1D (`SelfAttention1D`)
Được đặt ở Bottleneck (mức sâu nhất của U-Net).
- Áp dụng cơ chế **Multi-Head Self-Attention** tiêu chuẩn cho dữ liệu chuỗi 1 chiều để mô hình có tầm nhìn toàn cục (global receptive field), cho phép dữ liệu từ một điểm thời gian có thể tương tác và lấy thông tin từ mọi điểm thời gian khác.

---

## 2. Kỹ Thuật Huấn Luyện (Training Techniques)

### 2.1. Hàm Mất Mát Đa Miền (Multi-domain Loss)
Điểm nổi bật của dự án này là thay vì chỉ tính toán sự sai khác của nhiễu trong miền thời gian (như DDPM chuẩn), mô hình kết hợp thêm miền tần số.
* **Thời gian (Time-domain Loss)**: Sử dụng L1 Loss giữa nhiễu thật (thêm vào) và nhiễu dự đoán bởi mô hình. L1 Loss giúp mô hình ít nhạy cảm hơn với các ngoại lai (outliers) so với L2.
* **Tần số (Frequency-domain Loss)**: Sử dụng MSE Loss trên biên độ STFT (Short-Time Fourier Transform). Mô hình từ nhiễu dự đoán sẽ tính ngược ra tín hiệu gốc dự đoán ($\hat{x}_0$). Sau đó, tính STFT cho $\hat{x}_0$ và tín hiệu sạch thật ($x_0$), sau đó tính chênh lệch độ lớn (magnitude). Điều này ép mô hình tạo ra tín hiệu không chỉ đúng dạng sóng mà còn đúng phổ tần số.
* **Tổng hợp**: $\mathcal{L}_{total} = \lambda_{time} \mathcal{L}_{L1} + \lambda_{freq} \mathcal{L}_{STFT}$ (mặc định $\lambda_{time} = 1.0$, $\lambda_{freq} = 0.1$).

### 2.2. Trượt Trung Bình Mũ (EMA - Exponential Moving Average)
- Để tăng cường sự ổn định và hiệu suất tổng quát hóa trong quá trình test/inference, dự án sử dụng EMA cho các tham số trọng số của mô hình với tỷ lệ cập nhật $\mu = 0.9$ (hoặc 0.999). 
- Nghĩa là trọng số thực tế dùng để đánh giá là trung bình động có trọng số của các trọng số học được trong quá khứ, giúp giảm độ nhiễu của gradient trong quá trình tối ưu.

### 2.3. Tối ưu hóa
- **Optimizer**: Sử dụng Adam Optimizer.
- **Scheduler**: StepLR giảm tốc độ học (learning rate) sau mỗi số lượng epochs nhất định (vd: 150 epochs).
- **Gradient Clipping**: Giới hạn chuẩn của gradient (clip_grad_norm_ = 1.0) để ngăn chặn hiện tượng exploding gradients.

---

## 3. Quy trình Dữ liệu (Data Preparation & Synthesis)

Dự án có luồng xử lý và tổng hợp dữ liệu nhiễu nhân tạo một cách bài bản:
1. **Nguồn nhiễu (Noise Types)**: Sử dụng cơ sở dữ liệu MIT-BIH NST bao gồm: nhiễu đường nền (bw - baseline wander), nhiễu điện cực chuyển động (em - electrode motion), nhiễu cơ bắp (ma - muscle artifact).
2. **Tiền xử lý (Resampling & Segmentation)**: Các tín hiệu (sạch và nhiễu) được nội suy (resample) về cùng tần số lấy mẫu (500Hz). Tín hiệu được cắt thành các đoạn nhỏ (segments) có độ dài 512 mẫu (khoảng 1 giây).
3. **Tổng hợp tự động (Synthesizer)**:
   - Các đoạn tín hiệu nhiễu được chọn ngẫu nhiên.
   - Biên độ nhiễu được tinh chỉnh tỷ lệ (scale) dựa trên Năng lượng (Power) của tín hiệu sạch (tức là dựa trên tỷ số SNR).
   - Thêm một hệ số biến động ngẫu nhiên (delta từ 0.2 đến 2.0) để tạo ra độ đa dạng cho tập nhiễu.
   - Sinh ra 3 bản sao nhiễu cho mỗi đoạn tín hiệu gốc để gia tăng dữ liệu (Data Augmentation).

---

## 4. Kỹ thuật Inference (Suy luận và Lấy mẫu)

### 4.1. DDIM Sampler (Denoising Diffusion Implicit Models)
- Trong Diffusion thông thường (DDPM), quá trình lấy mẫu mất rất nhiều bước (vd: 1000 bước) tương ứng với lúc huấn luyện.
- Để tăng tốc độ thực thi, dự án triển khai bộ lấy mẫu **DDIM** (`DDIMDenoiser`), có khả năng nhảy bước (skip steps) thông qua một chuỗi con determinisitic. Trong code, số bước DDIM (`ddim_steps`) được rút ngắn xuống còn 50 hoặc 15 bước, giúp suy luận theo thời gian thực (real-time) tốt hơn rất nhiều.

### 4.2. Metric Đánh giá
Sử dụng đa dạng các chỉ số kỹ thuật và lâm sàng để đánh giá mức độ khôi phục tín hiệu:
- **SNR Improvement**: Độ cải thiện Tỷ số Tín hiệu trên Nhiễu (dB).
- **SSD (Sum of Squared Differences)**.
- **MAD (Maximum Absolute Difference)**.
- **PRD (Percent Root-mean-square Difference)**: Thước đo quan trọng trong xử lý ECG.
- **Cosine Similarity**: Đánh giá sự tương đồng về góc/hình dạng của dạng sóng.
    