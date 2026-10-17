# Đề xuất đánh giá ECG với ba loại nhiễu BW, EM và MA

## 1. Mục tiêu

Mở rộng thí nghiệm hiện tại từ một loại nhiễu `BW` sang đầy đủ ba loại nhiễu trong MIT-BIH Noise Stress Test Database (NSTDB):

| Ký hiệu | Tên đầy đủ | Ý nghĩa |
|---|---|---|
| `BW` | Baseline Wander | Trôi đường nền, thường có tần số thấp |
| `EM` | Electrode Motion Artifact | Nhiễu do điện cực dịch chuyển hoặc tiếp xúc không ổn định |
| `MA` | Muscle Artifact | Nhiễu do hoạt động cơ, có thể chồng lấn phổ tần với ECG |

Trong mã nguồn hiện tại, cả ba tín hiệu đã được đọc từ `NoiseBWL.pkl`, nhưng chỉ `BW` được sử dụng:

```python
[bw_signals, _, _] = nstdb
```

Vì vậy, `EM` và `MA` hiện chưa tham gia vào quá trình tạo dữ liệu train hoặc test.

## 2. Ý nghĩa của `noise_type_1` và `noise_type_2`

`noise_type_1` và `noise_type_2` trong project hiện tại không phải hai loại nhiễu khác nhau. Chúng là hai cách chia dữ liệu của cùng một loại nhiễu.

Mỗi bản ghi nhiễu có hai kênh. Mỗi kênh tiếp tục được chia thành hai nửa:

```text
channel 1 → channel1_a + channel1_b
channel 2 → channel2_a + channel2_b
```

Hai cách chia train/test hiện tại:

| Split | Noise train | Noise test |
|---|---|---|
| `noise_type_1` | Nửa đầu kênh 1 (`channel1_a`) | Nửa sau kênh 2 (`channel2_b`) |
| `noise_type_2` | Nửa đầu kênh 2 (`channel2_a`) | Nửa sau kênh 1 (`channel1_b`) |

Mục đích là tránh đánh giá model trên đúng đoạn và đúng kênh nhiễu đã dùng để huấn luyện. Tên dễ hiểu hơn cho hai trường hợp này là `split_1` và `split_2` hoặc `fold_1` và `fold_2`.

Khi mở rộng sang ba artifact, mỗi artifact vẫn nên giữ hai split:

```text
BW: split_1, split_2
EM: split_1, split_2
MA: split_1, split_2
```

## 3. Ba nhóm thí nghiệm được khuyến nghị

### 3.1. Thí nghiệm A — đánh giá đúng miền

Huấn luyện và kiểm tra model trên cùng loại artifact, nhưng noise train và noise test vẫn thuộc các kênh/đoạn khác nhau:

```text
BW-trained model → BW test
EM-trained model → EM test
MA-trained model → MA test
```

Mục đích:

- Đo hiệu năng tốt nhất model có thể đạt được trên từng loại nhiễu.
- So sánh độ khó giữa `BW`, `EM` và `MA`.
- Kiểm tra xem kiến trúc diffusion có phù hợp với từng loại artifact hay không.

Nếu huấn luyện riêng theo từng artifact và mỗi artifact có hai split, cần tổng cộng sáu checkpoint:

```text
check_points/
├── bw/
│   ├── split_1/model.pth
│   └── split_2/model.pth
├── em/
│   ├── split_1/model.pth
│   └── split_2/model.pth
└── ma/
    ├── split_1/model.pth
    └── split_2/model.pth
```

### 3.2. Thí nghiệm B — đánh giá chéo loại nhiễu

Mỗi model được đánh giá trên cả ba artifact:

| Model | BW test | EM test | MA test |
|---|---:|---:|---:|
| BW-trained | ✓ | ✓ | ✓ |
| EM-trained | ✓ | ✓ | ✓ |
| MA-trained | ✓ | ✓ | ✓ |

Mục đích:

- Đánh giá khả năng tổng quát hóa sang loại nhiễu chưa xuất hiện trong train.
- Xác định model có đang học đặc trưng ECG sạch hay chỉ học cách loại bỏ một kiểu artifact cụ thể.
- Xác định artifact nào gây suy giảm hiệu năng lớn nhất.

Ví dụ, dùng hai checkpoint hiện có để test thêm `EM` và `MA` sẽ trả lời câu hỏi:

> Model được huấn luyện bằng baseline wander có khử được electrode motion và muscle artifact hay không?

Trường hợp này không cần huấn luyện checkpoint mới, nhưng kết quả trên `EM` và `MA` có thể thấp vì đây là dữ liệu ngoài miền huấn luyện.

### 3.3. Thí nghiệm C — model huấn luyện bằng nhiễu hỗn hợp

Tạo dữ liệu train có chứa cả ba artifact:

```text
BW + EM + MA → Mixed-noise model
```

Nên đánh giá mixed model riêng trên từng tập test:

```text
Mixed model → BW test
Mixed model → EM test
Mixed model → MA test
```

Mục đích:

- Xây dựng một model phù hợp hơn với tín hiệu thực tế.
- Giảm phụ thuộc vào việc phải biết trước loại artifact tại thời điểm suy luận.
- Kiểm tra sự đánh đổi giữa hiệu năng chuyên biệt và độ ổn định trên nhiều loại nhiễu.

Mixed model có thể không đứng đầu trên từng artifact riêng, nhưng có khả năng ổn định hơn khi loại nhiễu đầu vào chưa biết trước hoặc khi nhiều artifact xuất hiện đồng thời.

## 4. Thiết kế thực nghiệm đề xuất

Nên tiến hành theo thứ tự sau.

### Giai đoạn 1 — dùng checkpoint hiện tại

Giữ nguyên hai model đã train bằng `BW` và đánh giá trên:

```text
BW test
EM test
MA test
```

Kết quả giai đoạn này tạo baseline về khả năng generalization mà chưa phải tốn thời gian train lại.

### Giai đoạn 2 — huấn luyện model chuyên biệt

Huấn luyện thêm:

```text
EM split 1 và split 2
MA split 1 và split 2
```

Sau đó lập bảng đánh giá đúng miền và bảng đánh giá chéo.

### Giai đoạn 3 — huấn luyện mixed model

Trộn `BW`, `EM` và `MA` trong tập train. Cần kiểm soát tỷ lệ lấy mẫu để một artifact không chiếm ưu thế chỉ vì có nhiều mẫu hơn.

Khuyến nghị ban đầu:

```text
BW : EM : MA = 1 : 1 : 1
```

Sau đó đánh giá mixed model trên từng artifact riêng biệt và so sánh với các model chuyên biệt.

## 5. Cách tổ chức bảng kết quả

### 5.1. Bảng đánh giá đúng miền

| Train noise | Test noise | SSD ↓ | MAD ↓ | PRD ↓ | Cosine similarity ↑ | SNR out ↑ | SNR improvement ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| BW | BW |  |  |  |  |  |  |
| EM | EM |  |  |  |  |  |  |
| MA | MA |  |  |  |  |  |  |

### 5.2. Bảng đánh giá chéo

Có thể lập một bảng cho mỗi metric. Ví dụ với `SNR improvement`:

| Model train | BW test | EM test | MA test | Trung bình |
|---|---:|---:|---:|---:|
| BW |  |  |  |  |
| EM |  |  |  |  |
| MA |  |  |  |  |
| Mixed |  |  |  |  |

Không nên chỉ nối toàn bộ kết quả `BW`, `EM`, `MA` rồi báo cáo một giá trị chung. Cách đó có thể che khuất việc model rất tốt trên một artifact nhưng rất kém trên artifact khác.

Nên báo cáo:

1. Metric riêng cho từng artifact.
2. Mean và standard deviation trên từng artifact.
3. Giá trị trung bình macro giữa ba artifact nếu cần một con số tổng quát.
4. Kết quả riêng cho `split_1` và `split_2`, sau đó mới báo cáo trung bình hai split.

## 6. Ý nghĩa của 1, 3, 5 và 10-shot

Trong project này, `shot` không phải few-shot learning. Nó là số lần chạy quá trình diffusion denoising trên cùng một đầu vào rồi lấy trung bình:

```text
1-shot  → denoise 1 lần
3-shot  → denoise 3 lần và lấy trung bình
5-shot  → denoise 5 lần và lấy trung bình
10-shot → denoise 10 lần và lấy trung bình
```

Nếu đầu ra của mỗi lần sampling là \(\hat{x}_i\), kết quả cuối với \(K\)-shot là:

$$
\hat{x}=\frac{1}{K}\sum_{i=1}^{K}\hat{x}_i
$$

Lấy trung bình nhiều lần có thể làm giảm dao động ngẫu nhiên của diffusion sampling, nhưng thời gian chạy tăng gần tuyến tính theo số shot.

Với một artifact, hai split và bốn mức shot, số lượt tương đối là:

$$
(1+3+5+10)\times2=38
$$

Với ba artifact:

$$
(1+3+5+10)\times2\times3=114
$$

Do đó, để kiểm tra pipeline trước, nên chạy `1-shot`. Chỉ chạy đủ `1, 3, 5, 10-shot` sau khi dữ liệu, checkpoint và metric đã được xác nhận chính xác.

## 7. Các nguyên tắc tránh data leakage

Khi thêm `EM` và `MA`, cần giữ các nguyên tắc sau:

- Không dùng cùng đoạn noise cho cả train và test.
- Không để hai kênh hoặc hai nửa đã dùng cho test xuất hiện trong train của cùng fold.
- Giữ nguyên danh sách ECG test giữa các artifact để phép so sánh công bằng.
- Giữ nguyên seed tạo mức nhiễu khi cần so sánh trực tiếp giữa các model.
- Không lựa chọn checkpoint dựa trên kết quả của test set.
- Khi huấn luyện mixed model, vẫn phải tách train/test độc lập cho từng artifact trước khi trộn.

## 8. Kiểm soát mức nhiễu

Code hiện tại tạo hệ số nhiễu ngẫu nhiên trong khoảng:

```python
0.2 <= noise level < 2.0
```

và chia kết quả thành các khoảng:

```text
0.2–0.6
0.6–1.0
1.0–1.5
1.5–2.0
```

Nên giữ cùng seed và cùng phân bố noise level cho `BW`, `EM`, `MA` để so sánh công bằng. Tuy nhiên, cùng một hệ số biên độ không đảm bảo ba artifact có cùng độ khó, vì đặc tính phổ tần và hình thái của chúng khác nhau.

Nếu cần so sánh chặt chẽ hơn, nên bổ sung thí nghiệm theo input SNR cố định, ví dụ:

```text
-5 dB, 0 dB, 5 dB, 10 dB
```

## 9. Khối lượng checkpoint và thời gian chạy

| Thiết kế | Số model/checkpoint theo hai split |
|---|---:|
| Chỉ BW | 2 |
| BW, EM, MA huấn luyện riêng | 6 |
| Một mixed model với hai split | 2 |
| Ba model riêng và một mixed model | 8 |

Đánh giá đủ ba artifact với `1, 3, 5, 10-shot` sẽ mất khoảng gấp ba lần đánh giá chỉ `BW`. Huấn luyện ba model chuyên biệt cũng có chi phí xấp xỉ gấp ba, chưa tính mixed model.

## 10. Khuyến nghị cuối cùng

Thiết kế nên dùng cho báo cáo hoặc khóa luận:

1. Giữ kết quả `BW-trained → BW test` làm baseline.
2. Dùng checkpoint hiện tại đánh giá thêm `BW-trained → EM test` và `BW-trained → MA test` để đo generalization.
3. Huấn luyện model chuyên biệt cho `EM` và `MA` với cùng cấu hình như `BW`.
4. Báo cáo ma trận đánh giá chéo giữa ba model và ba test artifact.
5. Huấn luyện thêm một mixed-noise model nếu mục tiêu là triển khai trên ECG thực tế.
6. Báo cáo kết quả riêng theo artifact, split, noise level và số shot.
7. Dùng `1-shot` để kiểm tra pipeline; chỉ chạy đủ `1, 3, 5, 10-shot` cho thí nghiệm chính thức.

Thiết kế này cho phép trả lời ba câu hỏi quan trọng:

- Model xử lý tốt nhất loại artifact nào?
- Model có tổng quát hóa sang artifact chưa gặp khi train không?
- Model chuyên biệt hay mixed-noise model phù hợp hơn cho ứng dụng thực tế?

