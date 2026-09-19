# Quy trình đánh giá tương đương cho nghiên cứu A về khử nhiễu ECG

Tài liệu này hệ thống hóa **quy trình đánh giá và báo cáo chỉ số** của dự án khử nhiễu ECG bằng diffusion, để một nghiên cứu A có thể triển khai phép đánh giá tương đương. Nội dung tập trung vào dữ liệu, điều kiện suy luận, công thức, cách tổng hợp và cách diễn giải kết quả. Nghiên cứu A có thể dùng bất kỳ kiến trúc mô hình nào nếu đầu ra là ECG đã khử nhiễu và tuân thủ cùng protocol đánh giá.

## 1. Chốt mục tiêu so sánh trước khi thực hiện

Cần phân biệt hai mức tương đương:

1. **Tái hiện cùng benchmark:** dùng cùng ECG sạch, cùng nhiễu, cùng cách chia tập, cùng mức nhiễu và cùng công thức chỉ số. Khi đó số của nghiên cứu A có thể đặt cạnh số của dự án trên một tập test chung.
2. **Áp dụng cùng phương pháp đánh giá trên dữ liệu mới:** dùng cặp ECG sạch–nhiễu của nghiên cứu A nhưng giữ cách suy luận, tính chỉ số và báo cáo. Khi đó chỉ có **phương pháp đánh giá** tương đương; không so trực tiếp giá trị chỉ số giữa hai tập dữ liệu như thể chúng là cùng benchmark.

Chốt trước khi xem kết quả test: đối tượng đánh giá, tập test, loại nhiễu, cách tạo nhiễu, các mô hình, số lần suy mẫu, công thức chỉ số, nhóm cường độ nhiễu và quy tắc xử lý giá trị không hữu hạn. Mọi mô hình được so sánh phải dùng cùng một tập cặp tín hiệu và cùng đơn vị biên độ khi chấm điểm.

## 2. Xác định đơn vị mẫu và các tín hiệu bắt buộc

Một đơn vị đánh giá là một đoạn ECG gồm ba tín hiệu căn chỉnh theo từng điểm thời gian:

- `x`: ECG sạch, dùng làm ground truth.
- `y`: ECG nhiễu đầu vào của mô hình.
- `x_hat`: ECG đã khử nhiễu do mô hình tạo ra.

Trong benchmark của dự án, mỗi đoạn có **một kênh và 512 điểm ở 360 Hz**. `x`, `y` và `x_hat` phải cùng chiều dài, tần số lấy mẫu, kênh, đơn vị và gốc thời gian. Sau khi suy luận, cả ba phải nằm trên **cùng thang biên độ** trước khi tính SSD, MAD, PRD hoặc SNR. Nếu mô hình A chuẩn hóa đầu vào, phải đảo phép chuẩn hóa của đầu ra một cách nhất quán.

Mỗi cặp cần ID ổn định, ID bệnh nhân hoặc record, loại nhiễu, mức nhiễu và protocol. Không thể tính bộ chỉ số tham chiếu dưới đây nếu chỉ có ECG nhiễu thật mà không có ECG sạch tương ứng. Trong trường hợp đó, nghiên cứu A cần một thiết kế đánh giá khác và phải nêu rõ không phải phép đánh giá tương đương.

## 3. Thiết kế tập dữ liệu trước khi huấn luyện hoặc chấm điểm

### 3.1 Chia ECG sạch

Benchmark dùng ECG từ QT Database. Tập test được giữ riêng theo **record**, gồm:

`sel123`, `sel233`, `sel302`, `sel307`, `sel820`, `sel853`, `sel16420`, `sel16795`, `sele0106`, `sele0121`, `sel32`, `sel49`, `sel14046`, `sel15814`.

Các record còn lại tạo nguồn train/validation. Trong quy trình huấn luyện gốc, các đoạn từ nguồn này được chia **70% train và 30% validation**. Nếu nghiên cứu A dùng dữ liệu riêng, ưu tiên chia theo bệnh nhân hoặc record trước khi tạo các cửa sổ/đoạn; không để đoạn chồng lấp, nhiều lead hoặc nhiều bản nhiễu của cùng một record xuất hiện ở cả train và test. Khóa danh sách ID của từng tập và lưu seed chia tập.

### 3.2 Chuẩn bị ECG sạch tương đương benchmark

Để tái hiện benchmark, xử lý ECG sạch theo thứ tự:

1. Dùng kênh ECG đầu tiên và annotation để xác định điểm bắt đầu sóng P.
2. Đặt đầu đoạn sớm hơn điểm bắt đầu sóng P **40 ms**; cắt đoạn giữa hai điểm bắt đầu liên tiếp. Chỉ giữ đoạn có tối đa một R-peak ở khoảng giữa theo quy tắc trích đoạn của dự án.
3. Resample từng đoạn về **360 Hz**; hạn chế hiệu ứng biên trong quá trình resample.
4. Bỏ đoạn dài hơn **496 điểm** sau resample.
5. Trừ khỏi toàn đoạn giá trị trung bình của **hai điểm đầu và cuối** của đoạn.
6. Đặt đoạn vào một vector zero dài **512 điểm**, bắt đầu ở vị trí 16.

Không tự thêm lọc tín hiệu, chuẩn hóa biên độ hoặc thay cách đệm nếu mục tiêu là so sánh số trực tiếp với benchmark. Nếu nghiên cứu A thay preprocessing, mô tả chính xác từng phép biến đổi và coi kết quả là một protocol mới. Đặc biệt, SSD và MAD phụ thuộc đơn vị biên độ nên các bảng chỉ so trực tiếp được khi đơn vị giống nhau.

### 3.3 Tách nguồn nhiễu

Benchmark đánh giá **baseline wander (BW)** từ nguồn NSTDB. Nguồn dữ liệu này có thêm electrode motion và muscle artifact, nhưng các kết quả của protocol hiện tại chỉ dùng BW. Chia mỗi kênh BW thành nửa đầu và nửa sau theo thời gian, rồi dùng hai protocol:

| Protocol | Nhiễu dùng khi train | Nhiễu dùng khi test |
|---|---|---|
| 1 | Nửa đầu kênh 1 | Nửa sau kênh 2 |
| 2 | Nửa đầu kênh 2 | Nửa sau kênh 1 |

Huấn luyện/chọn mô hình riêng cho từng protocol nếu tái hiện thiết kế gốc. Nghiên cứu A cũng có thể dùng một mô hình chung cho cả hai protocol, nhưng phải ghi rõ. Trong từng protocol, đoạn nhiễu test không được xuất hiện trong train. Nếu đánh giá thêm loại nhiễu khác, báo cáo thành nhóm riêng; không gộp vào bảng BW rồi gọi đó là kết quả cùng benchmark.

## 4. Tạo cặp ECG sạch–nhiễu và nhãn cường độ

Với mỗi đoạn sạch `x`, lấy một đoạn BW liên tiếp `n` dài 512 điểm. Dự án đi lần lượt qua nguồn nhiễu theo các đoạn 512 điểm và quay lại đầu khi gần hết nguồn. Chọn hệ số cường độ rời rạc:

```text
r thuộc {0.20, 0.21, 0.22, ..., 1.99}
A_x   = max(x) - min(x)
A_n   = max(n) - min(n)
alpha = r * A_x / A_n
y     = x + alpha * n
```

Sau phép scale, tỉ số biên độ đỉnh–đỉnh của thành phần nhiễu so với ECG sạch bằng `r`. **`r` không phải SNR dB.** Tín hiệu được tạo một cách xác định khi giữ nguyên danh sách đoạn sạch, thứ tự nguồn nhiễu và seed sinh `r` (benchmark dùng seed NumPy **1234**). Nếu áp dụng cho dữ liệu A, xử lý hoặc loại trường hợp `A_x=0` hay `A_n=0` theo quy tắc định trước để tránh chia cho 0.

Lưu một **bảng manifest** với ít nhất: `sample_id`, `patient_id`, `record_id`, `protocol`, `split`, `clean_segment_id`, `noise_source_id`, `noise_channel`, `noise_start`, `r`, `sampling_rate`, `preprocessing_version`, `seed`. Thứ tự của manifest phải trùng thứ tự cặp tín hiệu được chấm. Với cùng một ECG sạch được ghép với hai protocol nhiễu, tạo hai bản ghi phân biệt bằng cặp `(sample_id, protocol)`.

## 5. Huấn luyện mô hình A và khóa checkpoint

### 5.1 Bộ dữ liệu dùng để học

Với mỗi protocol, chuẩn bị tập cặp `(x, y)` trong đó `x` là ECG sạch và `y` là ECG đã cộng BW theo mục 4. **Đầu vào khử nhiễu là `y`, đích cần phục hồi là `x`.** Các đoạn ECG sạch dùng cho train/validation đến từ những record không nằm trong 14 record test; nhiễu của train/validation đến từ nửa đầu của kênh BW tương ứng protocol. Nhiễu của test đến từ nửa sau của kênh còn lại. Do đó, khi huấn luyện protocol 1 và 2, phần ECG sạch có thể giống nhau nhưng bản nhiễu và mô hình được huấn luyện là khác nhau.

Trong cách làm gốc, toàn bộ cặp train/validation được tạo trước, rồi chia ngẫu nhiên **70% đoạn cho train và 30% đoạn cho validation**. Validation dùng các đoạn ECG sạch khác train nhưng vẫn có thể chứa các đoạn từ cùng record; vì vậy đây không phải validation độc lập theo record. Phép chia gốc **không cố định seed**, nên nếu chỉ có checkpoint đã huấn luyện mà không có danh sách chỉ số split, không thể bảo đảm khôi phục chính xác tập train/validation đã dùng cho checkpoint đó. Khi huấn luyện A, hãy cố định và lưu danh sách chỉ số/seed. Nếu mục tiêu của A là ước lượng khả năng tổng quát hóa chặt chẽ hơn, chia train/validation theo record hoặc bệnh nhân; khi đó ghi rõ khác biệt với quy trình gốc. Trong cả hai trường hợp, không dùng record hoặc đoạn nhiễu test để huấn luyện, chuẩn hóa tham số hay chọn checkpoint.

Huấn luyện A trên **cùng nguồn ECG và nhiễu train** là yêu cầu tối thiểu khi so với checkpoint gốc. Để khẳng định hai mô hình học trên đúng **cùng các cặp train**, cần có manifest split của cả hai; nếu không, chỉ khẳng định chúng dùng cùng nguồn train và cùng tập test. Nếu A dùng tăng cường dữ liệu, tạo lại nhiễu mỗi epoch, dữ liệu bổ sung hoặc pretraining bên ngoài, công bố phạm vi dữ liệu và tạo thêm một so sánh có kiểm soát khi cần.

### 5.2 Mục tiêu học của mô hình gốc và cách áp dụng cho A

Mô hình gốc là conditional diffusion. Trong một bước huấn luyện, nó lấy ECG sạch `x`, chọn bước diffusion `t`, sinh nhiễu Gaussian `epsilon` và tạo tín hiệu trung gian `x_t`. Mạng nhận `(x_t, y, mức diffusion)` và dự đoán `epsilon`. Loss là **L1 tổng** giữa nhiễu Gaussian thật và nhiễu dự đoán:

```text
x_t          = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * epsilon
epsilon_hat  = model(x_t, y, mức_diffusion)
loss         = Σ |epsilon - epsilon_hat|
```

Một bước `t` được chọn cho cả batch; mức diffusion liên tục được lấy riêng cho từng phần tử trong khoảng của bước đó. Loss L1 dạng tổng tăng theo batch size và độ dài đoạn, nên không so trực tiếp trị số loss với một mô hình có loss trung bình hoặc batch size khác.

**Nghiên cứu A không bắt buộc học dự đoán nhiễu Gaussian.** A có thể học trực tiếp ánh xạ `y → x`, dự đoán thành phần nhiễu `y - x`, hoặc dùng objective khác. Điều kiện để đánh giá tương đương là sau huấn luyện, A xuất được `x_hat` cùng miền với `x` trên **tập test chung**. Trong báo cáo, mô tả rõ input, target, loss, augmentation, cách chuẩn hóa và nguồn dữ liệu của A.

### 5.3 Thông số huấn luyện gốc để tham chiếu

| Thành phần | Thiết lập của mô hình gốc |
|---|---|
| Số mô hình | Hai mô hình riêng, mỗi protocol một mô hình |
| Số kênh đặc trưng | 80 |
| Số bước diffusion | 50 |
| Lịch beta | Quadratic, từ `0.0001` đến `0.5` |
| Optimizer | Adam |
| Learning rate ban đầu | `1e-3` |
| Batch train/validation | 96 đoạn |
| Số epoch tối đa | 400 |
| Điều chỉnh learning rate | Giảm còn `0.1` lần sau mỗi 150 epoch |
| Giới hạn gradient | Chuẩn gradient tối đa `1.0` |
| Đánh giá validation | Mỗi epoch |
| Chọn checkpoint | Validation loss trung bình theo batch thấp nhất |

Batch train được xáo trộn mỗi epoch; batch train và validation cuối nếu thiếu 96 đoạn được bỏ. Tập test **không** bỏ batch cuối. Validation loss của mô hình diffusion cũng có thành phần ngẫu nhiên do chọn bước diffusion và sinh Gaussian; nếu huấn luyện lại, lưu seed và đường cong loss, và cân nhắc đánh giá validation ổn định hơn qua nhiều lần lấy mẫu. Checkpoint có validation loss tốt nhất là checkpoint dùng để test; checkpoint cuối epoch 400 chỉ là điểm kết thúc huấn luyện và không mặc nhiên tốt nhất.

Đây là cấu hình **tham chiếu**, không phải yêu cầu A phải dùng Adam, 400 epoch hay cùng loss. Khi A dùng kiến trúc khác, chọn siêu tham số bằng validation theo ngân sách đã công bố. Để so sánh có ý nghĩa, công bố cả tài nguyên huấn luyện, số tham số hoặc chi phí suy luận khi những yếu tố này khác đáng kể.

### 5.4 Khóa mô hình và giao diện suy luận

- Dùng train để tối ưu tham số mô hình, dùng validation để chọn kiến trúc, siêu tham số, epoch và checkpoint. **Không dùng test để chọn checkpoint hoặc chọn số lần suy mẫu tốt nhất rồi chỉ báo cáo lựa chọn đó.**
- Ghi rõ mô hình nào dùng cho protocol 1 và protocol 2. Nếu có hai checkpoint, mỗi checkpoint phải được chọn theo validation của protocol tương ứng.
- Ghi cấu hình mô hình, phiên bản môi trường, seed huấn luyện, seed suy luận, checkpoint, batch size, thiết bị và số lượng mẫu test.
- Đưa mô hình về chế độ đánh giá và tắt tính gradient khi suy luận. Nếu có Dropout/BatchNorm, điều này ảnh hưởng trực tiếp đến kết quả.
- Tạo một giao diện suy luận nhận batch `[B, 1, 512]` và trả ECG đã khử nhiễu cùng shape. Không cần dùng diffusion trong nghiên cứu A; cách chấm điểm chỉ đòi hỏi `x_hat` hợp lệ.
- Nếu đầu ra của A có độ trễ, bị cắt hoặc resample, xác định cách căn chỉnh bằng quy tắc cố định hoặc dữ liệu validation. Không tối ưu độ dịch cho từng mẫu bằng ground truth test mà không công bố bước đó.

Để tái hiện cấu hình suy luận của mô hình gốc: mô hình điều kiện nhận ECG nhiễu, bắt đầu từ Gaussian ngẫu nhiên và thực hiện **50 bước diffusion ngược**. Đây là đặc điểm của mô hình gốc, không phải điều kiện bắt buộc với kiến trúc của A.

## 6. Suy luận một lần và nhiều lần

Đánh giá các mức `K = 1, 3, 5, 10` đối với mô hình có suy luận ngẫu nhiên. Với cùng đầu vào `y`, chạy `K` lượt độc lập, mỗi lượt tạo một tín hiệu đầu ra `x_hat^(k)`, rồi tính:

```text
x_hat_K = (x_hat^(1) + ... + x_hat^(K)) / K
```

**Tính metric trên `x_hat_K`**, không tính metric từng lượt rồi lấy trung bình các metric. Mỗi K là một điều kiện đánh giá riêng; lưu seed hoặc trạng thái bộ sinh ngẫu nhiên đủ để lặp lại. Batch test của benchmark có tối đa **50 đoạn**, không xáo trộn và vẫn giữ batch cuối dù chưa đủ 50. Nếu đổi batch size, kiểm tra ảnh hưởng đến metric PRD gốc ở mục 7.

Với mô hình tất định, `K>1` chỉ lặp lại cùng kết quả nên báo cáo `K=1`. Với mô hình ngẫu nhiên, báo cáo thêm thời gian suy luận, vì chi phí thường tăng gần theo K. Không so độ chính xác của A tại `K=10` với một mô hình khác tại `K=1` mà bỏ qua khác biệt chi phí.

## 7. Tính chỉ số cho **từng đoạn**, trước khi tổng hợp

Gọi `i` là điểm thời gian trong một đoạn. Với `x` sạch, `y` nhiễu và `x_hat` đã khử nhiễu, dùng các định nghĩa sau:

| Chỉ số | Công thức | Kết quả tốt hơn khi |
|---|---|---|
| SSD | `Σ_i (x_i - x_hat_i)^2` | Thấp |
| MAD | `max_i abs(x_i - x_hat_i)` | Thấp |
| Cosine similarity | `(x · x_hat) / (||x||₂ ||x_hat||₂)` | Cao, gần 1 |
| SNR-in | `10 log10[Σ_i x_i² / Σ_i (y_i - x_i)²]` | Cao |
| SNR-out | `10 log10[Σ_i x_i² / Σ_i (x_hat_i - x_i)²]` | Cao |
| SNR improvement | `SNR-out - SNR-in` của **cùng đoạn** | Cao |

### 7.1 Định nghĩa PRD cần đặc biệt chú ý

Để tái hiện đúng phép tính của dự án, PRD được tính theo batch đang chấm:

```text
PRD_repo(sample) = 100 * sqrt(
    Σ_i (x_hat_i - x_i)² /
    Σ_i (x_hat_i - mean(x_batch))²
)
```

`mean(x_batch)` là trung bình của **toàn bộ giá trị ECG sạch trong batch hiện tại**, không phải trung bình riêng từng đoạn. Mẫu số dùng `x_hat`. Do đó `PRD_repo` có thể thay đổi nếu đổi batch size hoặc thứ tự mẫu trong batch, dù `x_hat` không đổi. Muốn so khớp con số gốc, giữ batch size 50, thứ tự test cố định và báo cáo rõ định nghĩa này.

Để báo cáo thêm một PRD theo từng đoạn, có thể tính:

```text
PRD_standard(sample) = 100 * sqrt(
    Σ_i (x_hat_i - x_i)² /
    Σ_i (x_i - mean(x_sample))²
)
```

Đặt tên hai cột riêng biệt. Không thay `PRD_repo` bằng `PRD_standard` rồi coi như tái hiện cùng chỉ số. Khi so nhiều mô hình, tất cả phải dùng cùng một định nghĩa trong cùng bảng.

### 7.2 Trường hợp chỉ số không xác định

Trước khi tổng hợp, kiểm tra mẫu số bằng 0 và giá trị `NaN`/`inf` ở PRD, SNR hoặc cosine similarity; lưu số mẫu gặp lỗi theo từng protocol và từng mô hình. Chọn quy tắc xử lý trước khi chấm, ví dụ loại các đoạn ECG sạch hằng ngay từ khâu tạo tập test hoặc báo cáo số ngoại lệ riêng. Áp dụng **cùng quy tắc cho mọi mô hình**; không âm thầm bỏ các kết quả xấu của một mô hình.

## 8. Tổng hợp kết quả toàn tập và theo mức nhiễu

Với mỗi `(mô hình, protocol, K)`, tính từng metric cho từng đoạn, sau đó tính **mean ± std** của các giá trị per-segment. Độ lệch chuẩn tương ứng với `std` quần thể trên tập đoạn đang xét (`ddof=0`). Đây là độ phân tán giữa các đoạn, **không phải** độ bất định giữa nhiều lần huấn luyện hay khoảng tin cậy.

### 8.1 Kết quả toàn tập

Xuất ba nhóm bảng: riêng protocol 1, riêng protocol 2 và gộp hai protocol. Cách gộp của benchmark là **nối các dòng per-segment của hai protocol rồi tính mean/std**, nên protocol nào nhiều dòng hơn có trọng số lớn hơn. Cùng một ECG sạch có thể xuất hiện hai lần với hai nhiễu khác nhau; số dòng gộp không phải số bệnh nhân độc lập. Ghi `N` của từng bảng.

### 8.2 Phân tầng mức nhiễu

Các mốc của benchmark là `r = 0.2, 0.6, 1.0, 1.5, 2.0`. Để mỗi dòng thuộc đúng một nhóm, dùng:

```text
Nhóm 1: 0.20 ≤ r < 0.60
Nhóm 2: 0.60 ≤ r < 1.00
Nhóm 3: 1.00 ≤ r < 1.50
Nhóm 4: 1.50 ≤ r ≤ 2.00
```

Phép phân nhóm nguyên bản dùng hai biên đóng, nên các mẫu có `r` bằng đúng `0.6`, `1.0` hoặc `1.5` có thể được tính trong **hai** nhóm. Nếu mục tiêu là tái hiện đúng từng dòng báo cáo nguyên bản, giữ thêm một bảng theo quy tắc hai biên đóng và chú thích việc đếm trùng. Bảng phân tích chính của nghiên cứu A nên dùng nhóm không giao nhau và xác nhận tổng `N` bốn nhóm bằng `N` toàn tập.

Nếu nghiên cứu A dùng SNR dB hoặc thang nhiễu khác làm nhãn, phải đặt khoảng mới và công bố định nghĩa. Không đổi tên SNR dB thành `r` hoặc đưa trực tiếp vào bốn khoảng trên.

## 9. Báo cáo độ tin cậy và ý nghĩa kết quả

- Báo cáo riêng kết quả theo **record hoặc bệnh nhân** nếu muốn kết luận cho quần thể bệnh nhân. Tính metric trên từng đoạn, tổng hợp trong mỗi record/bệnh nhân, rồi mới tổng hợp các đơn vị độc lập.
- Nếu so sánh hai mô hình, chấm **cùng cặp tín hiệu** và báo cáo chênh lệch theo cặp. Khoảng tin cậy hoặc kiểm định nên tôn trọng việc nhiều đoạn thuộc cùng bệnh nhân/record và một ECG sạch có thể có hai bản nhiễu.
- Nếu kết quả thay đổi theo khởi tạo hoặc sampling, lặp lại với nhiều seed và báo cáo cả biến thiên giữa các lần chạy. `mean ± std` qua các đoạn của một lần chạy không thay thế điều này.
- Ghi hiệu năng theo loại nhiễu, mức nhiễu, record/bệnh nhân và thời gian suy luận. Với ứng dụng lâm sàng, đánh giá thêm khả năng bảo toàn P/QRS/T hoặc ảnh hưởng đến thuật toán phát hiện R-peak; các chỉ số sai số dạng sóng không tự chứng minh giá trị lâm sàng.
- Nêu rõ dữ liệu test thuộc BW tổng hợp từ ECG sạch. Không suy rộng kết quả đó cho mọi nhiễu thực tế, thiết bị hoặc bệnh viện khi chưa có test ngoài miền.

## 10. Bảng kết quả và thông tin cần lưu

Lưu **một dòng kết quả cho mỗi `(mô hình, protocol, K, sample)`** với tối thiểu các trường:

```text
experiment_id, model_id, checkpoint_id, train_seed, inference_seed,
protocol, K, sample_id, patient_id, record_id, noise_type, r,
ssd, mad, prd_repo, prd_standard, cosine_similarity,
snr_in_db, snr_out_db, snr_improvement_db, inference_time_ms
```

Nếu không tính `PRD_standard`, bỏ cột này. Lưu hoặc liên kết đầu ra `x_hat` với `sample_id` để có thể vẽ lại tín hiệu và kiểm tra metric. Bảng tổng hợp phải được tạo lại từ các dòng per-sample, không chép tay từ log màn hình.

Mẫu bảng tổng hợp:

| Mô hình | Protocol | K | Nhóm r | N | SSD | MAD | PRD (ghi biến thể) | Cosine | SNR-in | SNR-out | ΔSNR | Thời gian/đoạn |
|---|---|---:|---|---:|---|---|---|---|---|---|---|---:|
| A | 1 | 1 | ALL | ... | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | ... |
| A | 2 | 1 | ALL | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| A | Gộp | 1 | ALL | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Lặp lại cho các K và nhóm r đã định nghĩa. Báo cáo phương pháp phải ghi: nguồn ECG và nhiễu, tiêu chí loại đoạn, tần số, chiều dài đoạn, split, seed, cách tạo cặp, cách chọn checkpoint, số bước suy luận nếu có, thiết bị, batch size, định nghĩa của `std`, công thức PRD và quy tắc xử lý chỉ số không hữu hạn.

## 11. Quy trình thực hiện theo thứ tự

1. **Khóa protocol:** chọn mục tiêu tái hiện benchmark hay áp dụng cùng phương pháp trên dữ liệu A; chốt metric và định nghĩa PRD.
2. **Tạo split:** tách bệnh nhân/record cho train, validation, test; ghi ID và seed, kiểm tra không rò rỉ.
3. **Tạo ECG sạch:** xử lý kênh, annotation, resample, cắt và đệm đoạn theo protocol đã chốt.
4. **Tạo nhiễu test:** tách nguồn BW theo hai protocol, ghép nhiễu với ECG sạch, tính `r`, cố định seed và manifest.
5. **Huấn luyện theo từng protocol:** tạo cặp train/validation đúng nguồn ECG và BW, học mô hình A với objective đã công bố; chọn checkpoint và mọi siêu tham số bằng validation, rồi khóa mô hình trước khi chấm test.
6. **Kiểm tra một batch:** xác nhận shape `[B,1,512]`, số mẫu, ID, thứ tự, biên độ, không có `NaN`/`inf`, và đầu ra căn chỉnh với ECG sạch.
7. **Chạy suy luận:** theo từng protocol, từng K và từng batch; lấy trung bình tín hiệu đầu ra khi K lớn hơn 1.
8. **Tính metric từng đoạn:** SSD, MAD, PRD theo định nghĩa đã chốt, cosine, SNR-in, SNR-out và ΔSNR.
9. **Kiểm toán dữ liệu:** đối chiếu số dòng, giá trị không hữu hạn, mối quan hệ `ΔSNR = SNR-out - SNR-in`, và nhóm mức nhiễu.
10. **Tổng hợp và viết báo cáo:** bảng riêng từng protocol, bảng gộp, bảng theo mức nhiễu, `N`, mean ± std, thời gian và giới hạn diễn giải.

## 12. Checklist đạt trước khi công bố

- [ ] Mỗi đầu vào nhiễu có một ground truth sạch đúng cặp và một ID ổn định.
- [ ] Train/validation/test độc lập theo bệnh nhân hoặc record; nhiễu test không bị dùng trong train của cùng protocol.
- [ ] ECG sạch, ECG nhiễu và đầu ra cùng chiều dài, tần số, kênh, đơn vị, thứ tự và thang biên độ khi chấm.
- [ ] Cùng tập test và cùng quy tắc tiền xử lý được áp dụng cho mọi mô hình so sánh.
- [ ] Mô hình và checkpoint đã được khóa bằng validation trước khi xem điểm test.
- [ ] Phạm vi dữ liệu huấn luyện, target, loss, augmentation và siêu tham số của A đã được công bố; có ánh xạ rõ giữa từng protocol và mô hình dùng để chấm.
- [ ] Số dòng kết quả bằng số cặp test của từng protocol và từng K; batch cuối không bị bỏ.
- [ ] Khi K lớn hơn 1, tín hiệu đầu ra được lấy trung bình **trước** khi tính metric.
- [ ] Công thức PRD được đặt tên rõ; nếu cần tái hiện PRD gốc, batch size và thứ tự test được giữ cố định.
- [ ] Số mẫu có metric không hữu hạn được đếm và xử lý nhất quán giữa các mô hình.
- [ ] Với nhóm mức nhiễu không giao nhau, tổng `N` các nhóm bằng `N` toàn tập.
- [ ] Mỗi dòng có `ΔSNR = SNR-out - SNR-in`; báo cáo tổng hợp được tính từ các dòng kết quả từng mẫu.
- [ ] Báo cáo có `N`, mean ± std, kết quả từng protocol, kết quả gộp, nhóm mức nhiễu, seed, thiết bị và thời gian suy luận.
