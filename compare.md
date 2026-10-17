# Quy tắc tái lập giao thức TFCDiff để huấn luyện và so sánh mô hình ECG

Tài liệu này ghi lại **hành vi của mã trong repository tại thời điểm đọc**, ưu tiên mã ở thư mục gốc (`main.py`, `Data_Preparation/`, `utils.py`, `diffusion.py`, `complete_eval.py`) và phân biệt với khuyến nghị để so sánh công bằng. Đây là giao thức **nhiễu hỗn hợp BW + EM + MA** của TFCDiff. `05_tfcdiff.md` trong cùng thư mục nói về một benchmark **chỉ nhiễu BW** trên dữ liệu chung; không dùng lẫn hai bộ kết quả.

## 1. Nguồn dữ liệu và đơn vị mẫu

- ECG sạch: QT Database (QTDB), đọc hai kênh của từng record bằng `wfdb.rdsamp`.
- Nhiễu: MIT-BIH Noise Stress Test Database (NSTDB), ba record `bw` (baseline wander), `em` (electrode motion), `ma` (muscle artifact), mỗi record có hai kênh.
- Đánh giá ngoài tập: README nêu SimEMG, nhưng repository **không có quy trình tạo cặp `X_test_simEMG.pt`/`y_test_simEMG.pt` hoàn chỉnh**. Vì vậy không thể xác nhận đầy đủ cách tạo test SimEMG từ riêng mã này.
- Mỗi ví dụ là **một kênh ECG dài 10 giây ở 360 Hz = 3.600 mẫu**. Hai kênh QTDB được đưa thành các ví dụ riêng, không ghép thành đầu vào hai kênh.
- Tên tensor: `X_*` là ECG có nhiễu, `y_*` là ECG sạch. `Data_Preparation()` trả `(X_train, y_train, X_test, y_test)` ở dạng `(N, 3600, 1)` kiểu `float32`; `main.py` đổi thành `(N, 1, 3600)`.

Nguồn: `Data_Preparation/Prepare_QTDatabase.py`, `Data_Preparation/Prepare_NSTDB.py`, `Data_Preparation/data_preparation.py`.

## 2. Tiền xử lý QTDB

Thực hiện theo thứ tự trong `Prepare_QTDatabase.prepare()`:

1. Đọc từng record QTDB và hai kênh ECG. Đọc annotation `pu0` cho kênh 1, `pu1` cho kênh 2; lấy vị trí ký hiệu `N`.
2. Lọc thông dải Butterworth bậc 5 từ **0,5 đến 40 Hz**, qua `scipy.signal.filtfilt`. Hàm dùng `fs=250` mặc định tại bước này. Sau đó lọc trung bình trượt cửa sổ **9** mẫu (`np.convolve(..., mode='same')`).
3. Chia thành các đoạn tại trung điểm giữa các annotation `N` liên tiếp. Khử đường nền **tuyến tính** bằng `np.polyfit(..., 1)` riêng cho từng đoạn; dùng nội suy Cubic Hermite tại chỗ nối các đoạn.
4. Resample mỗi kênh lên **360 Hz** bằng `scipy.signal.resample_poly(processed, 360, fs_goc)`; bỏ mẫu đầu và cuối (`[1:-1]`).
5. Từ mỗi kênh lấy tất cả cửa sổ **không chồng lấn** dài 3.600 mẫu; thêm **90 cửa sổ ngẫu nhiên** dài 3.600 mẫu trên mỗi kênh. Bỏ đoạn cuối thiếu chiều dài. Các cửa sổ ngẫu nhiên có thể chồng lấn cửa sổ đều.
6. Lưu dictionary `{record_id: danh_sách_đoạn}` vào `data/QTDatabase.pkl`; lần sau dùng lại cache. Không có bước chuẩn hóa min-max trong đường xử lý này (dòng `normalize` đã bị comment).

Mã `Prepare_QTDatabase.py` đặt seed 42 khi import, nhưng `Data_Preparation()` đặt lại NumPy seed **1234 trước khi gọi `prepare()`** nếu chưa có cache. Việc tạo cache trước/sau và thứ tự duyệt record có thể làm thay đổi các cửa sổ ngẫu nhiên. Muốn so sánh nghiêm ngặt, hãy **đóng băng chính tập cặp tín hiệu và ID record**, không chỉ ghi seed.

## 3. Chia train / validation / test

### ECG sạch

Test được tách theo **record ID**, gồm 14 record sau (theo mã hiện tại):

```text
sel123 sel233 sel302 sel307 sel820 sel853 sel16420 sel16795
sele0106 sele0121 sel32 sel49 sel14046 sel15814
```

Mọi đoạn từ record khác đi vào tập nguồn train. README ghi bản sửa lỗi chia dữ liệu ngày 14/04/2026: số ví dụ nguồn train đổi từ **37.590 thành 32.578**, số ví dụ test đổi từ **4.296 thành 5.012**. Các con số này là thông tin README, cần kiểm tra lại trên cache/dataset cụ thể trước khi báo cáo.

Sau khi đã tạo nhiễu cho tập nguồn train, `main.py` dùng `train_test_split(range(...), test_size=0.3)` để chia **70% train / 30% validation theo đoạn**, không truyền `random_state`. Validation do đó có thể chứa các đoạn cùng record hoặc chồng lấn với train. Test theo record vẫn tách riêng. Khi tái lập, lưu lại chỉ số chia train/validation; khi muốn đánh giá nghiêm ngặt hơn, có thể bổ sung split validation theo record và báo cáo rõ đó là **giao thức khác**.

### NSTDB: chia vùng nhiễu

Mỗi record `bw`, `em`, `ma` được chia hai nửa theo thời gian: `a = [0 : floor(L/2)]`, `b = [floor(L/2) : -1]` (mã bỏ mẫu cuối). Với mỗi loại nhiễu, ký hiệu `ch1/ch2` là hai kênh. `main.py` dùng `--n_type=1` mặc định.

| `n_type` | Vùng nhiễu train | Vùng nhiễu test |
|---|---|---|
| 1 | `ch1_a` | `ch2_b` |
| 2 | `ch2_a` | `ch1_b` |
| 3 | `ch2_b` | `ch1_a` |
| 4 | `ch1_b` | `ch2_a` |

Cùng một lựa chọn được áp dụng cho cả BW, EM và MA. Không thay các vùng này nếu muốn so trực tiếp với kết quả `n_type=1`.

## 4. Sinh nhiễu hỗn hợp và mức nhiễu

Với mỗi ví dụ ECG sạch `s` dài 3.600 mẫu, `generate_noise()` cắt **một đoạn liên tục ngẫu nhiên** dài 3.600 mẫu từ mỗi nguồn BW, EM, MA của vùng train hoặc test tương ứng. Ký hiệu ba đoạn là `bw`, `em`, `ma` và `a1,a2` là hai số nguyên độc lập lấy từ `np.random.randint(0,101)` (0–100, gồm cả 100). Mã tính đúng theo biểu thức:

```text
n_mix = [a1*bw + (100-a1)*(a2*em + (100-a2)*ma)] / 100
r     = (max(s)-min(s)) / (max(n_mix)-min(n_mix))
alpha = q*r
x     = s + alpha*n_mix
```

Trong đó `q` là `rnd_train[i]` hoặc `rnd_test[i]`, lấy bằng `np.random.randint(20,200)/100`: các giá trị rời rạc **0,20 đến 1,99**, bước 0,01; không có đúng 2,00. Đây là **hệ số tỷ lệ biên độ đỉnh-đỉnh**, không phải mức SNR dB cố định. SNR đầu vào phải đo sau khi tạo cặp dữ liệu.

**Cần giữ nguyên dấu ngoặc khi tái lập:** công thức trên chia `100` **một lần ở ngoài cùng**, đúng như mã; không tự thay bằng phép trộn lồi có tổng trọng số bằng 1. `rnd_train.npy` và `rnd_test.npy` được lưu/tái sử dụng, nhưng các vị trí cắt nhiễu và `a1,a2` không được lưu riêng. Vì thế chỉ giữ `rnd_*.npy` chưa đủ đảm bảo tạo lại đúng từng cặp; nên lưu `X`, `y`, record ID và mức `q` của toàn bộ test trước khi chạy nhiều mô hình. Mã không xử lý trường hợp `max(n_mix)==min(n_mix)`; dữ liệu chuẩn thông thường tránh được nhưng bản tái lập nên kiểm tra.

`Data_Preparation()` đặt NumPy seed **1234** ở mỗi lần gọi. Vòng augmentation hiện là `range(1)`: mỗi đoạn sạch có **một phiên bản nhiễu** trong mỗi lần tạo dataset, không tạo lại nhiễu mỗi epoch.

## 5. Huấn luyện mô hình của bạn trên cùng bài toán

- Đầu vào: `X_train` ECG có nhiễu, mục tiêu: `y_train` ECG sạch; shape sau đổi trục `(N,1,3600)`.
- Huấn luyện trên phần 70%, chọn checkpoint bằng validation 30%, sau đó chỉ đánh giá test một lần. Không dùng test để chọn hyperparameter hoặc checkpoint.
- Để so kiến trúc công bằng, ưu tiên cấp **cùng các tensor train/validation/test đã lưu** cho TFCDiff và mô hình của bạn. Mô hình của bạn có thể dùng miền thời gian hoặc tần số riêng; ghi rõ mọi phép biến đổi và bảo đảm đầu ra được trả về cùng 3.600 mẫu và cùng thang biên độ với `y_test` trước khi tính metric.
- Trong mã TFCDiff, `TensorDataset` theo thứ tự `(clean, noisy)` vì hàm train của nó nhận hai tensor theo thứ tự đó. Với mô hình riêng, cần kiểm tra rõ chiều `input/target` để tránh đảo ngược.

Tham số train gốc trong `config/base.yaml` và `utils.train()`:

| Tham số | Giá trị / hành vi |
|---|---|
| Epoch | 400 |
| Batch train / validation | 128 |
| Optimizer | Adam, learning rate `1e-3`, weight decay `5e-4` |
| Learning-rate scheduler | StepLR, `step_size=150`, `gamma=0.1`, gọi cuối mỗi epoch |
| Gradient clipping | Global norm 1.0 trên tham số `model.model` |
| Train loader | shuffle, `drop_last=True`, `num_workers=0` |
| Validation loader | không shuffle, `drop_last=True`, `num_workers=0` |
| Validation và chọn model | Mỗi epoch; lưu `model.pth` khi **mean validation diffusion loss theo batch** giảm; lưu `final.pth` sau epoch cuối |
| Test loader | Batch 50, không `drop_last` |

Đây là hyperparameter của **TFCDiff**, không phải điều kiện bắt buộc áp lên kiến trúc khác. Khi thay đổi để phù hợp mô hình của bạn, công bố rõ và giữ **dữ liệu, split, test, metric** giống nhau.

## 6. Đường xử lý riêng của TFCDiff

Phần này cần để tái lập baseline TFCDiff; mô hình mới không buộc phải dùng DCT hoặc diffusion.

- Từ mỗi `clean` và `noisy` dài 3.600, dùng **DCT trực chuẩn** (`torch_dct.dct(..., norm='ortho')`), chia `eta=3`, giữ **1.000 hệ số đầu**. U-Net nhận tensor `(B,2,1000)` gồm điều kiện noisy và trạng thái diffusion; dự đoán nhiễu Gauss `(B,1,1000)`.
- Cấu hình U-Net: `in_channel=2`, `out_channel=1`, `inner_channel=64`, `channel_mults=[1,2,2,2]`, `attn_res=[250]`, `res_blocks=2`, `dropout=0`, `seq_len=1000`, `norm_groups=16`.
- Diffusion: **50 bước**, beta theo lịch `quad` từ `0.0001` đến `0.5`. Mã còn **rescale SNR lịch diffusion bằng factor 150**, rồi tính lại alpha/beta hiệu dụng; không thể chỉ dựng lịch beta thường mà coi là giống baseline.
- Train lấy một timestep nguyên `t` ngẫu nhiên từ 1 đến 50 cho cả batch, lấy hệ số `sqrt(alpha_bar)` liên tục riêng từng mẫu trong khoảng của bước đó, thêm Gaussian noise, rồi tối ưu dự đoán nhiễu. Loss đang chạy là **hybrid Charbonnier**: tổng `sqrt(err^2+1e-8)` trong miền hệ số DCT cộng với tổng cùng dạng trên `IDCT(zero_pad(err,2600))`; trọng số hai phần mặc định 1. Không có chia trung bình theo chiều dài trong hàm loss.
- Inference mặc định **1 lần sinh (1-shot)**: DCT của noisy → giữ 1.000 hệ số, chia 3 → 50 bước reverse diffusion → pad thêm 2.600 hệ số 0 → nhân 3 → inverse DCT ra 3.600 mẫu. Reverse diffusion khởi tạo Gaussian ngẫu nhiên, nên kết quả chịu ảnh hưởng seed PyTorch.
- `complete_eval.py` còn thử `shots=[1,3,5,10]`: trung bình nhiều đầu ra stochastic trong miền hệ số trước inverse DCT. Khi so sánh, ghi rõ số shot; 1-shot là mặc định của `main.py` và `denoising.py`.

## 7. Metric: dùng đúng công thức trong `utils.py`

Tính **theo từng đoạn** dài `L=3600` rồi báo cáo trung bình (và độ lệch chuẩn nếu theo `complete_eval.py`) trên các đoạn. Ký hiệu `y` là sạch, `x` là noisy, `ŷ` là đầu ra; mọi tín hiệu cần cùng biên độ gốc.

```text
SSD(y,ŷ)    = Σ_i (y_i - ŷ_i)^2                         [thấp tốt]
MAD(y,ŷ)    = max_i |y_i - ŷ_i|                         [thấp tốt]
SNR(y,z)    = 10 log10(Σ_i y_i^2 / Σ_i (z_i-y_i)^2)   [dB; cao tốt]
SNR_in      = SNR(y,x)
SNR_out     = SNR(y,ŷ)
ImSNR       = SNR_out - SNR_in                        [dB; cao tốt]
CosSim      = (y·ŷ) / (||y||₂ ||ŷ||₂)                  [cao tốt]
```

**PRD đặc thù của mã** cần giữ đúng nếu so với số của repo:

```text
PRD_j = 100 * sqrt(
    Σ_i (ŷ_{j,i} - y_{j,i})²
    / Σ_i (ŷ_{j,i} - mean(Y_batch))²
)
```

`mean(Y_batch)` là `np.mean(y)` **toàn batch**, vì hàm `metrics.PRD(y,y_pred)` nhận array `(B,3600,1)` và gọi `np.mean(y)` không chỉ định trục. Mẫu số dùng **`ŷ`**, không dùng `y`. Do đó PRD của repo phụ thuộc cách chia batch và khác PRD chuẩn thường gặp. Nếu cần PRD chuẩn, báo cáo **thêm một cột riêng** với tên và công thức khác; không thay thế PRD trên trong phép so trực tiếp. `CosSim` được tính từng mẫu bằng `sklearn.metrics.pairwise.cosine_similarity`.

`complete_eval.py` phân nhóm theo `q` bằng các khoảng `[0.2,0.6]`, `[0.6,1.0]`, `[1.0,1.5]`, `[1.5,2.0]` rồi báo cáo mean ± std. Điều kiện mã là `>=` và `<=` ở cả hai đầu, nên mẫu tại ranh giới **xuất hiện ở hai nhóm**; lưu ý khi tái lập bảng theo nhóm.

## 8. Cách chạy và các điểm cần ghi trong báo cáo

Từ gốc repository, README đưa lệnh:

```text
python -W ignore main.py --device cuda:0 --n_type=1
python complete_eval.py
```

`main.py` huấn luyện rồi đánh giá validation cho `final.pth`, validation cho `model.pth`, sau đó test trên `model.pth`. `complete_eval.py` đọc `./check_points/model.pth`, khác đường dẫn checkpoint có timestamp do một lần `main.py` mới tạo; muốn dùng checkpoint vừa train cần **chỉ rõ đường dẫn** trong bản chạy của bạn. `complete_eval.py` đọc/tạo `X_test.pt`, `y_test.pt`; nên lưu và chia sẻ đúng cặp test được dùng cùng metadata split. Mã hiện gọi `Data_Preparation(1)` trước khi kiểm tra cache test, nên nó vẫn cần dữ liệu gốc/các pickle liên quan.

Checklist tối thiểu cho phép so sánh có thể kiểm chứng:

1. Ghi nguồn QTDB/NSTDB, phiên bản dữ liệu, 14 record test, `n_type`, preprocessing và chiều dài 3.600.
2. Lưu `sample_id`, `record_id`, `X_train/y_train`, chỉ số train/validation, `X_test/y_test`, và `q` của từng mẫu test. Kiểm tra không có record test trong train; nêu rõ validation theo đoạn có thể chồng lấn.
3. Cố định đúng test tensors cho **mọi** mô hình; kiểm tra thứ tự mẫu, shape, dtype và thang biên độ.
4. Tính metric từ **cùng một implementation**, cùng batch size nếu cần khớp PRD repo; báo cáo mean ± std và số lượng mẫu. Nêu rõ cách xử lý mẫu số bằng 0 nếu phát sinh.
5. Ghi seed NumPy/PyTorch, checkpoint được chọn, epoch chọn, số shot, thời gian/chế độ inference và mọi thay đổi hyperparameter.
6. Không ghép số của benchmark **BW-only** trong `05_tfcdiff.md`, số SimEMG hoặc kết quả các baseline cũ vào bảng mixed-noise QTDB/NSTDB nếu test pairs/preprocessing/metric chưa được xác nhận giống nhau.

## 9. Phạm vi chưa xác minh từ repository

- Repository không chứa sẵn QTDB/NSTDB đã giải nén ở `data/`, nên tài liệu này là đọc mã, **chưa chạy lại toàn bộ pipeline** và chưa xác nhận số lượng tensor thực tế.
- README mô tả kết quả SimEMG nhưng không cung cấp đủ mã/metadata tạo cặp SimEMG để tái lập chính xác giao thức ngoài tập. Các script đối chứng có đọc cặp `.pt` và một số script chia biên độ cho `200`; không suy ra quy tắc đó cho mixed-noise test chính.
- `rnd_train.npy`/`rnd_test.npy` có trong repo nhưng không bảo đảm cùng cặp noisy nếu chạy lại generator; cần lưu tensor test cuối cùng.

## 10. Quy trình tiền xử lý khi so sánh một mô hình mới

**Ranh giới quan trọng:** các bước ở mục 2 tạo **ground truth sạch từ QTDB trước khi trộn nhiễu**. Không lấy ECG đã cộng BW/EM/MA rồi chạy lại bộ lọc 0,5–40 Hz, khử đường nền hoặc Hermite để “làm sạch đầu vào”; làm như vậy sẽ đổi bài toán mà TFCDiff đã được đánh giá. Thứ tự đúng là:

```text
QTDB hai kênh + annotation
  → bandpass / trung bình trượt / khử nền / Hermite
  → resample 360 Hz → cắt từng đoạn 3600 → y_clean
NSTDB bw, em, ma → chọn vùng train/test → cắt và trộn nhiễu
  → x_noisy = y_clean + nhiễu đã nhân alpha
(x_noisy, y_clean) → huấn luyện hoặc đánh giá mô hình
```

Để có **so sánh trực tiếp**, nên làm theo trình tự sau:

1. Tạo một lần bộ cặp `(x_noisy, y_clean)` theo mục 2–4, với record test và `n_type` cố định. Lưu kèm `record_id`, chỉ số train/validation và `q`. Kiểm tra `len(x)==len(y)==3600`, tín hiệu hữu hạn, và test không có record nguồn train.
2. Đưa **cùng các cặp** vào TFCDiff và mô hình của bạn. Nếu có thể, chạy checkpoint TFCDiff trên chính cặp test này; không so số từ một lần tạo nhiễu khác vì nhiễu được cắt ngẫu nhiên.
3. Nếu mô hình mới cần chuẩn hóa, lọc thêm, STFT, wavelet hoặc đổi tần số lấy mẫu, coi đó là **pipeline của mô hình mới**: xác định tham số chỉ trên train, áp dụng đồng nhất cho validation/test, và không dùng `y_test` để tính thống kê hay hiệu chỉnh. Ghi đầy đủ biến đổi. Trước khi chấm điểm, đảo mọi phép biến đổi cần thiết để đầu ra trở về **3600 mẫu/360 Hz và cùng đơn vị biên độ** với `y_clean`. Không chuẩn hóa mỗi tín hiệu dự đoán bằng min/max riêng để tăng metric.
4. Chọn checkpoint/hyperparameter chỉ bằng validation. Chạy test ở chế độ inference với đúng checkpoint đã chọn. Lưu `sample_id` và prediction theo đúng thứ tự test để có thể chấm lại độc lập.
5. Chạy **một script metric chung** trên `x_noisy`, `y_clean` và prediction của mọi mô hình. Báo cáo SSD, MAD, PRD theo định nghĩa repo, CosSim, SNR_in, SNR_out, ImSNR; ghi `N`, mean ± std, batch size của PRD và số lần sinh nếu mô hình có ngẫu nhiên.

Nếu nghiên cứu của bạn dùng **ECG thô hoặc cách lọc/khử nền khác**, hãy báo cáo đây là thí nghiệm khác giao thức TFCDiff. Có thể bổ sung thí nghiệm đó để kiểm tra tính tổng quát, nhưng phép so chính với số của bài cần dùng **cùng ground truth đã tiền xử lý và cùng noisy test pairs**. Các bước biến đổi nội bộ sau khi nhận `x_noisy` là lựa chọn mô hình, miễn là chấm kết quả trên cùng miền đầu ra.

## 11. Quy tắc áp dụng cho nghiên cứu hiện tại của bạn (`my_research.md`)

**Mục tiêu so sánh chính:** huấn luyện **mô hình đề xuất của bạn**, sau đó chạy nó và **checkpoint TFCDiff có sẵn** trên cùng tập test QTDB một đạo trình, 3.600 mẫu/10 giây/360 Hz, nhiễu **trộn đồng thời BW + EM + MA**. **Không yêu cầu huấn luyện lại TFCDiff.** Mục tiêu này khác common benchmark QTDB–BW-only–3600 trong `my_research.md` và `05_tfcdiff.md`; giữ BW-only làm thí nghiệm phụ với bảng riêng.

### 11.1. Chuẩn bị dữ liệu train cho mô hình bạn và tập test chung

1. Lấy QTDB và NSTDB gốc đúng cấu trúc mà `Data_Preparation/Prepare_QTDatabase.py` và `Prepare_NSTDB.py` đọc. Tiền xử lý ECG sạch đúng mục 2; giữ **14 record test** ở mục 3. Không dùng `clean_*` của common BW-only như bằng chứng là đã khớp pipeline bài nếu chưa xác minh nguồn, cách cắt và record ID.
2. Sinh BW+EM+MA theo **đúng công thức mục 4**, dùng `n_type=1` cho kết quả chính, hệ số `q` từ 0,20 đến 1,99. Đây là hỗn hợp **ba** artifact trong từng mẫu; thí nghiệm chọn riêng BW/EM/MA hoặc chỉ BW phải có bảng riêng.
3. Chia train/validation từ nguồn train với `test_size=0.3`; **không có record test trong hai tập này**. Để tái lập chính lần chạy, lưu `train_idx`/`val_idx` và manifest record/mẫu. Nếu chọn validation theo record để giảm leakage thì áp dụng cho mọi mô hình và ghi rõ đó là biến thể khác so với `main.py` TFCDiff.
4. Nếu có **cặp test chính xác đã dùng để công bố số của TFCDiff**, ưu tiên dùng lại cặp đó. Nếu không có, sinh **một tập test mới duy nhất** bằng mã TFCDiff, lưu `clean_test/noisy_test` cùng `sample_id`, `record_id`, `q`, `n_type`, seed, phiên bản preprocessing và hash tệp. Chạy **cả checkpoint TFCDiff lẫn mô hình bạn trên chính tập test đã lưu này**. Chỉ lưu `rnd_*.npy` không đủ vì còn vị trí cắt noise và hệ số trộn ngẫu nhiên.
5. Lưu cả `clean_train/noisy_train`, `clean_val/noisy_val` và chỉ số chia tập dùng huấn luyện mô hình bạn. Các cặp train tạo lại có thể **khác cặp đã huấn luyện checkpoint TFCDiff**, vì repository không kèm nguyên tensor train gốc; ghi rõ giới hạn này. Không để từng notebook tự tạo test riêng.
6. Kiểm tra shape `(N,1,3600)`, `float32`, hữu hạn, cùng thứ tự `sample_id`; kiểm tra các record test không lọt vào train/validation. Đối chiếu số lượng với thông tin README (**32.578 nguồn train, 5.012 test**) nhưng chỉ công bố số thực tế của dataset đã lưu và giải thích mọi khác biệt.

### 11.2. Huấn luyện mô hình bạn, suy luận bằng checkpoint TFCDiff có sẵn

| Thành phần | Quy tắc cho phép so sánh chính |
|---|---|
| TFCDiff | Dùng `check_points/model.pth` có sẵn làm **baseline cố định**, đặt `model.eval()` và chạy inference trên `noisy_test` đã khóa. `check_points/readme` nói checkpoint này giống exp5; hash của `model.pth` trùng `Ablation_Study/Checkpoints/exp5_noise_type_1_2026-04-11_16-38-58/model.pth`. Giữ DCT 3.600→1.000, `eta=3`, reverse DDPM và inverse DCT theo `utils.py`/`complete_eval.py`. Không huấn luyện lại. |
| Mô hình đề xuất | Huấn luyện kiến trúc HNF–FiLM–attention **1 kênh, 3.600 mẫu** trên QTDB train records và vùng NSTDB train theo giao thức gốc; chọn checkpoint chỉ bằng validation đã lưu. Không khẳng định các cặp nhiễu train mới trùng từng mẫu với dữ liệu lịch sử của checkpoint TFCDiff. Các hyperparameter/loss/EMA/DDIM riêng được phép, nhưng phải công bố. |
| Chuẩn hóa riêng của mô hình bạn | Nếu dùng `s_i=max|noisy_i|` từ `my_research.md`, chỉ tính từ **noisy input** của từng mẫu, áp cùng scale lên clean target khi train, rồi **nhân lại `s_i`** ở prediction trước khi chấm. Không đưa `clean_test` vào phép tính scale. |
| Test | Mỗi mô hình nhận đúng cùng `noisy_test`; lưu prediction theo `sample_id` với shape `(N_test,1,3600)` và thang biên độ như `clean_test`. Test không dùng để chọn epoch, seed, loss weight hoặc sampler. Giữ nguyên checkpoint TFCDiff sau khi nhìn kết quả test. |
| Inference | Báo cáo rõ TFCDiff DDPM 50 bước, 1-shot và mô hình bạn DDIM 50 bước, 1-shot, `eta=0` nếu giữ cấu hình trong `my_research.md`. Ghi seed và latency. So sánh chất lượng được, nhưng thời gian chạy cần ghi kèm sampler và phần cứng. |

`complete_eval.py` mặc định đọc `./check_points/model.pth`, nhưng còn gọi `Data_Preparation(1)` trước khi kiểm tra cache `X_test.pt/y_test.pt`. Để bảo đảm hai mô hình nhận **đúng cùng test pairs**, nên viết driver đánh giá chung đọc dataset đã khóa và gọi phần inference của TFCDiff trực tiếp; không để script này tạo lại noisy test trong từng lần chạy. `my_research.md` hiện mô tả benchmark BW-only; trước khi chạy mixed-noise cần cập nhật **nguồn dataset, manifest, tên output/checkpoint và notebook đánh giá** để hai nhánh không ghi đè nhau.

### 11.3. Chấm điểm và phạm vi kết luận

- Dùng **một script đánh giá chung** để căn `sample_id`, đọc `clean_test/noisy_test` và prediction của từng mô hình, rồi tính SSD, MAD, PRD theo `utils.py`, CosSim, SNR_in, SNR_out, ImSNR. Giữ batch size/nhóm mẫu cố định cho PRD đặc thù của repo (mục 7); nếu dùng PRD chuẩn thì đặt tên cột riêng.
- Báo cáo ít nhất `N`, mean ± std toàn test, bốn dải `q` ở mục 7, checkpoint/epoch, số shot và runtime. Có thể bổ sung kết quả theo record và nhiều seed để kiểm tra độ ổn định.
- Bảng trên cùng test pairs cho phép kết luận **mô hình của bạn so với checkpoint TFCDiff có sẵn trên tập test đã khóa**. Đây là phép so trực tiếp giữa hai prediction trên cùng dữ liệu, dù cặp train của hai checkpoint có thể không hoàn toàn trùng nhau. Muốn đối chiếu trực tiếp với **con số công bố của bài**, phải xác nhận dataset/preprocessing/split/test pairs/metric/phiên bản mã khớp bài; cùng tên QTDB và cùng mức `q` chưa đủ.
- Chỉ nói vượt các mô hình trong `Comparative_study` khi chúng cũng được chạy trên **cùng test pairs và cùng script metric** (hoặc khi đã chứng minh số công bố dùng đúng giao thức). Kết quả TFCDiff vượt baseline trong bài không tự động chứng minh mô hình của bạn vượt baseline đó trên dataset mới.

### 11.4. Điều kiện hiện tại của workspace

Tại lần kiểm tra này, `check_points/model.pth` và `final.pth` **đã có sẵn**; `data/` chưa chứa QTDB/NSTDB đã giải nén và không có `X_test.pt/y_test.pt` ở gốc. Tệp `common_qtdb_bw_3600.npz` là **BW-only** với 5.728 train, 2.418 validation và 1.252 test; nó không thay thế test mixed-noise. Cần có cặp mixed-noise test cố định trước khi chạy đánh giá chung, nhưng **không cần train lại TFCDiff**.

