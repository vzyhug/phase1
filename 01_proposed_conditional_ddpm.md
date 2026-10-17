# Proposed Conditional DDPM — chuẩn bị model cho QTDB-BW-3600

## Mục tiêu

Huấn luyện lại mô hình của nghiên cứu trên input một đạo trình dài 3600 điểm, sau đó xuất prediction theo contract của notebook Kaggle. Checkpoint BW-512 hiện có chỉ nên dùng để smoke test hoặc báo cáo phụ.

## Nguyên tắc giữ nguyên mã nguồn

- Không viết lại HNFBlock, FiLM, self-attention, loss hoặc diffusion process.
- Dùng một training driver bên ngoài repo để cung cấp common train/validation tensors.
- Chỉ đổi cấu hình chiều dài input từ 512 thành 3600 nếu model có tham số này.
- Giữ `in_channels=2` cho phép nối `x_t` và noisy condition; giữ `out_channels=1`.

## Input chung

```text
clean_train: [N, 1, 3600]
noisy_train: [N, 1, 3600]
clean_val:   [N, 1, 3600]
noisy_val:   [N, 1, 3600]
```

Nếu dùng normalization, scale phải được tính từ noisy input và lưu lại để inverse-transform output trước khi benchmark.

## Việc cần kiểm tra trước khi train

1. Ba lần downsampling/upsampling phải trả output đúng 3600 điểm; 3600 chia hết cho 8 nên phù hợp với ba mức downsampling.
2. STFT loss phải xử lý được 3600 điểm với `n_fft=128`, `hop_length=64`.
3. Không dùng lại split hoặc noise generator riêng trong repo; sử dụng common manifest.
4. Checkpoint selection chỉ dựa trên validation set, không dựa vào 14 record test.
5. EMA phải được lưu và dùng nhất quán nếu EMA là cấu hình chính thức.

## Cấu hình khuyến nghị

Giữ các tham số nghiên cứu hiện tại trừ batch size nếu GPU Kaggle không đủ bộ nhớ:

```text
epochs: 400
learning_rate: 1e-4
diffusion_steps: 50
beta_start: 1e-4
beta_end: 0.5
schedule: quadratic
```

Giảm batch size không thay đổi định nghĩa mô hình. Ghi lại batch size thực tế trong metadata.

## Inference và export

- Kết quả chính: DDIM hoặc DDPM 1-shot theo cấu hình đã đăng ký trước.
- Multi-shot phải xuất thành experiment khác, không ghi đè prediction 1-shot.
- Prediction phải được inverse-normalize về cùng miền với `clean`.

```python
np.savez_compressed(
    "proposed_seed2024.npz",
    sample_id=sample_id,
    prediction=prediction,  # [N,1,3600] hoặc [N,3600]
    latency_ms=latency_ms,
)
```

## Điều kiện chấp nhận

- Load checkpoint bằng `strict=True` thành công.
- Input/output có shape `[N,1,3600]`.
- Không có NaN/Inf.
- Thứ tự `sample_id` được giữ nguyên.
- Không tính metric trong model script; notebook Kaggle sẽ tính lại toàn bộ.

