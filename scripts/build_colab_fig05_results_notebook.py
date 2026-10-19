"""Build self-contained Colab pilot inference and waveform Figure 5 notebook."""

import ast
import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'notebooks/kaggle_ablation_u4_all_noise_512_100ep.ipynb'
RUNTIME = ROOT / 'scripts/fig05_ecg_results.py'
DESTINATION = ROOT / 'notebooks/colab_draw_fig05_results.ipynb'


def main():
    training = nbf.read(TRAINING, as_version=4)
    runtime_source = RUNTIME.read_text(encoding='utf-8')
    tree = ast.parse(runtime_source)
    imports = '\n'.join(ast.get_source_segment(runtime_source, node) for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom)))
    functions = {node.name: ast.get_source_segment(runtime_source, node)
                 for node in tree.body if isinstance(node, ast.FunctionDef)}

    def helpers(*names):
        return '\n\n\n'.join(functions[name] for name in names)

    network_source = training.cells[6].source
    network = '\n\n\n'.join(ast.get_source_segment(network_source, node)
                             for node in ast.parse(network_source).body
                             if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)))
    diffusion = training.cells[8].source
    preprocessing = training.cells[10].source
    hashes = {name: hashlib.sha256(source.encode('utf-8')).hexdigest()
              for name, source in [('network', network), ('diffusion_loss_sampler', diffusion),
                                    ('preprocessing', preprocessing)]}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md('''# Hình 5 — ECG phục hồi thực tế từ checkpoint U4 MIX7–512

Notebook **suy luận, không huấn luyện**. Tự chứa đúng kiến trúc HNF–FiLM U-Net, diffusion/loss và sampler từ notebook U4; **không phải tải `src`**. Chỉ cần checkpoint `best_u4_mix7_protocol1_mixed512_100ep_v1.pth`.

**Colab:** bật GPU → Run all → chọn checkpoint nếu được hỏi → xem hình ngay ở mục 7–8. Nếu file nằm trong Drive, mount Drive và sửa `CHECKPOINT_PATH` ở mục 2. Không tự tải ảnh xuống.

**Phạm vi mặc định:** kiểm tra định tính nhỏ, 8 đoạn hợp lệ đầu tiên từ mỗi record `sel123`, `sel233`, tạo đủ 7 nhóm MIX7, tổng tối đa **112 cặp**. Đây **không phải đánh giá toàn bộ tập test**, không dùng các số trung bình này làm kết quả benchmark cuối. Chọn mẫu trước suy luận, không lựa theo chất lượng mô hình.

Đây là tín hiệu QTDB thực + artifact NSTDB cộng có kiểm soát, không phải nhiễu lâm sàng tự nhiên. Dữ liệu được tải trực tiếp từ PhysioNet và cache trong phiên; cần Internet. Không sử dụng `.npz` benchmark 3600 điểm.
'''),
        md('## 1. Cài thư viện còn thiếu (giữ PyTorch có sẵn của Colab)'),
        code('''import importlib.util
import subprocess
import sys

requirements = {'wfdb': 'wfdb>=4.1,<5', 'scipy': 'scipy>=1.10',
                'pandas': 'pandas>=2.0', 'matplotlib': 'matplotlib>=3.7', 'tqdm': 'tqdm>=4.62'}
missing = [package for module, package in requirements.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
print('Thư viện sẵn sàng; không cài lại torch.')
'''),
        md('''## 2. Tham số của lần thử

- `WEIGHTS_KIND='ema'`: dùng EMA **tại chính epoch được chọn bằng validation của mạng raw**. Có thể chọn `raw`, nhưng không đổi qua lại theo kết quả test để chọn bản đẹp hơn.
- `NOISE_LEVEL_R=1.0`: tỉ lệ peak-to-peak của **tổng nhiễu**, **không phải SNR 1 dB**. Cộng các patch trước, rồi scale tổng đúng một lần.
- Protocol 1: dùng **nửa sau NSTDB, kênh chỉ số 1**; train dùng nửa đầu, kênh chỉ số 0.
- Giữ sampler quadratic-tau nguyên bản U4, DDIM 50 bước, eta 0, 1 shot. Cố định seed, thứ tự mẫu, batch; khác thiết bị/phiên bản có thể sai khác số học.
- `PREVIEW_SAMPLE_ID=None`: lấy mẫu clean đầu tiên đã chọn, dùng cùng mẫu cho các nhóm nhiễu. Có thể nhập mã cụ thể sau khi xem manifest.
'''),
        code(imports + '''

CHECKPOINT_PATH = 'best_u4_mix7_protocol1_mixed512_100ep_v1.pth'
# Nếu dùng Drive: mount thủ công rồi thay CHECKPOINT_PATH bằng đường dẫn đầy đủ.
WEIGHTS_KIND = 'ema'  # 'ema' hoặc 'raw'; không tự fallback
TEST_RECORD_NAMES = ('sel123', 'sel233')
MAX_BEATS_PER_RECORD = 8
NOISE_GROUPS = ('BW', 'EM', 'MA', 'BW+EM', 'BW+MA', 'EM+MA', 'BW+EM+MA')
NOISE_LEVEL_R = 1.0
DDIM_STEPS, DDIM_ETA, NUM_SHOTS = 50, 0.0, 1
BATCH_SIZE = 8
INFERENCE_SEED = 1234
PREVIEW_SAMPLE_ID = None
FIGURE_GROUPS = ('BW', 'EM', 'MA', 'BW+EM+MA')
PREVIEW_DPI = 150
GRAYSCALE = False
CACHE_DIR = Path('/content/fig05_cache') if Path('/content').exists() else Path('output/fig05_cache')
EXPORT_DIR = Path('/content/fig05_output') if Path('/content').exists() else Path('output/fig05_output')

TARGET_FS, LENGTH, MAX_UNPADDED, INSERT_OFFSET = 360, 512, 496, 16
PHYSIONET_QT, PHYSIONET_NST = 'qtdb/1.0.0', 'nstdb/1.0.0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(4, torch.get_num_threads()))
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
print('Thiết bị:', DEVICE)
if DEVICE.type != 'cuda':
    print('CPU vẫn chạy được, nhưng nên bật GPU trong Runtime > Change runtime type.')
'''),
        code(helpers('sha256_file', 'sha_array', 'wfdb_read_with_retry', 'resolve_checkpoint')),
        md('''## 3. Kiến trúc và diffusion nguyên bản (đã nhúng)

Khối HNF, FiLM, attention, U-Net và các buffer diffusion giữ nguyên từ `kaggle_ablation_u4_all_noise_512_100ep.ipynb`. Kiểm tra state dict bằng `strict=True`; không bỏ qua trọng số thiếu/thừa. Mã loss được giữ để tương thích lớp checkpoint, nhưng notebook này **không gọi huấn luyện**.
'''),
        code(network),
        code(diffusion),
        code(helpers('load_u4_checkpoint') + '''

checkpoint_path = resolve_checkpoint(CHECKPOINT_PATH)
MODEL, PROVENANCE = load_u4_checkpoint(checkpoint_path, WEIGHTS_KIND, DEVICE)
'''),
        md('''## 4. QTDB test và NSTDB giữ riêng

Tiền xử lý giống train: `.pu1`, mốc bắt đầu trước P 40 ms; resample có reflect padding 100 ms về 360 Hz; loại đoạn không hợp lệ; trừ trung bình hai đầu; chèn tối đa 496 điểm vào tensor 512 tại offset 16. Không kéo giãn từng nhịp thành 512 điểm.

Chỉ chọn record trong danh sách test của checkpoint và **không bỏ qua record khi tải lỗi**. Kiểm tra hash của toàn bộ BW/EM/MA với fingerprint lúc train. Cache tạo ở `CACHE_DIR`; nếu cache lỗi, chỉ xóa đúng file cache lỗi rồi chạy lại.
'''),
        code(preprocessing),
        code(helpers('prepare_pilot_data') + '''

CLEAN_RAW, CLEAN_META, NOISE, AUDIT = prepare_pilot_data(
    PROVENANCE, TEST_RECORD_NAMES, MAX_BEATS_PER_RECORD, CACHE_DIR)
display(AUDIT[['record_id', 'accepted_segments', 'selected_for_pilot']])
display(CLEAN_META[['sample_id', 'record_id', 'unpadded_length']].head(10))
'''),
        md('''## 5. Tạo các cặp noisy test

Đầu vào mô hình và ECG tham chiếu dùng chung hệ số `max(abs(clean))`, đúng giao thức nghiên cứu hiện tại. **Đây là chuẩn hóa dựa vào tham chiếu**, chưa phải quy tắc triển khai trên ECG thực chưa biết clean. Không chuẩn hóa riêng prediction để làm hình đẹp hơn. Dùng biên độ chuẩn hóa trên hình, không gắn nhãn mV.
'''),
        code(helpers('make_pilot_pairs') + '''

CLEAN, NOISY, MANIFEST = make_pilot_pairs(
    CLEAN_RAW, CLEAN_META, NOISE, NOISE_GROUPS, NOISE_LEVEL_R)
print('Clean/noisy:', CLEAN.shape, NOISY.shape)
display(MANIFEST.groupby('noise_group', sort=False).size().rename('Số cặp pilot'))
'''),
        md('''## 6. Suy luận DDIM và metric

`DENOISED` là ECG **sau toàn bộ quá trình lấy mẫu**, không phải Gaussian do U-Net dự đoán. Không clamp đầu ra hoặc làm mượt hậu xử lý.

Lưu riêng metric **full512** (gồm padding) và **valid** (phần tín hiệu thực). Metric trên hình dùng **valid**, không trộn với số toàn đoạn. SNR và RMSE tính trên cùng cặp, cùng scale. Seed được đặt ngay trước suy luận. Kết quả pilot không thay thế benchmark chính thức.
'''),
        code(helpers('prediction_metrics', 'infer_pilot') + '''

DENOISED = infer_pilot(MODEL, CLEAN, NOISY, MANIFEST, PROVENANCE,
                       BATCH_SIZE, DDIM_STEPS, DDIM_ETA, NUM_SHOTS, INFERENCE_SEED, DEVICE)
METRICS = prediction_metrics(CLEAN, NOISY, DENOISED, MANIFEST)
METRICS['checkpoint_sha256'] = PROVENANCE['checkpoint_sha256']
METRICS['weights'] = WEIGHTS_KIND
print('Chỉ là thống kê của tập pilot nhỏ:')
display(METRICS.groupby('noise_group', sort=False)[[
    'rmse_valid', 'snr_in_db_valid', 'snr_out_db_valid', 'delta_snr_db_valid']].mean().round(4))
'''),
        md('''## 7. Hình 5 — bốn nhóm nhiễu, hiển thị trực tiếp

Mỗi hàng: waveform chồng tham chiếu–nhiễu–phục hồi; phóng to quanh đỉnh biên độ tuyệt đối của **tham chiếu** ±120 ms; sai số phục hồi trừ tham chiếu. Vùng zoom là quy tắc hình học, **không phải xác nhận vị trí sóng R hay đánh giá lâm sàng**.

Cùng thang biên độ giữa các hàng, cùng scale lỗi giữa các hàng. Loại padding khỏi trục thời gian nhưng **không loại khỏi input mô hình**. Mẫu mặc định được xác định trước theo thứ tự dữ liệu, không tự chọn mẫu có chất lượng tốt nhất.
'''),
        code(helpers('make_waveform_figure', 'show_figure') + '''

selected_sample = PREVIEW_SAMPLE_ID or str(CLEAN_META.sample_id.iloc[0])
show_figure(make_waveform_figure(CLEAN, NOISY, DENOISED, METRICS,
                               selected_sample, FIGURE_GROUPS, GRAYSCALE), PREVIEW_DPI)
display(METRICS.loc[METRICS.sample_id == selected_sample, [
    'pair_id', 'rmse_valid', 'snr_in_db_valid', 'snr_out_db_valid', 'delta_snr_db_valid']])
'''),
        md('''## 8. Kiểm tra riêng các tổ hợp trộn

Dùng cùng mẫu clean đang xem. Khi chỉnh giao diện, chỉ chạy lại mục 7 hoặc 8, **không cần chạy lại DDIM**. Nếu muốn xem một mã mẫu khác, sửa `PREVIEW_SAMPLE_ID` và chạy lại mục 7 (chạy lại cell tham số nếu thay biến ở đó), không chạy lại các bước tải và suy luận.
'''),
        code('''mixed_groups = tuple(g for g in NOISE_GROUPS if '+' in g)
if mixed_groups:
    show_figure(make_waveform_figure(CLEAN, NOISY, DENOISED, METRICS,
                                   selected_sample, mixed_groups, GRAYSCALE), PREVIEW_DPI)
'''),
        md('''## 9. Xuất khi đã kiểm tra (mặc định tắt)

Bật `EXPORT_FILES=True` và chạy cell này để lưu **`5.png` (600 dpi), `5.svg`**, arrays clean/noisy/denoised, manifest + metric và thông tin checkpoint/sampler/seed. Không tự tải xuống. Nếu thay mẫu hiển thị, chạy lại mục 7 trước khi xuất. Ảnh chỉ chứa các ví dụ pilot, không tự trở thành bằng chứng toàn tập test.
'''),
        code(helpers('export_results') + '''

EXPORT_FILES = False
if EXPORT_FILES:
    PROVENANCE['figure_sample_id'] = selected_sample
    PROVENANCE['figure_noise_groups'] = list(FIGURE_GROUPS)
    PROVENANCE['figure_metric_region'] = 'valid, excluding zero padding'
    PROVENANCE['figure_zoom_rule'] = 'reference absolute peak +/-120 ms'
    figure = make_waveform_figure(CLEAN, NOISY, DENOISED, METRICS,
                                 selected_sample, FIGURE_GROUPS, GRAYSCALE)
    export_results(EXPORT_DIR, figure, CLEAN, NOISY, DENOISED, METRICS, PROVENANCE)
    plt.close(figure)
else:
    print('Chưa xuất file; hình đã hiển thị phía trên. Cache dữ liệu vẫn được giữ trong phiên.')
'''),
        md(r'''## 10. Đưa lên Overleaf sau khi duyệt

Tải `5.png` lên cùng workspace Overleaf và dùng tên file trực tiếp:

```latex
\includegraphics[width=\textwidth]{5.png}
```

Chưa tự sửa `paper.tex`: cần duyệt waveform và xác nhận bộ ví dụ trước. Khi viết caption, ghi mã mẫu, U4 MIX7 512, loại trọng số, cấu hình DDIM, chuẩn hóa và phạm vi pilot. Không diễn giải cải thiện SNR của vài đoạn thành kết luận toàn bộ tập test hoặc bảo toàn chẩn đoán.
'''),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'colab': {'provenance': [], 'name': DESTINATION.name},
        'figure05': {'source_training_notebook': str(TRAINING.relative_to(ROOT)),
                     'source_sha256': hashes, 'scope': 'real-data held-out pilot',
                     'execution_status': 'not_run_on_colab'},
    })
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook-cell>', 'exec')
    nbf.validate(notebook)
    nbf.write(notebook, DESTINATION)
    print(DESTINATION)


if __name__ == '__main__':
    main()
