"""Turn the requested QTDB training notebook into a fresh 3600-point mixed run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle_train_proposed_research_mixed_3600.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_train_qtdb_bw_em_ma.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, got {source.count(old)}")
    return source.replace(old, new)


class Cell(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = [Cell(cell) for cell in notebook["cells"]]
notebook["cells"] = cells
for cell in cells:
    cell.source = "".join(cell.source)
cells[0].source = """# QTDB 3600 — một Conditional DDPM cho nhiễu trộn BW+EM+MA

Huấn luyện **mới từ epoch 0** bằng kiến trúc HNF–FiLM–attention và ResearchLossDDPM của nghiên cứu. Mỗi cặp train/validation được tạo bằng công thức trộn đồng thời ba nguồn NSTDB BW, EM, MA (n_type=1); không train ba model riêng. Tín hiệu sạch và test cố định lấy từ bộ common QTDB 3600. Không nạp checkpoint của lần chạy cũ trong phiên đầu.

Các hệ số biên `a1` hoặc `a2` bằng 0/100 có thể làm một thành phần bằng 0 ở một số ví dụ; đây là đúng miền hệ số của giao thức TFCDiff, không phải lỗi chọn một loại nhiễu thay vì trộn.
"""
cells[1].source = """## 1. Input Kaggle

- Gắn Dataset chứa `common_qtdb_mixed_bwem_ma_3600_ntype1.npz`: notebook dùng `clean_train`, `clean_val`, `clean_test`, `noisy_test`, ID và mức `q`. Không dùng `noisy_train/noisy_val` trong NPZ để train; chúng được sinh lại từ ba nguồn nhiễu trong cell dữ liệu.
- Bật GPU và Internet để đọc NSTDB qua WFDB. Lần chạy đầu **không cần `last_*.pth`**; để `START_FROM_SCRATCH=True`.
- Bộ test giữ nguyên `noisy_test`/`clean_test`/`sample_id_test` trong NPZ. Nếu dùng kết quả với các model khác, chúng phải đánh giá trên đúng cặp test này và cùng script metric.

Việc tạo lại nhiễu train làm cặp train khác với bản NPZ đóng băng; do đó đây là so sánh trên **cùng test**, không khẳng định cùng chính xác cặp train với checkpoint TFCDiff lịch sử.
"""
cells[2].source = """## 2. Cấu hình nghiên cứu và giao thức

ECG 1 kênh, 360 Hz, 3600 mẫu. Với `n_type=1`, mỗi nguồn BW/EM/MA lấy kênh 1, nửa đầu cho train/validation; test cố định trong NPZ. Mỗi clean segment được ghép một cặp nhiễu duy nhất trước khi huấn luyện, dùng seed cố định. Công thức trộn giữ nguyên dấu ngoặc và chỉ chia 100 một lần theo `compare.md`. Mức `q` lấy từ manifest NPZ. Mô hình: U-Net HNF–FiLM–attention, DDPM 50 bước, ResearchLossDDPM, Adam 1e-4, EMA 0.9.
"""
cells.insert(3, Cell({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": "%pip -q install 'wfdb>=4.1,<5'"}))

config = cells[4].source
config = replace_once(config, "REQUIRE_RESUME_CHECKPOINT = True", "REQUIRE_RESUME_CHECKPOINT = False")
config = replace_once(config, "MODE = 'train'", "START_FROM_SCRATCH = True  # True cho run đầu: bỏ qua mọi last checkpoint\nMODE = 'train'")
config = replace_once(config, "/kaggle/working/proposed_research_mixed_3600", "/kaggle/working/qtdb_mixed_bwem_ma_3600_fresh")
config = replace_once(config, "'name': 'proposed_research_ddpm_tfcdiff_mixed_bwem_ma_ntype1'", "'name': 'qtdb_mixed_bwem_ma_3600_fresh_generated_train'")
config = replace_once(config, "'data_protocol': 'tfcdiff_mixed_bwem_ma', 'n_type': 1,", "'data_protocol': 'tfcdiff_mixed_bwem_ma_generated_train', 'n_type': 1,\n        'mixed_train_seed': 1234,")
cells[4].source = config

# In the source notebook the data loader is cell 7; inserting the install cell shifts it to 8.
data = cells[8].source
marker = "if CLEAN_TRAIN.shape != NOISY_TRAIN.shape or CLEAN_VAL.shape != NOISY_VAL.shape:"
if data.count(marker) != 1:
    raise RuntimeError("Cannot locate train/validation pair check")
mix_code = '''# Regenerate train/validation noisy pairs from all three NSTDB artifacts.
# n_type=1: first temporal half, first channel. Test remains the locked NPZ test.
import wfdb
from scipy import signal

MIX_SEED = 1234
MIX_LENGTH = 3600
NOISE_TRAIN = {}
for artifact in ('BW', 'EM', 'MA'):
    record = wfdb.rdrecord(artifact.lower(), pn_dir='nstdb/1.0.0')
    values = np.asarray(record.p_signal, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f'NSTDB {artifact}: expected two channels, got {values.shape}')
    if int(record.fs) != 360:
        target_length = int(round(len(values) * 360 / float(record.fs)))
        values = signal.resample(values, target_length, axis=0).astype(np.float32)
    midpoint = len(values) // 2
    NOISE_TRAIN[artifact] = np.ascontiguousarray(values[:midpoint, 0])
    if len(NOISE_TRAIN[artifact]) < MIX_LENGTH or not np.isfinite(NOISE_TRAIN[artifact]).all():
        raise ValueError(f'Invalid NSTDB {artifact} train region')
    print(f'NSTDB {artifact}: ch1_a, {len(NOISE_TRAIN[artifact])} samples')

def make_three_artifact_pairs(clean, q, seed, split_name):
    rng = np.random.RandomState(seed)
    q = np.asarray(q, dtype=np.float32)
    if len(q) != len(clean) or not np.isfinite(q).all() or np.any((q < .20) | (q > 1.99)):
        raise ValueError(f'{split_name}: invalid q levels')
    mixed_noisy = np.empty_like(clean, dtype=np.float32)
    audit = []
    for index in tqdm(range(len(clean)), desc=f'Mix BW+EM+MA {split_name}'):
        components = {}
        starts = {}
        for artifact in ('BW', 'EM', 'MA'):
            source = NOISE_TRAIN[artifact]
            start = int(rng.randint(0, len(source) - MIX_LENGTH + 1))
            starts[artifact] = start
            components[artifact] = source[start:start + MIX_LENGTH]
        a1 = int(rng.randint(0, 101))
        a2 = int(rng.randint(0, 101))
        # Exact TFCDiff expression: only one /100, outside the whole sum.
        mix = (a1 * components['BW'] + (100 - a1) *
               (a2 * components['EM'] + (100 - a2) * components['MA'])) / 100.0
        clean_row = clean[index, 0]
        mix_range = float(np.ptp(mix))
        clean_range = float(np.ptp(clean_row))
        if mix_range <= 0 or clean_range <= 0:
            raise ValueError(f'{split_name} row {index}: flat clean or mixed noise')
        mixed_noisy[index, 0] = clean_row + float(q[index]) * clean_range / mix_range * mix
        audit.append((index, a1, a2, starts['BW'], starts['EM'], starts['MA'], float(q[index])))
    if not np.isfinite(mixed_noisy).all():
        raise FloatingPointError(f'{split_name}: nonfinite noisy pairs')
    audit_frame = pd.DataFrame(audit, columns=['row', 'a1', 'a2', 'bw_start', 'em_start', 'ma_start', 'q'])
    audit_frame.to_csv(REPORT_DIR / f'{split_name}_mix_manifest.csv', index=False)
    return mixed_noisy

del NOISY_TRAIN, NOISY_VAL
NOISY_TRAIN = make_three_artifact_pairs(CLEAN_TRAIN, TRAIN_MANIFEST.q.to_numpy(), MIX_SEED, 'train')
NOISY_VAL = make_three_artifact_pairs(CLEAN_VAL, VAL_MANIFEST.q.to_numpy(), MIX_SEED + 1, 'validation')
print('Generated mixed BW+EM+MA pairs:', NOISY_TRAIN.shape, NOISY_VAL.shape)

'''
data = data.replace(marker, mix_code + marker)
data = data.replace("'protocol_verified': protocol_verified, 'noise_version': noise_version,", "'protocol_verified': protocol_verified, 'generated_train_mixture': True,\n                'mixed_train_seed': MIX_SEED, 'noise_version': noise_version,")
cells[8].source = data

# After the inserted install cell, checkpoint logic is cell 14.
checkpoint = cells[14].source
for old, new in (
    ("last_proposed_research_mixed_3600.pth", "last_qtdb_mixed_bwem_ma_3600_fresh.pth"),
    ("best_proposed_research_mixed_3600.pth", "best_qtdb_mixed_bwem_ma_3600_fresh.pth"),
    ("training_proposed_research_mixed_3600.csv", "training_qtdb_mixed_bwem_ma_3600_fresh.csv"),
):
    checkpoint = checkpoint.replace(old, new)
checkpoint = replace_once(checkpoint, "resume_path = resolve_resume_checkpoint()", "resume_path = None if START_FROM_SCRATCH else resolve_resume_checkpoint()")
cells[14].source = checkpoint

for cell in cells[15:]:
    source = cell.source
    for old, new in (
        ("last_proposed_research_mixed_3600.pth", "last_qtdb_mixed_bwem_ma_3600_fresh.pth"),
        ("best_proposed_research_mixed_3600.pth", "best_qtdb_mixed_bwem_ma_3600_fresh.pth"),
        ("training_proposed_research_mixed_3600.csv", "training_qtdb_mixed_bwem_ma_3600_fresh.csv"),
        ("proposed_research_mixed_ntype1_seed1234.npz", "qtdb_mixed_bwem_ma_3600_fresh_seed1234.npz"),
        ("proposed_research_ddpm_tfcdiff_mixed_bwem_ma_ntype1", "qtdb_mixed_bwem_ma_3600_fresh_generated_train"),
        ("proposed_research_mixed_3600_outputs.zip", "qtdb_mixed_bwem_ma_3600_fresh_outputs.zip"),
    ):
        source = source.replace(old, new)
    cell.source = source

cells[24].source = replace_once(
    cells[24].source,
    "'protocol_verified': protocol_verified, 'noise_version': noise_version,",
    "'protocol_verified': protocol_verified, 'generated_train_mixture': True, "
    "'mixed_train_seed': MIX_SEED, 'noise_version': noise_version,",
)

cells[-1].source = """## Chạy từ đầu và lưu kết quả

Trong phiên Kaggle đầu, để `START_FROM_SCRATCH=True`: notebook khởi tạo model mới ở epoch 0, không cần gắn `last_*.pth`. `MAX_EPOCHS_THIS_SESSION=5` là giới hạn an toàn để đo thời gian; tăng sau khi xác nhận thời gian/epoch. Notebook vẫn lưu `last` và `best` sau mỗi epoch để có đường phục hồi nếu Kaggle hết thời gian. Nếu cần tiếp tục qua phiên khác, gắn output, đặt `START_FROM_SCRATCH=False`; khi đó checkpoint, optimizer, EMA, AMP scaler và RNG được khôi phục.

Train/validation: QTDB clean 3600 từ NPZ + hỗn hợp BW/EM/MA được tạo trong notebook. Test: **chỉ** `noisy_test` cố định trong NPZ; prediction được inverse-normalize và xuất theo `sample_id_test`. Không chọn checkpoint bằng test.
"""

for cell in cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

assert notebook["nbformat"] == 4
assert all(cell.cell_type in {"code", "markdown"} for cell in cells)
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET}")
