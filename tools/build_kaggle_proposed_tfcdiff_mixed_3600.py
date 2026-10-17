"""Build a Kaggle notebook for the proposed model on TFCDiff mixed BW+EM+MA data."""

from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "tools" / "build_kaggle_proposed_conditional_ddpm_3600.py"
BASE_NOTEBOOK = ROOT / "notebooks" / "kaggle_proposed_conditional_ddpm_3600.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_train_proposed_tfcdiff_mixed_3600.ipynb"


def source(text: str) -> list[str]:
    return text.splitlines(True)


# Refresh the base notebook first so its embedded model source is byte-for-byte current.
runpy.run_path(str(BASE_BUILDER), run_name="__main__")
notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0]["source"] = source("""# Proposed HNF–FiLM–Attention DDPM — TFCDiff mixed BW+EM+MA

Notebook Kaggle huấn luyện **kiến trúc hiện tại của đề tài** trên dữ liệu hỗn hợp BW+EM+MA theo `compare.md`:

- QTDB, một kênh/mẫu, 360 Hz, 10 giây, 3.600 mẫu;
- split test cố định 14 record và `n_type=1` của TFCDiff;
- train/validation 70/30 được khóa bằng seed và lưu chỉ số nếu input còn ở dạng source-train;
- U-Net 1D HNF–FiLM–attention + DDPM/L1–STFT loss được nhúng nguyên byte từ `src/`;
- chuẩn hóa max-abs lấy **chỉ từ noisy input**, checkpoint chọn chỉ bằng validation loss;
- export DDIM 50 bước, 1-shot, `eta=0`, không tính metric trong notebook train.

Đây là benchmark **mixed-noise riêng**. Không gắn `common_qtdb_bw_3600.npz` vì đó là benchmark BW-only khác protocol.
""")

cells[1]["source"] = source("""## 1. Input contract

Gắn một Kaggle Dataset chứa file `common_qtdb_mixed_bwem_ma_3600_ntype1.npz` đã sinh một lần theo `compare.md`.

NPZ đầy đủ nên có:

- `clean_test`, `noisy_test`, `sample_id_test`, `record_id_test`, `q_test`;
- metadata scalar: `protocol_name='tfcdiff_mixed_bwem_ma'`, `n_type=1`, `sampling_rate=360`, `segment_length=3600`;
- một trong hai dạng train:
  - đã chia: `clean_train/noisy_train`, `clean_val/noisy_val` và các trường `sample_id_*`, `record_id_*`, `q_*`; hoặc
  - chưa chia: `clean_source_train/noisy_source_train`, `sample_id_source_train`, `record_id_source_train`, `q_source_train`. Notebook sẽ chia 70/30 với `SPLIT_SEED=1234`, lưu `split_indices.npz` và fingerprint.

Khuyến nghị lưu thêm `a1_*`, `a2_*`, vị trí cắt BW/EM/MA và manifest CSV để audit. Với bundle tối giản không có metadata, notebook có thể chạy ở chế độ tương thích và ghi rõ `protocol_verified=False`; tuy nhiên nó vẫn chặn fingerprint shape của `common_qtdb_bw_3600.npz` cũ để tránh gắn nhãn sai BW-only thành mixed-noise.
""")

cells[2]["source"] = source("""## 2. Cấu hình Kaggle và thí nghiệm

`compare.md` cho phép hyperparameter riêng cho kiến trúc khác, nhưng yêu cầu công bố. Notebook giữ 400 epoch, Adam `1e-3`, weight decay `5e-4`, StepLR `(150, 0.1)` và clip gradient 1.0. Vì HNF–attention ở độ dài 3.600 không vừa batch vật lý 128 trên GPU Kaggle, mặc định dùng micro-batch 4 và cập nhật optimizer sau 128 mẫu bằng gradient accumulation; cả hai giá trị được ghi vào log/checkpoint. Do loss L1-sum + STFT-mean của mô hình không tuyến tính hoàn toàn theo cách gộp batch, đây không được tuyên bố là bitwise tương đương một forward batch 128.
""")

cells[3]["source"] = source(r'''from pathlib import Path
import gc
import base64
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
try:
    from IPython.display import display
except ImportError:
    display = print

# ---- Chỉ sửa khối này giữa các phiên Kaggle ----
MODE = 'train'                 # 'train' hoặc 'export'
TARGET_EPOCHS = 400
MAX_EPOCHS_THIS_SESSION = 5    # None = cố chạy đến TARGET_EPOCHS
MICRO_BATCH_SIZE = 4           # giảm 2/1 nếu CUDA OOM
REFERENCE_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = REFERENCE_BATCH_SIZE // MICRO_BATCH_SIZE
NUM_WORKERS = 2
USE_AMP = True
EMA_DECAY = 0.9
SEED = 2024                    # model/training RNG
SPLIT_SEED = 1234             # khóa phép chia 70/30 source-train
ALLOW_UNVERIFIED_BUNDLE = True  # tương thích NPZ tối giản; xem cảnh báo data audit

TRAIN_BUNDLE_NAME = 'common_qtdb_mixed_bwem_ma_3600_ntype1.npz'
TEST_BUNDLE_NAME = TRAIN_BUNDLE_NAME
MANIFEST_NAME = 'mixed_manifest.csv'
RUN_EXPORT_WHEN_COMPLETE = True
ALLOW_INCOMPLETE_EXPORT = False
INFERENCE_BATCH_SIZE = 4
DDIM_STEPS = 50
DDIM_ETA = 0.0
NUM_SHOTS = 1

INPUT_ROOT = Path('/kaggle/input')
WORK_ROOT = Path('/kaggle/working/proposed_tfcdiff_mixed_3600')
SOURCE_SNAPSHOT = WORK_ROOT / 'source_snapshot'
CHECKPOINT_DIR = WORK_ROOT / 'checkpoints'
REPORT_DIR = WORK_ROOT / 'reports'
for directory in (SOURCE_SNAPSHOT, CHECKPOINT_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Bật GPU Accelerator trong Kaggle trước khi chạy notebook.')
if MODE not in {'train', 'export'}:
    raise ValueError(MODE)
if REFERENCE_BATCH_SIZE % MICRO_BATCH_SIZE:
    raise ValueError('REFERENCE_BATCH_SIZE phải chia hết cho MICRO_BATCH_SIZE')
if TARGET_EPOCHS != 400:
    print('WARNING: compare.md mô tả 400 epoch; hiện tại:', TARGET_EPOCHS)
if NUM_SHOTS != 1:
    raise ValueError('Benchmark chính là 1-shot.')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

MODEL_CONFIG = {
    'train': {
        'feats': 80, 'batch_size': REFERENCE_BATCH_SIZE,
        'micro_batch_size': MICRO_BATCH_SIZE, 'gradient_accumulation_steps': GRAD_ACCUM_STEPS,
        'epochs': TARGET_EPOCHS, 'lr': 1e-3, 'weight_decay': 5e-4,
        'scheduler': {'name': 'StepLR', 'step_size': 150, 'gamma': 0.1},
        'gradient_clip_norm': 1.0, 'lambda_time': 1.0, 'lambda_freq': 0.1,
    },
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
    'experiment': {
        'name': 'proposed_hnf_ddpm_tfcdiff_mixed_bwem_ma_ntype1',
        'input_length': 3600, 'sampling_rate': 360,
        'in_channels': 2, 'out_channels': 1,
        'normalization': 'per_sample_noisy_maxabs_shared_v1',
        'stft_n_fft': 128, 'stft_hop_length': 64,
        'ema_decay': EMA_DECAY, 'seed': SEED, 'split_seed': SPLIT_SEED,
        'data_protocol': 'tfcdiff_mixed_bwem_ma', 'n_type': 1,
    },
}

print('Device:', torch.cuda.get_device_name(0))
print('Mode:', MODE, '| target epochs:', TARGET_EPOCHS,
      '| max this session:', MAX_EPOCHS_THIS_SESSION)
print('Micro/effective batch:', MICRO_BATCH_SIZE, '/', REFERENCE_BATCH_SIZE,
      '| accumulation:', GRAD_ACCUM_STEPS, '| AMP:', USE_AMP)
''')

cells[6]["source"] = source("""## 4. Nạp và khóa common mixed-noise tensors

Cell này kiểm tra protocol, record leakage, shape, q và sample ID trước khi model nhìn thấy dữ liệu. Nếu input là source-train, phép chia 70/30 được tái lập deterministically và chỉ số được xuất ra để mọi mô hình dùng lại.
""")

cells[7]["source"] = source(r'''TEST_RECORDS = {
    'sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853', 'sel16420',
    'sel16795', 'sele0106', 'sele0121', 'sel32', 'sel49', 'sel14046', 'sel15814',
}

def unique_input(filename, required=True):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) > 1:
        raise RuntimeError(f'Nhiều file {filename}: {matches}')
    if required and not matches:
        raise FileNotFoundError(
            f'Không tìm thấy {filename}. Không dùng common_qtdb_bw_3600.npz vì đó là BW-only.')
    return matches[0] if matches else None

def as_ncl3600(array, name):
    value = np.asarray(array)
    if value.ndim == 2 and value.shape[1] == 3600:
        value = value[:, None, :]
    elif value.ndim == 3 and value.shape[1:] == (3600, 1):
        value = value.transpose(0, 2, 1)
    if value.ndim != 3 or value.shape[1:] != (1, 3600):
        raise ValueError(f'{name}: cần [N,1,3600], nhận {value.shape}')
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f'{name}: có NaN/Inf')
    return value

def scalar(bundle, key, default=None):
    if key not in bundle.files:
        if default is not None:
            return default
        raise KeyError(f'NPZ thiếu metadata bắt buộc: {key}')
    return np.asarray(bundle[key]).reshape(-1)[0].item()

def vector(bundle, key, count, dtype=str, default=None):
    if key not in bundle.files:
        if default is None:
            raise KeyError(f'NPZ thiếu {key}')
        value = np.asarray(default)
    else:
        value = np.asarray(bundle[key]).reshape(-1)
    value = value.astype(dtype)
    if len(value) != count:
        raise ValueError(f'{key}: {len(value)} != {count}')
    return value

train_bundle_path = unique_input(TRAIN_BUNDLE_NAME)
train_bundle = np.load(train_bundle_path, allow_pickle=True)
metadata_keys = {'protocol_name', 'n_type', 'sampling_rate', 'segment_length'}
protocol_verified = metadata_keys.issubset(train_bundle.files)
if not protocol_verified and not ALLOW_UNVERIFIED_BUNDLE:
    raise KeyError(f'NPZ thiếu metadata {sorted(metadata_keys-set(train_bundle.files))}')
protocol_name = str(scalar(train_bundle, 'protocol_name', 'tfcdiff_mixed_bwem_ma')).lower()
n_type = int(scalar(train_bundle, 'n_type', 1))
sampling_rate = int(scalar(train_bundle, 'sampling_rate', 360))
segment_length = int(scalar(train_bundle, 'segment_length', 3600))
if protocol_name != 'tfcdiff_mixed_bwem_ma':
    raise ValueError(f'protocol_name={protocol_name!r}; cần tfcdiff_mixed_bwem_ma')
if n_type != 1:
    raise ValueError('Kết quả chính trong compare.md yêu cầu n_type=1')
if sampling_rate != 360 or segment_length != 3600:
    raise ValueError(f'Cần 360 Hz/3600 mẫu; nhận {sampling_rate} Hz/{segment_length}')
if not protocol_verified:
    print('WARNING: NPZ không có protocol metadata; đang chạy compatibility mode.')
    print('WARNING: chỉ tiếp tục nếu file này thực sự là mixed BW+EM+MA n_type=1 theo compare.md.')
    print('Available keys:', train_bundle.files)

has_presplit = {'clean_train', 'noisy_train', 'clean_val', 'noisy_val'}.issubset(train_bundle.files)
has_source = {'clean_source_train', 'noisy_source_train'}.issubset(train_bundle.files)
if has_presplit == has_source:
    raise ValueError('NPZ phải có đúng một dạng: pre-split hoặc source-train chưa chia')

def make_frame(bundle, suffix, count, split):
    generated_ids = np.asarray([f'{split}:{index:08d}' for index in range(count)])
    sample_id = vector(bundle, f'sample_id_{suffix}', count, str, generated_ids)
    inferred_records = np.asarray([
        value.split(':')[1] if len(value.split(':')) >= 3 else 'unknown'
        for value in sample_id
    ])
    record_id = vector(bundle, f'record_id_{suffix}', count, str, inferred_records)
    q = vector(bundle, f'q_{suffix}', count, np.float32,
               np.full(count, np.nan, dtype=np.float32))
    if len(set(sample_id.tolist())) != count:
        raise ValueError(f'sample_id_{suffix} bị trùng')
    finite_q = np.isfinite(q)
    if (((q < .20) | (q > 1.99)) & finite_q).any():
        raise ValueError(f'q_{suffix} ngoài [0.20, 1.99]')
    return pd.DataFrame({'sample_id': sample_id, 'record_id': record_id,
                         'q': q, 'split': split, 'array_row': np.arange(count)})

if has_presplit:
    CLEAN_TRAIN = as_ncl3600(train_bundle['clean_train'], 'clean_train')
    NOISY_TRAIN = as_ncl3600(train_bundle['noisy_train'], 'noisy_train')
    CLEAN_VAL = as_ncl3600(train_bundle['clean_val'], 'clean_val')
    NOISY_VAL = as_ncl3600(train_bundle['noisy_val'], 'noisy_val')
    TRAIN_MANIFEST = make_frame(train_bundle, 'train', len(CLEAN_TRAIN), 'train')
    VAL_MANIFEST = make_frame(train_bundle, 'val', len(CLEAN_VAL), 'validation')
    split_origin = 'pre_split_input'
    split_indices_path = None
    known_bw_only_shape = (len(CLEAN_TRAIN), len(CLEAN_VAL), len(train_bundle['clean_test']))
    if known_bw_only_shape == (5728, 2418, 1252) and not protocol_verified:
        raise RuntimeError(
            'Bundle có đúng fingerprint shape (5728,2418,1252) của common_qtdb_bw_3600.npz BW-only. '
            'Không thể dùng file này cho benchmark mixed BW+EM+MA; cần tạo/upload common mixed-noise thật.')
else:
    clean_source = as_ncl3600(train_bundle['clean_source_train'], 'clean_source_train')
    noisy_source = as_ncl3600(train_bundle['noisy_source_train'], 'noisy_source_train')
    if clean_source.shape != noisy_source.shape:
        raise ValueError('source clean/noisy shape mismatch')
    source_manifest = make_frame(train_bundle, 'source_train', len(clean_source), 'source_train')
    split_rng = np.random.RandomState(SPLIT_SEED)
    permutation = split_rng.permutation(len(clean_source))
    val_count = int(np.ceil(0.30 * len(clean_source)))
    val_indices, train_indices = permutation[:val_count], permutation[val_count:]
    CLEAN_TRAIN, NOISY_TRAIN = clean_source[train_indices], noisy_source[train_indices]
    CLEAN_VAL, NOISY_VAL = clean_source[val_indices], noisy_source[val_indices]
    TRAIN_MANIFEST = source_manifest.iloc[train_indices].copy().reset_index(drop=True)
    VAL_MANIFEST = source_manifest.iloc[val_indices].copy().reset_index(drop=True)
    TRAIN_MANIFEST['split'] = 'train'; TRAIN_MANIFEST['array_row'] = np.arange(len(TRAIN_MANIFEST))
    VAL_MANIFEST['split'] = 'validation'; VAL_MANIFEST['array_row'] = np.arange(len(VAL_MANIFEST))
    split_indices_path = REPORT_DIR / 'split_indices.npz'
    np.savez(split_indices_path, train_indices=train_indices, val_indices=val_indices,
             split_seed=np.asarray(SPLIT_SEED), validation_fraction=np.asarray(.30))
    split_origin = 'deterministic_70_30_source_train'
    del clean_source, noisy_source

if CLEAN_TRAIN.shape != NOISY_TRAIN.shape or CLEAN_VAL.shape != NOISY_VAL.shape:
    raise ValueError('Train/validation clean-noisy shape mismatch')
known_train_records = set(TRAIN_MANIFEST.record_id) - {'unknown'}
known_val_records = set(VAL_MANIFEST.record_id) - {'unknown'}
if known_train_records & TEST_RECORDS or known_val_records & TEST_RECORDS:
    raise RuntimeError('Record leakage: test record xuất hiện trong train/validation')

CLEAN_TEST_LOCKED = as_ncl3600(train_bundle['clean_test'], 'clean_test')
NOISY_TEST_LOCKED = as_ncl3600(train_bundle['noisy_test'], 'noisy_test')
if CLEAN_TEST_LOCKED.shape != NOISY_TEST_LOCKED.shape:
    raise ValueError('Test clean/noisy shape mismatch')
TEST_MANIFEST_LOCKED = make_frame(train_bundle, 'test', len(CLEAN_TEST_LOCKED), 'test')
actual_test_records = set(TEST_MANIFEST_LOCKED.record_id)
if actual_test_records != {'unknown'} and actual_test_records != TEST_RECORDS:
    raise RuntimeError(f'14 test records không khớp; thiếu={TEST_RECORDS-actual_test_records}, thừa={actual_test_records-TEST_RECORDS}')
if actual_test_records == {'unknown'}:
    print('WARNING: không có record_id_test và không suy ra được từ sample_id_test; không thể audit 14 records.')

manifest = pd.concat([TRAIN_MANIFEST, VAL_MANIFEST, TEST_MANIFEST_LOCKED], ignore_index=True)
if manifest.sample_id.duplicated().any():
    raise ValueError('sample_id phải duy nhất trên toàn bộ split')
manifest_origin = f'npz:{split_origin}'

def array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode()); digest.update(str(value.dtype).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()

DATA_FINGERPRINTS = {
    'bundle_file': sha256_file(train_bundle_path),
    'clean_train': array_sha256(CLEAN_TRAIN), 'noisy_train': array_sha256(NOISY_TRAIN),
    'clean_val': array_sha256(CLEAN_VAL), 'noisy_val': array_sha256(NOISY_VAL),
    'clean_test': array_sha256(CLEAN_TEST_LOCKED), 'noisy_test': array_sha256(NOISY_TEST_LOCKED),
    'manifest': hashlib.sha256(manifest.to_csv(index=False).encode('utf-8')).hexdigest(),
}
manifest.to_csv(REPORT_DIR / MANIFEST_NAME, index=False)
data_audit = pd.DataFrame([
    {'split': 'train', 'N': len(CLEAN_TRAIN), 'shape': str(CLEAN_TRAIN.shape), 'records': TRAIN_MANIFEST.record_id.nunique()},
    {'split': 'validation', 'N': len(CLEAN_VAL), 'shape': str(CLEAN_VAL.shape), 'records': VAL_MANIFEST.record_id.nunique()},
    {'split': 'test', 'N': len(CLEAN_TEST_LOCKED), 'shape': str(CLEAN_TEST_LOCKED.shape), 'records': TEST_MANIFEST_LOCKED.record_id.nunique()},
])
data_audit.to_csv(REPORT_DIR / 'data_audit.csv', index=False)
display(data_audit)
print('Protocol:', protocol_name, '| verified:', protocol_verified,
      '| n_type=1 | split:', split_origin)
train_bundle.close()
''')

# Names and experiment identity must never collide with the BW-only run.
replacements = {
    "BATCH_SIZE": "MICRO_BATCH_SIZE",
    "last_proposed_conditional_ddpm_3600.pth": "last_proposed_tfcdiff_mixed_3600.pth",
    "best_proposed_conditional_ddpm_3600.pth": "best_proposed_tfcdiff_mixed_3600.pth",
    "training_proposed_conditional_ddpm_3600.csv": "training_proposed_tfcdiff_mixed_3600.csv",
    "proposed_seed2024.npz": "proposed_tfcdiff_mixed_ntype1_seed2024.npz",
    "proposed_conditional_ddpm_qtdb_bw_3600": "proposed_hnf_ddpm_tfcdiff_mixed_bwem_ma_ntype1",
    "/kaggle/working/proposed_conditional_ddpm_3600_outputs.zip": "/kaggle/working/proposed_tfcdiff_mixed_3600_outputs.zip",
}
for index in range(8, len(cells)):
    text = "".join(cells[index]["source"])
    for old, new in replacements.items():
        text = text.replace(old, new)
    cells[index]["source"] = source(text)

cell23 = "".join(cells[23]["source"])
cell23 = cell23.replace(
    "'common_bundle': str(train_bundle_path), 'manifest_origin': manifest_origin,",
    "'common_bundle': str(train_bundle_path), 'manifest_origin': manifest_origin, "
    "'protocol_verified': protocol_verified, 'n_type': n_type, "
    "'sampling_rate': sampling_rate, 'segment_length': segment_length,")
cells[23]["source"] = source(cell23)

# Use the locked test tensors already validated in cell 7.
cells[18]["source"] = source("""## 10. Mở test chỉ sau khi checkpoint đã được khóa

Test tensors đã được fingerprint ở bước data audit, nhưng không đi vào DataLoader train/validation và không tham gia chọn checkpoint.
""")
cells[19]["source"] = source(r'''def load_test_contract():
    return (NOISY_TEST_LOCKED,
            TEST_MANIFEST_LOCKED.sample_id.astype(str).to_numpy(),
            TEST_MANIFEST_LOCKED.copy())

SHOULD_EXPORT = MODE == 'export' or (MODE == 'train' and RUN_EXPORT_WHEN_COMPLETE and TRAINING_COMPLETE)
if SHOULD_EXPORT and not TRAINING_COMPLETE and not ALLOW_INCOMPLETE_EXPORT:
    raise RuntimeError(f'Checkpoint mới có {completed_epochs}/{TARGET_EPOCHS} epoch; không export test.')
print('Export requested:', SHOULD_EXPORT)
''')

# Correct optimizer/scheduler and add gradient accumulation for effective batch 128.
cell13 = "".join(cells[13]["source"])
cell13 = cell13.replace(
    "optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG['train']['lr'])",
    "optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG['train']['lr'], "
    "weight_decay=MODEL_CONFIG['train']['weight_decay'])")
cell13 = cell13.replace(
    "scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=1.0)",
    "scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)")
cell13 = cell13.replace("drop_last=False", "drop_last=True")
cells[13]["source"] = source(cell13)

cell15 = "".join(cells[15]["source"])
old_loop = """        for clean, noisy in progress:
            clean = clean.to(DEVICE, non_blocking=True)
            noisy = noisy.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = model(clean, noisy)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Train loss NaN/Inf tại epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); ema.update(model)
            train_losses.append(float(loss.detach()))
            progress.set_postfix(loss=f'{np.mean(train_losses[-20:]):.5f}')
"""
new_loop = """        optimizer.zero_grad(set_to_none=True)
        usable_batches = (len(loader) // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
        for batch_index, (clean, noisy) in enumerate(progress):
            if batch_index >= usable_batches:
                break  # tương đương drop_last ở effective batch 128
            clean = clean.to(DEVICE, non_blocking=True)
            noisy = noisy.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = model(clean, noisy)
                backward_loss = loss / GRAD_ACCUM_STEPS
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Train loss NaN/Inf tại epoch {epoch + 1}')
            scaler.scale(backward_loss).backward()
            if (batch_index + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            train_losses.append(float(loss.detach()))
            progress.set_postfix(loss=f'{np.mean(train_losses[-20:]):.5f}')
"""
if old_loop not in cell15:
    raise RuntimeError("Base training loop changed; update mixed notebook builder")
cell15 = cell15.replace(old_loop, new_loop)
cell15 = cell15.replace(
    "'physical_batch_size': MICRO_BATCH_SIZE, 'amp': USE_AMP",
    "'micro_batch_size': MICRO_BATCH_SIZE, 'effective_batch_size': REFERENCE_BATCH_SIZE, "
    "'gradient_accumulation_steps': GRAD_ACCUM_STEPS, 'amp': USE_AMP")
cells[15]["source"] = source(cell15)

cells[24]["source"] = source("""## Kết quả và cách tiếp tục

- Nếu chưa đủ `400/400`: **Save Version**, gắn output vừa tạo làm Input của version tiếp theo rồi Run All; notebook tự resume từ `last_proposed_tfcdiff_mixed_3600.pth`.
- Khi đủ 400 epoch, output benchmark là `proposed_tfcdiff_mixed_ntype1_seed2024.npz`.
- File này chỉ chứa prediction/ID/latency. Dùng notebook đánh giá chung để ghép theo `sample_id` và tính metric cho cả 5 mô hình.
- Không đưa artifact hoặc checkpoint này vào bảng BW-only.
""")

notebook["metadata"].setdefault("kaggle", {})["accelerator"] = "gpu"
notebook["metadata"]["kaggle"]["internet"] = False
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {TARGET}")
