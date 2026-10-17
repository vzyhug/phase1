"""Build the Kaggle training/export notebook described by 01_proposed_conditional_ddpm.md."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "kaggle_proposed_conditional_ddpm_3600.ipynb"
REQUIRED_SOURCE_FILES = (
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/main_model.py",
    "src/models/unet_1d.py",
    "src/models/blocks/__init__.py",
    "src/models/blocks/hnf_block.py",
    "src/models/blocks/bridge_block.py",
    "src/models/blocks/attention_1d.py",
)
EMBEDDED_SOURCE_B64 = {
    relative: base64.b64encode((ROOT / relative).read_bytes()).decode("ascii")
    for relative in REQUIRED_SOURCE_FILES
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


AUTO_DATA_SOURCE = r'''COMMON_DATA_DIR = WORK_ROOT / 'common_data'
COMMON_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR = WORK_ROOT / 'physionet_raw'

attached_train_bundles = sorted(INPUT_ROOT.rglob(TRAIN_BUNDLE_NAME))
attached_manifests = sorted(INPUT_ROOT.rglob(MANIFEST_NAME))
if len(attached_train_bundles) > 1 or len(attached_manifests) > 1:
    raise RuntimeError('Chỉ gắn một common-data cache/output gần nhất')

NEED_TO_GENERATE_COMMON_DATA = not attached_train_bundles and not attached_manifests
if bool(attached_train_bundles) != bool(attached_manifests):
    raise RuntimeError('Cache không đầy đủ: cần cả NPZ và common_manifest.csv')

if NEED_TO_GENERATE_COMMON_DATA:
    print('Không có common data cache: bắt đầu tạo trực tiếp từ QTDB + NSTDB.')
    print('Lần đầu cần bật Kaggle Internet; các lần resume dùng cache trong output trước.')
    try:
        import wfdb
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'wfdb>=4.1,<5'], check=True)
        import wfdb
    from fractions import Fraction
    from scipy.signal import resample_poly

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def download_and_extract(name, url):
        archive = RAW_DATA_DIR / name
        if not archive.is_file():
            partial = archive.with_suffix(archive.suffix + '.part')
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 ECG-research/1.0'})
            with urllib.request.urlopen(request, timeout=300) as response, open(partial, 'wb') as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            partial.replace(archive)
        if not zipfile.is_zipfile(archive):
            raise zipfile.BadZipFile(archive)
        marker = RAW_DATA_DIR / f'.{name}.extracted'
        if not marker.is_file():
            with zipfile.ZipFile(archive) as zipped:
                if zipped.testzip() is not None:
                    raise zipfile.BadZipFile(f'CRC failure: {archive}')
                zipped.extractall(RAW_DATA_DIR)
            marker.write_text('ok', encoding='utf-8')

    download_and_extract(
        'qt-database-1.0.0.zip',
        'https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip')
    download_and_extract(
        'mit-bih-noise-stress-test-database-1.0.0.zip',
        'https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip')

    bw_headers = sorted(RAW_DATA_DIR.rglob('bw.hea'))
    if len(bw_headers) != 1:
        raise RuntimeError(f'Cần đúng một bw.hea, tìm thấy {bw_headers}')
    bw_header = bw_headers[0]
    qtdb_headers = sorted(path for path in RAW_DATA_DIR.rglob('*.hea')
                          if path.name.startswith('sel'))
    if not qtdb_headers:
        raise RuntimeError('Không tìm thấy QTDB record headers')
    qtdb_by_name = {path.stem: path for path in qtdb_headers}

    TEST_RECORDS = (
        'sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853', 'sel16420',
        'sel16795', 'sele0106', 'sele0121', 'sel32', 'sel49', 'sel14046', 'sel15814',
    )
    missing_test = sorted(set(TEST_RECORDS) - set(qtdb_by_name))
    if missing_test:
        raise RuntimeError(f'QTDB thiếu test records: {missing_test}')
    candidate_train_records = sorted(set(qtdb_by_name) - set(TEST_RECORDS))
    split_rng = np.random.RandomState(SEED)
    shuffled_records = np.asarray(candidate_train_records, dtype=object)
    split_rng.shuffle(shuffled_records)
    validation_count = max(1, int(round(VALIDATION_RECORD_FRACTION * len(shuffled_records))))
    VALIDATION_RECORDS = tuple(sorted(shuffled_records[:validation_count].tolist()))
    TRAIN_RECORDS = tuple(sorted(shuffled_records[validation_count:].tolist()))
    if set(TRAIN_RECORDS) & set(VALIDATION_RECORDS) or set(TRAIN_RECORDS) & set(TEST_RECORDS):
        raise RuntimeError('Record leakage trong split')

    def resample_to_360(signal, source_fs):
        if float(source_fs) == 360.0:
            return np.asarray(signal, dtype=np.float32)
        ratio = Fraction(360, int(round(float(source_fs)))).limit_denominator(1000)
        return resample_poly(np.asarray(signal, dtype=np.float64), ratio.numerator,
                             ratio.denominator).astype(np.float32)

    def clean_windows(record_names, split):
        segments, rows = [], []
        for record_name in tqdm(record_names, desc=f'QTDB {split}'):
            path = qtdb_by_name[record_name].with_suffix('')
            record = wfdb.rdrecord(str(path))
            signal = resample_to_360(record.p_signal[:, 0], record.fs)
            for start in range(0, len(signal) - 3600 + 1, WINDOW_STRIDE):
                segment = signal[start:start + 3600]
                if not np.isfinite(segment).all() or float(np.ptp(segment)) <= 1e-8:
                    continue
                row_index = len(segments)
                segments.append(segment.astype(np.float32, copy=False))
                rows.append({'sample_id': f'{split}:{record_name}:{start}',
                             'patient_id': record_name, 'record_id': record_name,
                             'split': split, 'array_row': row_index,
                             'clean_segment_id': f'{record_name}:{start}:3600',
                             'clean_start_360hz': start})
        if not segments:
            raise RuntimeError(f'Không tạo được clean segments cho {split}')
        return np.stack(segments)[:, None, :], pd.DataFrame(rows)

    CLEAN_TRAIN_GENERATED, TRAIN_META_GENERATED = clean_windows(TRAIN_RECORDS, 'train')
    CLEAN_VAL_GENERATED, VAL_META_GENERATED = clean_windows(VALIDATION_RECORDS, 'validation')
    CLEAN_TEST_GENERATED, TEST_META_GENERATED = clean_windows(TEST_RECORDS, 'test')

    bw_record = wfdb.rdrecord(str(bw_header.with_suffix('')))
    bw = np.stack([resample_to_360(bw_record.p_signal[:, channel], bw_record.fs)
                   for channel in range(2)], axis=1)
    midpoint = len(bw) // 2
    if PROTOCOL == 1:
        train_noise, train_channel, train_offset = bw[:midpoint, 0], 1, 0
        test_noise, test_channel, test_offset = bw[midpoint:, 1], 2, midpoint
    else:
        train_noise, train_channel, train_offset = bw[:midpoint, 1], 2, 0
        test_noise, test_channel, test_offset = bw[midpoint:, 0], 1, midpoint

    def add_bw(clean, metadata, noise_source, channel, source_offset, split, seed_offset):
        patch_count = len(noise_source) // 3600
        if patch_count < 1:
            raise RuntimeError('BW source ngắn hơn 3600 samples')
        level_rng = np.random.RandomState(SEED + seed_offset)
        levels = level_rng.randint(20, 200, size=len(clean)).astype(np.float32) / 100.0
        noisy = np.empty_like(clean)
        metadata = metadata.copy()
        noise_starts = []
        for index in range(len(clean)):
            patch_index = index % patch_count
            start = patch_index * 3600
            patch = noise_source[start:start + 3600].astype(np.float32)
            clean_range = float(np.ptp(clean[index, 0]))
            noise_range = float(np.ptp(patch))
            if clean_range <= 1e-8 or noise_range <= 1e-8:
                raise ValueError(f'Zero range tại {split} row {index}')
            noisy[index, 0] = clean[index, 0] + levels[index] * clean_range / noise_range * patch
            noise_starts.append(source_offset + start)
        metadata['protocol'] = PROTOCOL
        metadata['noise_source_id'] = 'NSTDB-bw'
        metadata['noise_channel'] = channel
        metadata['noise_start'] = noise_starts
        metadata['r'] = levels
        metadata['sampling_rate'] = 360
        metadata['preprocessing_version'] = 'qtdb_first_lead_resample360_nonoverlap3600_v1'
        metadata['seed'] = SEED
        return noisy, metadata

    NOISY_TRAIN_GENERATED, TRAIN_META_GENERATED = add_bw(
        CLEAN_TRAIN_GENERATED, TRAIN_META_GENERATED, train_noise, train_channel,
        train_offset, 'train', 11)
    NOISY_VAL_GENERATED, VAL_META_GENERATED = add_bw(
        CLEAN_VAL_GENERATED, VAL_META_GENERATED, train_noise, train_channel,
        train_offset, 'validation', 22)
    NOISY_TEST_GENERATED, TEST_META_GENERATED = add_bw(
        CLEAN_TEST_GENERATED, TEST_META_GENERATED, test_noise, test_channel,
        test_offset, 'test', 33)

    generated_manifest = pd.concat(
        [TRAIN_META_GENERATED, VAL_META_GENERATED, TEST_META_GENERATED], ignore_index=True)
    if generated_manifest.sample_id.duplicated().any():
        raise RuntimeError('Generated sample_id bị trùng')
    generated_manifest.to_csv(COMMON_DATA_DIR / MANIFEST_NAME, index=False)
    np.savez_compressed(
        COMMON_DATA_DIR / TRAIN_BUNDLE_NAME,
        clean_train=CLEAN_TRAIN_GENERATED, noisy_train=NOISY_TRAIN_GENERATED,
        clean_val=CLEAN_VAL_GENERATED, noisy_val=NOISY_VAL_GENERATED,
        clean_test=CLEAN_TEST_GENERATED, noisy_test=NOISY_TEST_GENERATED,
        sample_id_test=TEST_META_GENERATED.sample_id.astype(str).to_numpy())
    split_info = {'protocol': PROTOCOL, 'seed': SEED,
                  'train_records': list(TRAIN_RECORDS),
                  'validation_records': list(VALIDATION_RECORDS),
                  'test_records': list(TEST_RECORDS)}
    (COMMON_DATA_DIR / 'record_split.json').write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Generated:', COMMON_DATA_DIR / TRAIN_BUNDLE_NAME)
    print(generated_manifest.groupby('split').size())
    del (CLEAN_TRAIN_GENERATED, NOISY_TRAIN_GENERATED,
         CLEAN_VAL_GENERATED, NOISY_VAL_GENERATED,
         CLEAN_TEST_GENERATED, NOISY_TEST_GENERATED)
    gc.collect()
    shutil.rmtree(RAW_DATA_DIR)
    print('Đã xóa raw download tạm; giữ common data cache trong Kaggle Output.')
else:
    print('Dùng common data cache từ Kaggle Input:', attached_train_bundles[0].parent)
    shutil.copy2(attached_train_bundles[0], COMMON_DATA_DIR / TRAIN_BUNDLE_NAME)
    shutil.copy2(attached_manifests[0], COMMON_DATA_DIR / MANIFEST_NAME)
    attached_test_bundles = sorted(INPUT_ROOT.rglob(TEST_BUNDLE_NAME))
    if len(attached_test_bundles) > 1:
        raise RuntimeError(f'Nhiều {TEST_BUNDLE_NAME}: {attached_test_bundles}')
    if attached_test_bundles:
        shutil.copy2(attached_test_bundles[0], COMMON_DATA_DIR / TEST_BUNDLE_NAME)
    split_files = sorted(INPUT_ROOT.rglob('record_split.json'))
    if len(split_files) == 1:
        shutil.copy2(split_files[0], COMMON_DATA_DIR / 'record_split.json')
    print('Đã copy cache vào output hiện tại để phiên resume kế tiếp tự đầy đủ.')
'''


cells = [
    md("""# Proposed Conditional DDPM — QTDB-BW-3600

Notebook Kaggle này triển khai đúng contract trong `01_proposed_conditional_ddpm.md`:

- dùng **nguyên byte** `HNFBlock`, `FiLM/BridgeBlock`, self-attention, `DDPM`, STFT loss và diffusion process từ `src/` hiện tại, đã nhúng vào notebook cùng SHA-256;
- driver bên ngoài chỉ nạp common tensors/manifest, chuẩn hóa, chia batch, checkpoint/resume và export;
- input/output của mô hình là `[N,1,3600]`; U-Net nhận `x_t` và noisy condition nên `in_channels=2`, `out_channels=1`;
- chọn checkpoint chỉ bằng validation loss, lưu EMA, không mở test để chọn model;
- output chính là `proposed_seed2024.npz`; notebook **không tính metric**.

> Đây là notebook huấn luyện dài hạn. Mặc định mỗi Kaggle version chạy tối đa 10 epoch rồi dừng sạch để lưu output. Gắn output phiên trước làm Input và Run All để resume cho tới epoch 400."""),

    md("""## 1. Input contract

Gắn **một Kaggle Dataset dữ liệu chung** dùng cho cả 5 mô hình. Notebook này không tự tải QTDB/NSTDB, không tự chia tập và không tự sinh nhiễu; nhờ đó Conditional DDPM học và dự đoán đúng cùng các mẫu với bốn mô hình còn lại. Không cần upload `src/` vì source snapshot đã được nhúng nguyên byte.

Dataset chung có thể chỉ chứa **một file NPZ**. Tên mặc định là:

Dataset dữ liệu chứa:

`common_qtdb_bw_3600.npz` với bốn key bắt buộc:

```text
clean_train, noisy_train, clean_val, noisy_val: [N,1,3600]
```

Để export prediction, bundle trên hoặc `common_qtdb_bw_3600_test.npz` cần thêm:

```text
noisy_test:     [N,1,3600]
sample_id_test: [N]
clean_test:     [N,1,3600]  # tùy chọn; notebook không dùng để chọn model/tính metric
```

`common_manifest.csv` là tùy chọn nếu NPZ đã có `sample_id_test`. Nếu có manifest, nó cần tối thiểu `sample_id`, `split`; thứ tự các dòng trong mỗi split phải trùng thứ tự tensor. Nếu có `array_row`, notebook sẽ sắp theo cột đó. Các cột provenance như `patient_id`, `record_id`, `protocol`, `noise_source_id`, `seed` được giữ nguyên để kiểm toán."""),

    md("## 2. Cấu hình phiên Kaggle"),
    code(r'''from pathlib import Path
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

# ---- Chỉ sửa khối cấu hình này giữa các phiên ----
MODE = 'train'                 # 'train' hoặc 'export'
TARGET_EPOCHS = 400
MAX_EPOCHS_THIS_SESSION = 10   # None = cố chạy tới TARGET_EPOCHS trong một phiên
BATCH_SIZE = 4                 # batch thực tế; giảm còn 2/1 nếu CUDA OOM
NUM_WORKERS = 2
USE_AMP = True
EMA_DECAY = 0.9                # giữ theo src/training/utils.py
SEED = 2024
EXPECTED_PROTOCOL = None       # None = nhận protocol đã khóa trong common manifest

TRAIN_BUNDLE_NAME = 'common_qtdb_bw_3600.npz'
TEST_BUNDLE_NAME = 'common_qtdb_bw_3600_test.npz'
MANIFEST_NAME = 'common_manifest.csv'
RUN_EXPORT_WHEN_COMPLETE = True
ALLOW_INCOMPLETE_EXPORT = False
INFERENCE_BATCH_SIZE = 4
DDIM_STEPS = 50
DDIM_ETA = 0.0
NUM_SHOTS = 1

INPUT_ROOT = Path('/kaggle/input')
WORK_ROOT = Path('/kaggle/working/proposed_conditional_ddpm_3600')
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
if EXPECTED_PROTOCOL not in {None, 1, 2}:
    raise ValueError(EXPECTED_PROTOCOL)
if TARGET_EPOCHS != 400:
    print('WARNING: proposal đăng ký 400 epoch; cấu hình hiện tại:', TARGET_EPOCHS)
if NUM_SHOTS != 1:
    raise ValueError('Output chính phải là 1-shot; multi-shot cần experiment/output khác.')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

MODEL_CONFIG = {
    'train': {'feats': 80, 'batch_size': BATCH_SIZE, 'epochs': TARGET_EPOCHS,
              'lr': 1e-4, 'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
    'experiment': {
        'name': 'proposed_conditional_ddpm_qtdb_bw_3600',
        'input_length': 3600, 'in_channels': 2, 'out_channels': 1,
        'normalization': 'per_sample_noisy_maxabs_shared_v1',
        'stft_n_fft': 128, 'stft_hop_length': 64,
        'ema_decay': EMA_DECAY, 'seed': SEED,
        'data_source': 'user_uploaded_common_5model_dataset',
    },
}

print('Device:', torch.cuda.get_device_name(0))
print('Mode:', MODE, '| target epochs:', TARGET_EPOCHS,
      '| max this session:', MAX_EPOCHS_THIS_SESSION)
print('Physical batch size:', BATCH_SIZE, '| AMP:', USE_AMP)
'''),

    md("## 3. Tái tạo source snapshot nhúng sẵn và nhập model gốc"),
    code(r'''REQUIRED_SOURCE_FILES = (
    'src/__init__.py',
    'src/models/__init__.py',
    'src/models/main_model.py',
    'src/models/unet_1d.py',
    'src/models/blocks/__init__.py',
    'src/models/blocks/hnf_block.py',
    'src/models/blocks/bridge_block.py',
    'src/models/blocks/attention_1d.py',
)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

working_src = SOURCE_SNAPSHOT / 'src'
if working_src.exists():
    shutil.rmtree(working_src)
EMBEDDED_SOURCE_B64 = json.loads(r"""__EMBEDDED_SOURCE_B64__""")
if set(EMBEDDED_SOURCE_B64) != set(REQUIRED_SOURCE_FILES):
    raise RuntimeError('Embedded source manifest mismatch')
for relative, payload in EMBEDDED_SOURCE_B64.items():
    destination = SOURCE_SNAPSHOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(payload))

source_hashes = {name: sha256_file(SOURCE_SNAPSHOT / name) for name in REQUIRED_SOURCE_FILES}
(REPORT_DIR / 'source_hashes.json').write_text(
    json.dumps(source_hashes, indent=2, ensure_ascii=False), encoding='utf-8')
sys.path.insert(0, str(SOURCE_SNAPSHOT))

from src.models.unet_1d import UNet1D
from src.models.main_model import DDPM, EMA, stft_loss

print('Imported byte-identical source embedded in this notebook')
display(pd.DataFrame([{'file': key, 'sha256': value} for key, value in source_hashes.items()]))
'''),

    md("## 4. Nạp common tensors và manifest từ Kaggle Input"),
    code(r'''def unique_input(filename, required=True):
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) > 1:
        raise RuntimeError(f'Nhiều file {filename}: {matches}')
    if required and not matches:
        raise FileNotFoundError(f'Không tìm thấy {filename} trong Kaggle Inputs')
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

train_bundle_path = unique_input(TRAIN_BUNDLE_NAME, required=False)
if train_bundle_path is None:
    viable_bundles = []
    for candidate in sorted(INPUT_ROOT.rglob('*.npz')):
        try:
            with np.load(candidate, allow_pickle=False) as probe_bundle:
                if {'clean_train', 'noisy_train', 'clean_val', 'noisy_val'}.issubset(probe_bundle.files):
                    viable_bundles.append(candidate)
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
    if len(viable_bundles) != 1:
        raise FileNotFoundError(
            f'Cần đúng một NPZ chung có clean/noisy train+val; tìm thấy {viable_bundles}')
    train_bundle_path = viable_bundles[0]
    print('Auto-detected common bundle:', train_bundle_path)
manifest_path = unique_input(MANIFEST_NAME, required=False)
# Common datasets often save string IDs with dtype=object. This input is
# user-selected/trusted; allow_pickle=True is needed only to decode those IDs.
train_bundle = np.load(train_bundle_path, allow_pickle=True)
required_keys = ('clean_train', 'noisy_train', 'clean_val', 'noisy_val')
missing = [key for key in required_keys if key not in train_bundle.files]
if missing:
    raise KeyError(f'{train_bundle_path} thiếu key {missing}; có {train_bundle.files}')

CLEAN_TRAIN = as_ncl3600(train_bundle['clean_train'], 'clean_train')
NOISY_TRAIN = as_ncl3600(train_bundle['noisy_train'], 'noisy_train')
CLEAN_VAL = as_ncl3600(train_bundle['clean_val'], 'clean_val')
NOISY_VAL = as_ncl3600(train_bundle['noisy_val'], 'noisy_val')
if CLEAN_TRAIN.shape != NOISY_TRAIN.shape or CLEAN_VAL.shape != NOISY_VAL.shape:
    raise ValueError('Clean/noisy shape mismatch')

if manifest_path is not None:
    manifest = pd.read_csv(manifest_path)
    manifest_origin = str(manifest_path)
else:
    if 'sample_id_test' not in train_bundle.files:
        raise FileNotFoundError(
            'Không có common_manifest.csv: NPZ phải chứa sample_id_test để benchmark giữ đúng ID.')
    synthetic_frames = []
    for split_name, count in (
        ('train', len(CLEAN_TRAIN)), ('validation', len(CLEAN_VAL)),
        ('test', len(train_bundle['sample_id_test'])),
    ):
        if split_name == 'test':
            ids = np.asarray(train_bundle['sample_id_test']).astype(str).reshape(-1)
        else:
            optional_key = f'sample_id_{"val" if split_name == "validation" else split_name}'
            ids = (np.asarray(train_bundle[optional_key]).astype(str).reshape(-1)
                   if optional_key in train_bundle.files
                   else np.asarray([f'{split_name}:{index:08d}' for index in range(count)]))
        if len(ids) != count:
            raise ValueError(f'{split_name}: sample_id={len(ids)}, tensor={count}')
        synthetic_frames.append(pd.DataFrame({
            'sample_id': ids, 'split': split_name, 'array_row': np.arange(count)}))
    manifest = pd.concat(synthetic_frames, ignore_index=True)
    manifest_origin = 'synthesized_from_npz_sample_ids'
    print('Không có CSV manifest: dựng manifest từ sample IDs trong common NPZ.')
required_manifest_columns = {'sample_id', 'split'}
if not required_manifest_columns.issubset(manifest.columns):
    raise KeyError(f'Manifest cần {required_manifest_columns}; có {list(manifest.columns)}')
manifest['split'] = manifest['split'].astype(str).str.lower().replace({'val': 'validation'})
if manifest['sample_id'].astype(str).duplicated().any():
    raise ValueError('sample_id phải duy nhất trong common manifest')
if EXPECTED_PROTOCOL is not None and 'protocol' in manifest.columns:
    if set(manifest.protocol.astype(int)) != {EXPECTED_PROTOCOL}:
        raise ValueError(f'Common data không khớp EXPECTED_PROTOCOL={EXPECTED_PROTOCOL}')

def manifest_split(name, expected_length):
    rows = manifest.loc[manifest.split == name].copy()
    if 'array_row' in rows.columns:
        row_values = rows.array_row.to_numpy()
        if np.array_equal(np.sort(row_values), np.arange(len(rows))):
            rows = rows.sort_values('array_row')
        else:
            raise ValueError(f'{name}: array_row phải là hoán vị 0..N-1')
    rows = rows.reset_index(drop=True)
    if len(rows) != expected_length:
        raise ValueError(f'{name}: manifest={len(rows)}, tensor={expected_length}')
    return rows

TRAIN_MANIFEST = manifest_split('train', len(CLEAN_TRAIN))
VAL_MANIFEST = manifest_split('validation', len(CLEAN_VAL))

def array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()

DATA_FINGERPRINTS = {
    'clean_train': array_sha256(CLEAN_TRAIN), 'noisy_train': array_sha256(NOISY_TRAIN),
    'clean_val': array_sha256(CLEAN_VAL), 'noisy_val': array_sha256(NOISY_VAL),
    'manifest': (sha256_file(manifest_path) if manifest_path is not None else
                 hashlib.sha256(manifest.to_csv(index=False).encode('utf-8')).hexdigest()),
}
manifest.to_csv(REPORT_DIR / 'resolved_common_manifest.csv', index=False)
data_audit = pd.DataFrame([
    {'split': 'train', 'N': len(CLEAN_TRAIN), 'shape': str(CLEAN_TRAIN.shape),
     'records': TRAIN_MANIFEST.record_id.nunique() if 'record_id' in TRAIN_MANIFEST else np.nan},
    {'split': 'validation', 'N': len(CLEAN_VAL), 'shape': str(CLEAN_VAL.shape),
     'records': VAL_MANIFEST.record_id.nunique() if 'record_id' in VAL_MANIFEST else np.nan},
])
data_audit.to_csv(REPORT_DIR / 'data_audit.csv', index=False)
display(data_audit)
'''),

    md("## 5. Chuẩn hóa chỉ từ noisy input"),
    code(r'''NORMALIZATION_EPS = 1e-8

def noisy_scale(noisy):
    scale = np.max(np.abs(noisy), axis=(1, 2), keepdims=True).astype(np.float32)
    if (scale <= NORMALIZATION_EPS).any():
        bad = np.flatnonzero(scale.reshape(-1) <= NORMALIZATION_EPS)[:10]
        raise ValueError(f'Noisy maxabs bằng 0 tại rows {bad.tolist()}')
    return scale

class CommonPairDataset(Dataset):
    def __init__(self, clean, noisy):
        self.clean = clean
        self.noisy = noisy
        self.scale = noisy_scale(noisy)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        scale = self.scale[index]
        clean = torch.from_numpy((self.clean[index] / scale).astype(np.float32))
        noisy = torch.from_numpy((self.noisy[index] / scale).astype(np.float32))
        return clean, noisy

train_dataset = CommonPairDataset(CLEAN_TRAIN, NOISY_TRAIN)
val_dataset = CommonPairDataset(CLEAN_VAL, NOISY_VAL)
normalization_audit = pd.DataFrame({
    'split': ['train', 'validation'],
    'scale_min': [train_dataset.scale.min(), val_dataset.scale.min()],
    'scale_median': [np.median(train_dataset.scale), np.median(val_dataset.scale)],
    'scale_max': [train_dataset.scale.max(), val_dataset.scale.max()],
})
normalization_audit.to_csv(REPORT_DIR / 'normalization_audit.csv', index=False)
display(normalization_audit)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train_dataset.scale.reshape(-1), bins=50, alpha=.65, label='train', color='#3568A8')
ax.hist(val_dataset.scale.reshape(-1), bins=50, alpha=.55, label='validation', color='#D39C2C')
ax.set(title='Noisy-derived max-abs scale', xlabel='Scale (đơn vị tín hiệu gốc)', ylabel='Số segment')
ax.legend(); ax.grid(axis='y', alpha=.2)
fig.tight_layout(); fig.savefig(REPORT_DIR / 'normalization_scale_distribution.png', dpi=160)
plt.show()
'''),

    md("## 6. Preflight: length, STFT và strict state dict"),
    code(r'''def build_model():
    base = UNet1D(in_channels=2, base_channels=MODEL_CONFIG['train']['feats'],
                  emb_dim=128, out_channels=1)
    return DDPM(base, MODEL_CONFIG, DEVICE).to(DEVICE)

model = build_model()
parameter_count = sum(parameter.numel() for parameter in model.parameters())
with torch.no_grad():
    probe = torch.zeros(1, 1, 3600, device=DEVICE)
    alpha = torch.full((1, 1), 0.8, device=DEVICE)
    direct_output = model.model(probe, probe, alpha)
    if direct_output.shape != probe.shape:
        raise RuntimeError(f'U-Net shape sai: {direct_output.shape}')
    spectral_probe = stft_loss(probe, probe, n_fft=128, hop_length=64)
    if not torch.isfinite(spectral_probe):
        raise FloatingPointError('STFT preflight non-finite')

strict_probe = build_model()
strict_probe.load_state_dict(model.state_dict(), strict=True)
del strict_probe, probe, direct_output
torch.cuda.empty_cache()
print(f'Preflight OK: [1,1,3600] -> [1,1,3600]; params={parameter_count:,}; '
      '3 downsamplings: 3600 -> 1800 -> 900 -> 450; STFT(128,64) OK; strict=True OK')
'''),

    md("## 7. DataLoaders, EMA và checkpoint/resume"),
    code(r'''LAST_NAME = 'last_proposed_conditional_ddpm_3600.pth'
BEST_NAME = 'best_proposed_conditional_ddpm_3600.pth'
LOG_NAME = 'training_proposed_conditional_ddpm_3600.csv'

def make_loader(dataset, shuffle, epoch=0):
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      drop_last=False, num_workers=NUM_WORKERS,
                      pin_memory=True, persistent_workers=NUM_WORKERS > 0,
                      generator=generator)

def cpu_state_dict(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}

def ema_model_state(model, ema):
    state = cpu_state_dict(model.state_dict())
    for key, value in ema.state_dict().items():
        state[key] = value.detach().cpu().clone()
    return state

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def checkpoint_candidates(filename):
    local = CHECKPOINT_DIR / filename
    candidates = ([local] if local.is_file() else []) + sorted(INPUT_ROOT.rglob(filename))
    # The local copy and its source input may coexist only after an in-notebook save.
    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique

def resolve_resume_checkpoint():
    candidates = checkpoint_candidates(LAST_NAME)
    if not candidates:
        return None
    if len(candidates) > 1 and (CHECKPOINT_DIR / LAST_NAME) not in candidates:
        raise RuntimeError(f'Nhiều resume checkpoint: {candidates}')
    return CHECKPOINT_DIR / LAST_NAME if (CHECKPOINT_DIR / LAST_NAME).is_file() else candidates[0]

def resolve_named_checkpoint(filename):
    local = CHECKPOINT_DIR / filename
    if local.is_file():
        return local
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f'Cần đúng một {filename}: {matches}')
    return matches[0]

def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)

optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG['train']['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=1.0)
ema = EMA(EMA_DECAY)
ema.register(model)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
start_epoch, best_validation_loss = 0, float('inf')
resume_path = resolve_resume_checkpoint()

if resume_path is not None:
    payload = torch.load(resume_path, map_location='cpu', weights_only=False)
    if payload['config'] != MODEL_CONFIG:
        raise RuntimeError('Resume config khác phiên hiện tại')
    if payload['data_fingerprints'] != DATA_FINGERPRINTS:
        raise RuntimeError('Resume data fingerprint mismatch')
    if payload['source_hashes'] != source_hashes:
        raise RuntimeError('Resume source hash mismatch')
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict']); optimizer_to(optimizer, DEVICE)
    scheduler.load_state_dict(payload['scheduler_state_dict'])
    scaler.load_state_dict(payload['scaler_state_dict'])
    ema.load_state_dict({key: value.to(DEVICE) for key, value in payload['ema_state_dict'].items()})
    torch.set_rng_state(payload['torch_rng_state'])
    torch.cuda.set_rng_state(payload['cuda_rng_state'])
    start_epoch = int(payload['epoch']) + 1
    best_validation_loss = float(payload['best_validation_loss'])
    previous_log = resume_path.parent / LOG_NAME
    if not previous_log.is_file():
        log_matches = sorted(INPUT_ROOT.rglob(LOG_NAME))
        if len(log_matches) > 1:
            raise RuntimeError(f'Nhiều training log: {log_matches}')
        previous_log = log_matches[0] if log_matches else previous_log
    if previous_log.is_file():
        shutil.copy2(previous_log, REPORT_DIR / LOG_NAME)
    previous_best = resume_path.parent / BEST_NAME
    if not previous_best.is_file():
        best_matches = sorted(INPUT_ROOT.rglob(BEST_NAME))
        if len(best_matches) > 1:
            raise RuntimeError(f'Nhiều best checkpoint: {best_matches}')
        previous_best = best_matches[0] if best_matches else previous_best
    if previous_best.is_file():
        shutil.copy2(previous_best, CHECKPOINT_DIR / BEST_NAME)
    print(f'Resume từ {resume_path}: epoch {start_epoch}/{TARGET_EPOCHS}')
else:
    print('Khởi tạo training run mới')

def save_checkpoint(epoch, is_best=False):
    payload = {
        'state_dict': cpu_state_dict(model.state_dict()),
        'ema_state_dict': cpu_state_dict(ema.state_dict()),
        'ema_model_state_dict': ema_model_state(model, ema),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch, 'best_validation_loss': best_validation_loss,
        'config': MODEL_CONFIG, 'data_fingerprints': DATA_FINGERPRINTS,
        'source_hashes': source_hashes, 'parameter_count': parameter_count,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
    }
    atomic_torch_save(payload, CHECKPOINT_DIR / (BEST_NAME if is_best else LAST_NAME))
'''),

    md("## 8. Huấn luyện theo phiên và chọn best chỉ bằng validation"),
    code(r'''def validation_loss_fixed(model, epoch):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    torch.manual_seed(SEED + 1_000_000)
    torch.cuda.manual_seed(SEED + 1_000_000)
    model.eval()
    losses = []
    loader = make_loader(val_dataset, shuffle=False, epoch=0)
    with torch.no_grad():
        for clean, noisy in tqdm(loader, desc=f'Validation epoch {epoch + 1}', leave=False):
            clean = clean.to(DEVICE, non_blocking=True)
            noisy = noisy.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = model(clean, noisy)
            if not torch.isfinite(loss):
                raise FloatingPointError('Validation loss NaN/Inf')
            losses.append(float(loss))
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
    return float(np.mean(losses))

def append_log(row):
    path = REPORT_DIR / LOG_NAME
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)

session_end = TARGET_EPOCHS if MAX_EPOCHS_THIS_SESSION is None else min(
    TARGET_EPOCHS, start_epoch + MAX_EPOCHS_THIS_SESSION)

if MODE == 'train':
    if start_epoch >= TARGET_EPOCHS:
        print('Training đã đủ target; không chạy thêm epoch.')
    for epoch in range(start_epoch, session_end):
        model.train()
        train_losses = []
        loader = make_loader(train_dataset, shuffle=True, epoch=epoch)
        progress = tqdm(loader, desc=f'Epoch {epoch + 1}/{TARGET_EPOCHS}')
        for clean, noisy in progress:
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
        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val_loss = validation_loss_fixed(model, epoch)
        improved = val_loss < best_validation_loss
        if improved:
            best_validation_loss = val_loss
        append_log({'epoch': epoch + 1, 'train_loss': train_loss,
                    'validation_loss': val_loss, 'best_validation_loss': best_validation_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'physical_batch_size': BATCH_SIZE, 'amp': USE_AMP})
        save_checkpoint(epoch, is_best=False)
        if improved:
            save_checkpoint(epoch, is_best=True)
        print(f'Epoch {epoch + 1}: train={train_loss:.6f}; val={val_loss:.6f}; '
              f'best={best_validation_loss:.6f}' + (' [BEST]' if improved else ''))
        gc.collect(); torch.cuda.empty_cache()

last_candidates = checkpoint_candidates(LAST_NAME)
completed_epochs = start_epoch if not last_candidates else int(
    torch.load(last_candidates[0], map_location='cpu', weights_only=False)['epoch']) + 1
TRAINING_COMPLETE = completed_epochs >= TARGET_EPOCHS
print(f'Đã hoàn thành {completed_epochs}/{TARGET_EPOCHS} epoch. '
      + ('Có thể export.' if TRAINING_COMPLETE else 'Save Version, gắn output làm Input và chạy lại để resume.'))
'''),

    md("## 9. Training curves (không phải test metric)"),
    code(r'''log_path = REPORT_DIR / LOG_NAME
if log_path.is_file():
    training_log = pd.read_csv(log_path).drop_duplicates('epoch', keep='last').sort_values('epoch')
    training_log.to_csv(log_path, index=False)
    display(training_log.tail(10))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(training_log.epoch, training_log.train_loss, label='Train loss', color='#3568A8')
    ax.plot(training_log.epoch, training_log.validation_loss, label='Validation loss', color='#D39C2C')
    ax.plot(training_log.epoch, training_log.best_validation_loss, label='Best validation',
            color='#333333', linestyle='--')
    ax.set(title='Conditional DDPM training history — QTDB-BW-3600',
           xlabel='Epoch', ylabel='Source DDPM loss')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(REPORT_DIR / 'training_history.png', dpi=180, bbox_inches='tight')
    plt.show()
'''),

    md("## 10. Nạp test chỉ sau khi khóa checkpoint"),
    code(r'''def load_test_contract():
    separate_path = unique_input(TEST_BUNDLE_NAME, required=False)
    # Accept object-dtype sample IDs from the trusted common Kaggle Dataset.
    bundle = np.load(separate_path, allow_pickle=True) if separate_path else train_bundle
    if 'noisy_test' not in bundle.files:
        raise KeyError(f'Cần noisy_test trong {TEST_BUNDLE_NAME} hoặc {TRAIN_BUNDLE_NAME}')
    noisy = as_ncl3600(bundle['noisy_test'], 'noisy_test')
    test_manifest = manifest_split('test', len(noisy))
    ids = test_manifest.sample_id.astype(str).to_numpy()
    if 'sample_id_test' in bundle.files:
        bundle_ids = np.asarray(bundle['sample_id_test']).astype(str).reshape(-1)
        if not np.array_equal(ids, bundle_ids):
            raise RuntimeError('sample_id_test không trùng thứ tự test trong manifest')
    return noisy, ids, test_manifest

SHOULD_EXPORT = MODE == 'export' or (MODE == 'train' and RUN_EXPORT_WHEN_COMPLETE and TRAINING_COMPLETE)
if SHOULD_EXPORT and not TRAINING_COMPLETE and not ALLOW_INCOMPLETE_EXPORT:
    raise RuntimeError(f'Checkpoint mới có {completed_epochs}/{TARGET_EPOCHS} epoch; không export test.')
print('Export requested:', SHOULD_EXPORT)
'''),

    md("## 11. DDIM 1-shot và export `proposed_seed2024.npz`"),
    code(r'''if SHOULD_EXPORT:
    best_path = resolve_named_checkpoint(BEST_NAME)
    best_payload = torch.load(best_path, map_location='cpu', weights_only=False)
    if best_payload['config'] != MODEL_CONFIG:
        raise RuntimeError('Best checkpoint config mismatch')
    if best_payload['data_fingerprints'] != DATA_FINGERPRINTS:
        raise RuntimeError('Best checkpoint data mismatch')
    if best_payload['source_hashes'] != source_hashes:
        raise RuntimeError('Best checkpoint source mismatch')

    inference_model = build_model()
    inference_model.load_state_dict(best_payload['ema_model_state_dict'], strict=True)
    inference_model.eval()
    NOISY_TEST, SAMPLE_ID_TEST, TEST_MANIFEST = load_test_contract()
    TEST_MANIFEST.to_csv(REPORT_DIR / 'test_manifest_locked.csv', index=False)
    test_scale = noisy_scale(NOISY_TEST)
    predictions, latency = [], []
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        for start in tqdm(range(0, len(NOISY_TEST), INFERENCE_BATCH_SIZE), desc='DDIM 1-shot'):
            stop = min(start + INFERENCE_BATCH_SIZE, len(NOISY_TEST))
            condition = torch.from_numpy(
                (NOISY_TEST[start:stop] / test_scale[start:stop]).astype(np.float32)
            ).to(DEVICE)
            torch.cuda.synchronize(); began = time.perf_counter()
            prediction_norm = inference_model.denoising(
                condition, use_ddim=True, ddim_steps=DDIM_STEPS,
                ddim_eta=DDIM_ETA, num_shots=NUM_SHOTS)
            torch.cuda.synchronize(); elapsed_ms = (time.perf_counter() - began) * 1000
            prediction = prediction_norm.cpu().numpy() * test_scale[start:stop]
            predictions.append(prediction.astype(np.float32))
            latency.extend([elapsed_ms / (stop - start)] * (stop - start))
    prediction = np.concatenate(predictions, axis=0)
    latency_ms = np.asarray(latency, dtype=np.float32)
    if prediction.shape != NOISY_TEST.shape or prediction.shape[1:] != (1, 3600):
        raise RuntimeError(f'Prediction shape sai: {prediction.shape}')
    if not np.isfinite(prediction).all() or not np.isfinite(latency_ms).all():
        raise FloatingPointError('Prediction/latency có NaN/Inf')
    if len(np.unique(SAMPLE_ID_TEST)) != len(SAMPLE_ID_TEST):
        raise ValueError('Test sample_id không duy nhất')

    prediction_path = WORK_ROOT / 'proposed_seed2024.npz'
    np.savez_compressed(prediction_path, sample_id=SAMPLE_ID_TEST,
                        prediction=prediction, latency_ms=latency_ms,
                        normalization_scale=test_scale.reshape(-1))
    export_audit = {
        'prediction_file': str(prediction_path), 'N': len(prediction),
        'prediction_shape': list(prediction.shape), 'sample_id_order_preserved': True,
        'finite': True, 'checkpoint': str(best_path),
        'checkpoint_sha256': sha256_file(best_path),
        'selected_epoch': int(best_payload['epoch']) + 1,
        'validation_loss': float(best_payload['best_validation_loss']),
        'sampler': 'DDIM', 'ddim_steps': DDIM_STEPS, 'ddim_eta': DDIM_ETA,
        'num_shots': NUM_SHOTS, 'seed': SEED,
        'latency_ms_mean': float(latency_ms.mean()),
    }
    (REPORT_DIR / 'export_audit.json').write_text(
        json.dumps(export_audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(export_audit, indent=2, ensure_ascii=False))
'''),

    md("## 12. Metadata và gói output"),
    code(r'''run_info = {
    'experiment': 'proposed_conditional_ddpm_qtdb_bw_3600',
    'mode': MODE, 'target_epochs': TARGET_EPOCHS,
    'completed_epochs': completed_epochs, 'training_complete': TRAINING_COMPLETE,
    'config': MODEL_CONFIG, 'device': str(DEVICE),
    'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
    'python_version': platform.python_version(), 'parameter_count': parameter_count,
    'data_fingerprints': DATA_FINGERPRINTS, 'source_hashes': source_hashes,
    'common_bundle': str(train_bundle_path), 'manifest_origin': manifest_origin,
    'normalization_scale_source': 'noisy input only',
    'test_used_for_checkpoint_selection': False,
    'metrics_computed_in_model_notebook': False,
}
(REPORT_DIR / 'run_info.json').write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False), encoding='utf-8')

archive = Path('/kaggle/working/proposed_conditional_ddpm_3600_outputs.zip')
if archive.exists():
    archive.unlink()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output_zip:
    for path in sorted(WORK_ROOT.rglob('*')):
        if path.is_file():
            output_zip.write(path, path.relative_to(WORK_ROOT))
print('Output directory:', WORK_ROOT)
print('Archive:', archive, f'({archive.stat().st_size / 1024**2:.1f} MiB)')
'''),

    md("""## Kết quả và cách tiếp tục

- Nếu log ghi `<400/400 epoch`: bấm **Save Version**, sau đó gắn output vừa tạo làm Input của phiên mới và chạy lại với `MODE='train'`. Notebook tự tìm `last_proposed_conditional_ddpm_3600.pth` và resume.
- Khi đủ 400 epoch, notebook khóa checkpoint có validation loss thấp nhất, nạp EMA bằng `strict=True`, rồi xuất `proposed_seed2024.npz`.
- Có thể tách inference sang phiên khác: gắn output chứa checkpoint, đặt `MODE='export'`; common data phải giống hệt và notebook tự tái tạo đúng source snapshot đã nhúng.
- `proposed_seed2024.npz` chứa `sample_id`, `prediction`, `latency_ms` và `normalization_scale`. Prediction đã inverse-normalize về miền tín hiệu gốc.
- Notebook này không tạo SSD/PRD/SNR hay dùng 14 test records để chọn checkpoint. Hãy đưa `.npz` sang notebook benchmark chung để tính lại toàn bộ metric."""),
]

# Insert the current repository files as byte-for-byte base64 payloads. This is
# generated mechanically: model/loss/diffusion code is not reimplemented.
embedded_json = json.dumps(EMBEDDED_SOURCE_B64, sort_keys=True, separators=(",", ":"))
for cell in cells:
    if cell.get("cell_type") != "code":
        continue
    cell_text = "".join(cell["source"])
    if "__EMBEDDED_SOURCE_B64__" in cell_text:
        cell["source"] = cell_text.replace("__EMBEDDED_SOURCE_B64__", embedded_json).splitlines(True)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET}")
