"""Build parameterized Kaggle-training and Colab-evaluation notebooks for QTDB.

The generated notebooks extend the validated QTDB/BW benchmark to the three
NSTDB records BW, EM and MA while keeping the two Paper-A noise protocols.
Model and DDPM definitions are copied from the existing benchmark notebook so
there is only one architecture/loss source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle_qtdb_rule_benchmark.ipynb"
KAGGLE_OUT = ROOT / "notebooks" / "kaggle_train_qtdb_bw_em_ma.ipynb"
COLAB_OUT = ROOT / "notebooks" / "colab_evaluate_qtdb_bw_em_ma.ipynb"


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": value.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


source = json.loads(SOURCE.read_text(encoding="utf-8"))


def body(index: int) -> str:
    return "".join(source["cells"][index]["source"])


ARCHITECTURE = body(5)
DDPM = body(7)
CLEAN_FUNCTIONS = body(9)


COMMON_INSTALL = """%pip -q install 'wfdb>=4.1,<5' 'scipy>=1.10' 'pandas>=2.0' 'matplotlib>=3.7' 'pyyaml>=6.0'
"""


TRAIN_CONFIG = r'''import gc
import hashlib
import json
import math
import platform
import random
import shutil
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import wfdb

SEED = 1234
ARTIFACTS_TO_TRAIN = ('BW', 'EM', 'MA')
PROTOCOLS_TO_TRAIN = (1, 2)
TARGET_EPOCHS = 100
# Safe default for one Kaggle version. Increase after measuring seconds/epoch.
EPOCHS_PER_EXPERIMENT_THIS_SESSION = 5

TEST_RECORDS = (
    'sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853',
    'sel16420', 'sel16795', 'sele0106', 'sele0121', 'sel32',
    'sel49', 'sel14046', 'sel15814',
)
TARGET_FS = 360
LENGTH = 512
MAX_UNPADDED = 496
INSERT_OFFSET = 16
PREPROCESSING_VERSION = 'qtdb_pu1_p_and_beat_40ms_reflect100ms_360hz_endpoints_pad16_v2'
MODEL_NORMALIZATION = 'per_segment_clean_maxabs_shared_with_noisy_v1'
LOSS_VERSION = 'l1_mean_relative_stft_snr_weighted_continuous_alpha_v1'
CHECKPOINT_TAG = 'norm_loss_v2'
PHYSIONET_QT = 'qtdb/1.0.0'
PHYSIONET_NST = 'nstdb/1.0.0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Enable a GPU accelerator in Kaggle before training.')

OUT = Path('/kaggle/working/qtdb_bw_em_ma_training')
TRAIN_DIR = OUT / 'checkpoints'
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# None means the official full run. Use a small integer only for a smoke test.
TRAIN_RECORD_LIMIT = None
MAX_TRAIN_SEGMENTS = None

CONFIG = {
    'train': {'feats': 80, 'batch_size': 96, 'epochs': TARGET_EPOCHS, 'lr': 1e-4,
              'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
    'model_normalization': MODEL_NORMALIZATION,
    'loss_version': LOSS_VERSION,
    'preprocessing_version': PREPROCESSING_VERSION,
}

VALID_ARTIFACTS = {'BW': 'bw', 'EM': 'em', 'MA': 'ma'}
if not ARTIFACTS_TO_TRAIN or not set(ARTIFACTS_TO_TRAIN).issubset(VALID_ARTIFACTS):
    raise ValueError(f'ARTIFACTS_TO_TRAIN must be selected from {tuple(VALID_ARTIFACTS)}')
if not PROTOCOLS_TO_TRAIN or not set(PROTOCOLS_TO_TRAIN).issubset({1, 2}):
    raise ValueError('PROTOCOLS_TO_TRAIN must contain protocol 1 and/or 2')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
(OUT / 'config.json').write_text(json.dumps({
    'artifacts_to_train': ARTIFACTS_TO_TRAIN,
    'protocols_to_train': PROTOCOLS_TO_TRAIN,
    'epochs_per_experiment_this_session': EPOCHS_PER_EXPERIMENT_THIS_SESSION,
    **CONFIG,
}, indent=2), encoding='utf-8')
print('GPU:', torch.cuda.get_device_name(0), '| output:', OUT)
'''


CACHE_HELPERS = r'''def cached_input(filename):
    local = OUT / filename
    if local.is_file():
        return local
    root = Path('/kaggle/input')
    matches = sorted(root.rglob(filename)) if root.exists() else []
    if len(matches) > 1:
        raise RuntimeError(f'Multiple files named {filename}; attach only one prior output')
    if matches:
        shutil.copy2(matches[0], local)
        return local
    return None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
'''


TRAIN_DATA = r'''clean_cache = cached_input('train_clean_full.npy') if TRAIN_RECORD_LIMIT is None else None
meta_cache = cached_input('train_metadata_full.csv') if TRAIN_RECORD_LIMIT is None else None
audit_cache = cached_input('train_record_audit.csv') if TRAIN_RECORD_LIMIT is None else None
if any(path is not None for path in (clean_cache, meta_cache, audit_cache)) and not all(
        path is not None for path in (clean_cache, meta_cache, audit_cache)):
    raise RuntimeError('Incomplete QTDB cache: clean, metadata and audit must travel together')

if clean_cache is not None:
    TRAIN_CLEAN = np.load(clean_cache).astype(np.float32, copy=False)
    TRAIN_META = pd.read_csv(meta_cache)
    TRAIN_AUDIT = pd.read_csv(audit_cache)
    TRAIN_RECORDS = tuple(TRAIN_AUDIT.record_id)
else:
    records = wfdb.get_record_list('qtdb')
    TRAIN_RECORDS = tuple(sorted(set(records) - set(TEST_RECORDS)))
    if TRAIN_RECORD_LIMIT is not None:
        TRAIN_RECORDS = TRAIN_RECORDS[:TRAIN_RECORD_LIMIT]
    beats, rows, audits = [], [], []
    for record_name in tqdm(TRAIN_RECORDS, desc='QTDB train records'):
        record_beats, record_rows, audit = clean_segments_for_record(record_name)
        beats.extend(record_beats)
        rows.extend(record_rows)
        audits.append(audit)
    if not beats:
        raise RuntimeError('No QTDB training segments were produced')
    TRAIN_CLEAN = np.stack(beats).astype(np.float32)
    TRAIN_META = pd.DataFrame(rows)
    TRAIN_AUDIT = pd.DataFrame(audits)
    if MAX_TRAIN_SEGMENTS is not None:
        TRAIN_CLEAN = TRAIN_CLEAN[:MAX_TRAIN_SEGMENTS]
        TRAIN_META = TRAIN_META.iloc[:MAX_TRAIN_SEGMENTS].reset_index(drop=True)
    if TRAIN_RECORD_LIMIT is None and MAX_TRAIN_SEGMENTS is None:
        np.save(OUT / 'train_clean_full.npy', TRAIN_CLEAN)
        TRAIN_META.to_csv(OUT / 'train_metadata_full.csv', index=False)
        TRAIN_AUDIT.to_csv(OUT / 'train_record_audit.csv', index=False)

assert TRAIN_CLEAN.ndim == 2 and TRAIN_CLEAN.shape[1] == LENGTH
assert len(TRAIN_CLEAN) == len(TRAIN_META) and TRAIN_META.sample_id.is_unique
assert np.isfinite(TRAIN_CLEAN).all()
assert not set(TRAIN_META.record_id) & set(TEST_RECORDS)

# Preserve the existing benchmark contract: deterministic 70/30 segment split.
indices = np.random.RandomState(SEED).permutation(len(TRAIN_CLEAN))
cut = int(0.70 * len(indices))
TRAIN_INDICES, VAL_INDICES = indices[:cut], indices[cut:]
split = np.where(np.isin(np.arange(len(TRAIN_CLEAN)), TRAIN_INDICES), 'train', 'validation')
split_frame = TRAIN_META[['sample_id', 'record_id']].copy()
split_frame['split'] = split
split_frame.to_csv(OUT / 'train_validation_split.csv', index=False)
print('QTDB records:', len(TRAIN_RECORDS), '| segments:', len(TRAIN_CLEAN),
      '| train:', len(TRAIN_INDICES), '| validation:', len(VAL_INDICES))
'''


LOAD_NOISE = r'''NOISE = {}
NOISE_MIDPOINT = {}
for artifact, record_name in VALID_ARTIFACTS.items():
    filename = f'{record_name}_360.npy'
    cache = cached_input(filename)
    if cache is not None:
        values = np.load(cache).astype(np.float32, copy=False)
    else:
        record = wfdb.rdrecord(record_name, pn_dir=PHYSIONET_NST)
        values = record.p_signal.astype(np.float32)
        if int(record.fs) != TARGET_FS:
            new_length = int(round(len(values) * TARGET_FS / float(record.fs)))
            values = signal.resample(values, new_length, axis=0).astype(np.float32)
        np.save(OUT / filename, values)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise RuntimeError(f'Invalid NSTDB {artifact} record: {values.shape}')
    NOISE[artifact] = values
    NOISE_MIDPOINT[artifact] = len(values) // 2
    print(artifact, values.shape, 'midpoint=', NOISE_MIDPOINT[artifact])
'''


DATASET = r'''class FixedArtifactDataset(Dataset):
    def __init__(self, clean_array, indices, artifact, protocol):
        self.clean = clean_array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.artifact = artifact
        self.protocol = int(protocol)
        values = NOISE[artifact]
        midpoint = NOISE_MIDPOINT[artifact]
        train_channel = 0 if protocol == 1 else 1
        self.noise = values[:midpoint, train_channel]
        self.patch_count = len(self.noise) // LENGTH
        if self.patch_count == 0:
            raise RuntimeError(f'{artifact} training half is shorter than one patch')
        # Keep exactly the same r value for the same QTDB segment across all
        # artifacts and both protocols so paired comparisons are meaningful.
        rng = np.random.RandomState(SEED)
        self.levels = rng.randint(20, 200, size=len(clean_array)).astype(np.float32) / 100.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        source_index = int(self.indices[item])
        clean_raw = self.clean[source_index]
        patch_index = source_index % self.patch_count
        start = patch_index * LENGTH
        patch = self.noise[start:start + LENGTH]
        clean_range = float(np.ptp(clean_raw))
        noise_range = float(np.ptp(patch))
        if clean_range <= 0 or noise_range <= 0:
            raise ValueError(f'Zero range at QTDB sample {source_index}')
        noisy_raw = clean_raw + float(self.levels[source_index]) * clean_range / noise_range * patch
        scale = max(float(np.max(np.abs(clean_raw))), 1e-8)
        clean = torch.from_numpy((clean_raw / scale).astype(np.float32, copy=False))[None, :]
        noisy = torch.from_numpy((noisy_raw / scale).astype(np.float32, copy=False))[None, :]
        return clean, noisy


def make_loader(artifact, protocol, indices, shuffle, drop_last, epoch=0):
    dataset = FixedArtifactDataset(TRAIN_CLEAN, indices, artifact, protocol)
    artifact_seed = {'BW': 11, 'EM': 22, 'MA': 33}[artifact]
    generator = torch.Generator().manual_seed(SEED + artifact_seed * 100_000 + protocol * 1_000 + epoch)
    return DataLoader(dataset, batch_size=CONFIG['train']['batch_size'], shuffle=shuffle,
                      drop_last=drop_last, generator=generator, num_workers=0,
                      pin_memory=True)

probe_clean, probe_noisy = next(iter(make_loader(
    ARTIFACTS_TO_TRAIN[0], PROTOCOLS_TO_TRAIN[0], TRAIN_INDICES[:128], False, False)))
assert probe_clean.shape[1:] == (1, LENGTH) and probe_noisy.shape == probe_clean.shape
assert torch.isfinite(probe_clean).all() and torch.isfinite(probe_noisy).all()
print('Loader probe:', tuple(probe_clean.shape), tuple(probe_noisy.shape))
'''


TRAIN_LOOP = r'''def experiment_stem(artifact, protocol):
    return f'{artifact.lower()}_protocol{protocol}_{CHECKPOINT_TAG}'


def checkpoint_candidates(artifact, protocol):
    filename = f'last_{experiment_stem(artifact, protocol)}.pth'
    candidates = [TRAIN_DIR / filename]
    root = Path('/kaggle/input')
    if root.exists():
        candidates.extend(root.rglob(filename))
    return [path for path in candidates if path.is_file()]


def copy_prior_sidecars(checkpoint_path, artifact, protocol):
    stem = experiment_stem(artifact, protocol)
    for filename in (f'best_{stem}.pth', f'training_{stem}.csv'):
        source = checkpoint_path.parent / filename
        target = TRAIN_DIR / filename
        if source.is_file() and source.resolve() != target.resolve():
            shutil.copy2(source, target)


def save_state(model, optimizer, scheduler, ema, artifact, protocol, epoch, best_val, filename):
    ema_model_state = {key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()}
    for key, value in ema.state_dict().items():
        ema_model_state[key] = value.detach().cpu().clone()
    payload = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'ema_model_state_dict': ema_model_state,
        'artifact': artifact,
        'protocol': protocol,
        'epoch': epoch,
        'best_validation_loss': best_val,
        'config': CONFIG,
        'train_records': TRAIN_RECORDS,
        'test_records': TEST_RECORDS,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
    }
    target = TRAIN_DIR / filename
    temporary = target.with_suffix(target.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(target)


def validation_loss(model, loader, artifact, protocol):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    artifact_seed = {'BW': 11, 'EM': 22, 'MA': 33}[artifact]
    torch.manual_seed(SEED + artifact_seed * 10_000 + protocol)
    torch.cuda.manual_seed_all(SEED + artifact_seed * 10_000 + protocol)
    model.eval()
    sums = {'total': 0.0, 'time': 0.0, 'frequency': 0.0,
            'frequency_unweighted': 0.0, 'frequency_weight': 0.0}
    examples = 0
    with torch.no_grad():
        for clean, noisy in loader:
            clean, noisy = clean.to(DEVICE), noisy.to(DEVICE)
            components = model.loss_components(clean, noisy)
            for name in sums:
                sums[name] += float(components[name]) * len(clean)
            examples += len(clean)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)
    return {name: value / examples for name, value in sums.items()}


def train_experiment(artifact, protocol):
    stem = experiment_stem(artifact, protocol)
    last_name, best_name = f'last_{stem}.pth', f'best_{stem}.pth'
    log_path = TRAIN_DIR / f'training_{stem}.csv'
    model = ResearchLossDDPM(UNet1D(2, 80, 128, 1), CONFIG, DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['train']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    ema = EMA(0.9)
    ema.register(model)
    val_loader = make_loader(artifact, protocol, VAL_INDICES, False, True)
    candidates = checkpoint_candidates(artifact, protocol)
    if candidates:
        payloads = [(torch.load(path, map_location='cpu', weights_only=False), path)
                    for path in candidates]
        payload, checkpoint_path = max(payloads, key=lambda pair: int(pair[0]['epoch']))
        if payload.get('artifact') != artifact or int(payload.get('protocol', -1)) != protocol:
            raise RuntimeError(f'Checkpoint identity mismatch: {checkpoint_path}')
        if payload.get('config') != CONFIG or tuple(payload.get('train_records', ())) != TRAIN_RECORDS:
            raise RuntimeError(f'Checkpoint configuration/data mismatch: {checkpoint_path}')
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        ema.load_state_dict({key: value.to(DEVICE)
                             for key, value in payload['ema_state_dict'].items()})
        if 'torch_rng_state' in payload:
            torch.set_rng_state(payload['torch_rng_state'])
        if payload.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(payload['cuda_rng_state'])
        start_epoch = int(payload['epoch']) + 1
        best_val = float(payload['best_validation_loss'])
        copy_prior_sidecars(checkpoint_path, artifact, protocol)
        print(f'{artifact}/P{protocol}: resume epoch {start_epoch + 1} from {checkpoint_path}')
    else:
        start_epoch, best_val = 0, float('inf')
        print(f'{artifact}/P{protocol}: new run')

    stop_epoch = min(TARGET_EPOCHS, start_epoch + EPOCHS_PER_EXPERIMENT_THIS_SESSION)
    for epoch in range(start_epoch, stop_epoch):
        tick = time.perf_counter()
        loader = make_loader(artifact, protocol, TRAIN_INDICES, True, True, epoch)
        model.train()
        sums = {'total': 0.0, 'time': 0.0, 'frequency': 0.0}
        examples = 0
        for clean, noisy in tqdm(loader, desc=f'{artifact}/P{protocol} epoch {epoch + 1}', leave=False):
            clean, noisy = clean.to(DEVICE), noisy.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            components = model.loss_components(clean, noisy)
            if not torch.isfinite(components['total']):
                raise FloatingPointError(f'Nonfinite loss: {artifact}/P{protocol}/epoch{epoch + 1}')
            components['total'].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f'Nonfinite gradient: {artifact}/P{protocol}/epoch{epoch + 1}')
            optimizer.step()
            ema.update(model)
            for name in sums:
                sums[name] += float(components[name].detach()) * len(clean)
            examples += len(clean)
        scheduler.step()
        train_values = {name: value / examples for name, value in sums.items()}
        val_values = validation_loss(model, val_loader, artifact, protocol)
        val = val_values['total']
        if not all(np.isfinite(value) for value in val_values.values()):
            raise FloatingPointError(f'Nonfinite validation: {artifact}/P{protocol}/epoch{epoch + 1}')
        improved = val < best_val
        if improved:
            best_val = val
        row = {
            'artifact': artifact, 'protocol': protocol, 'epoch': epoch + 1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'train_loss': train_values['total'],
            'train_time_loss': train_values['time'],
            'train_frequency_loss': train_values['frequency'],
            'validation_loss': val,
            'validation_time_loss': val_values['time'],
            'validation_frequency_loss': val_values['frequency'],
            'validation_frequency_unweighted': val_values['frequency_unweighted'],
            'validation_frequency_weight': val_values['frequency_weight'],
            'best_validation_loss': best_val,
            'seconds': time.perf_counter() - tick,
        }
        pd.DataFrame([row]).to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
        save_state(model, optimizer, scheduler, ema, artifact, protocol, epoch, best_val, last_name)
        if improved:
            save_state(model, optimizer, scheduler, ema, artifact, protocol, epoch, best_val, best_name)
        print(row)
        torch.cuda.empty_cache()
        gc.collect()
    return {'artifact': artifact, 'protocol': protocol,
            'completed_epochs': stop_epoch, 'target_epochs': TARGET_EPOCHS,
            'best_validation_loss': best_val}


RUN_STATUS = []
for artifact in ARTIFACTS_TO_TRAIN:
    for protocol in PROTOCOLS_TO_TRAIN:
        RUN_STATUS.append(train_experiment(artifact, protocol))
status = pd.DataFrame(RUN_STATUS)
status.to_csv(OUT / 'session_status.csv', index=False)
display(status)
'''


TRAIN_PACKAGE = r'''fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for log_path in sorted(TRAIN_DIR.glob('training_*.csv')):
    frame = pd.read_csv(log_path).drop_duplicates('epoch', keep='last').sort_values('epoch')
    label = f"{frame.artifact.iloc[0]}/P{int(frame.protocol.iloc[0])}"
    axes[0].plot(frame.epoch, frame.train_loss, label=f'{label} train')
    axes[0].plot(frame.epoch, frame.validation_loss, '--', label=f'{label} val')
    axes[1].plot(frame.epoch, frame.best_validation_loss, label=label)
axes[0].set(title='Training and validation loss', xlabel='Epoch', ylabel='Total loss')
axes[1].set(title='Best validation loss', xlabel='Epoch', ylabel='Loss')
for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / 'training_curves.png', dpi=160)
plt.show()

checkpoint_rows = []
for path in sorted(TRAIN_DIR.glob('*.pth')):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    checkpoint_rows.append({
        'file': path.name, 'artifact': payload.get('artifact'),
        'protocol': payload.get('protocol'), 'epoch': int(payload.get('epoch', -1)) + 1,
        'best_validation_loss': payload.get('best_validation_loss'),
        'sha256': sha256(path), 'size_mb': path.stat().st_size / 1024**2,
    })
checkpoint_manifest = pd.DataFrame(checkpoint_rows)
checkpoint_manifest.to_csv(OUT / 'checkpoint_manifest.csv', index=False)
display(checkpoint_manifest)

archive_path = Path('/kaggle/working/qtdb_bw_em_ma_training_outputs.zip')
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(OUT))
print('Output:', OUT, '| archive:', archive_path)
'''


EVAL_CONFIG = r'''from google.colab import drive
drive.mount('/content/drive')

import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import torch
from tqdm.auto import tqdm
import wfdb
from IPython.display import display

SEED = 1234
ARTIFACTS_TO_EVALUATE = ('BW', 'EM', 'MA')
PROTOCOLS = (1, 2)
# Full 1-shot evaluation is the practical default. Add 3,5,10 only after it passes.
K_VALUES = (1,)
BATCH_SIZE = 50
DDIM_STEPS = 50
DDIM_ETA = 0.0

CHECKPOINT_DIR = Path('/content/drive/MyDrive/qtdb_bw_em_ma_training/checkpoints')
OUT = Path('/content/drive/MyDrive/qtdb_bw_em_ma_evaluation')
OUT.mkdir(parents=True, exist_ok=True)

TEST_RECORDS = (
    'sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853',
    'sel16420', 'sel16795', 'sele0106', 'sele0121', 'sel32',
    'sel49', 'sel14046', 'sel15814',
)
TARGET_FS = 360
LENGTH = 512
MAX_UNPADDED = 496
INSERT_OFFSET = 16
PREPROCESSING_VERSION = 'qtdb_pu1_p_and_beat_40ms_reflect100ms_360hz_endpoints_pad16_v2'
MODEL_NORMALIZATION = 'per_segment_clean_maxabs_shared_with_noisy_v1'
LOSS_VERSION = 'l1_mean_relative_stft_snr_weighted_continuous_alpha_v1'
CHECKPOINT_TAG = 'norm_loss_v2'
PHYSIONET_QT = 'qtdb/1.0.0'
PHYSIONET_NST = 'nstdb/1.0.0'
VALID_ARTIFACTS = {'BW': 'bw', 'EM': 'em', 'MA': 'ma'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Enable a GPU runtime in Colab before evaluation.')

# Set small values for a smoke test; keep None for the official evaluation.
MAX_TEST_RECORDS = None
MAX_TEST_SEGMENTS = None

CONFIG = {
    'train': {'feats': 80, 'batch_size': 96, 'epochs': 100, 'lr': 1e-4,
              'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
    'model_normalization': MODEL_NORMALIZATION,
    'loss_version': LOSS_VERSION,
    'preprocessing_version': PREPROCESSING_VERSION,
}

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print('GPU:', torch.cuda.get_device_name(0), '| output:', OUT)
'''


LOAD_MODELS = r'''def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


MODELS, CHECKPOINT_INFO = {}, []
for artifact in ARTIFACTS_TO_EVALUATE:
    for protocol in PROTOCOLS:
        filename = f'best_{artifact.lower()}_protocol{protocol}_{CHECKPOINT_TAG}.pth'
        matches = sorted(CHECKPOINT_DIR.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(f'Need exactly one {filename}; found {matches}')
        path = matches[0]
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('artifact') != artifact or int(payload.get('protocol', -1)) != protocol:
            raise RuntimeError(f'Checkpoint identity mismatch: {path}')
        if payload.get('config') != CONFIG:
            raise RuntimeError(f'Checkpoint config mismatch: {path}')
        if 'ema_model_state_dict' not in payload:
            raise RuntimeError(f'EMA inference weights missing: {path}')
        model = ResearchLossDDPM(UNet1D(2, 80, 128, 1), CONFIG, DEVICE).to(DEVICE)
        model.load_state_dict(payload['ema_model_state_dict'], strict=True)
        model.eval()
        probe = torch.zeros(1, 1, LENGTH, device=DEVICE)
        with torch.inference_mode():
            output = model.model(probe, probe, torch.ones(1, 1, device=DEVICE))
        assert output.shape == probe.shape and torch.isfinite(output).all()
        MODELS[(artifact, protocol)] = model
        CHECKPOINT_INFO.append({
            'artifact': artifact, 'protocol': protocol, 'path': str(path),
            'sha256': sha256(path), 'selected_epoch': int(payload['epoch']) + 1,
            'best_validation_loss': float(payload['best_validation_loss']),
            'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        })
display(pd.DataFrame(CHECKPOINT_INFO))
'''


TEST_DATA = r'''selected_records = TEST_RECORDS[:MAX_TEST_RECORDS] if MAX_TEST_RECORDS else TEST_RECORDS
all_beats, rows, audits = [], [], []
for record_name in tqdm(selected_records, desc='QTDB test records'):
    record_beats, record_rows, audit = clean_segments_for_record(record_name)
    all_beats.extend(record_beats)
    rows.extend(record_rows)
    audits.append(audit)
    if MAX_TEST_SEGMENTS and len(all_beats) >= MAX_TEST_SEGMENTS:
        all_beats = all_beats[:MAX_TEST_SEGMENTS]
        rows = rows[:MAX_TEST_SEGMENTS]
        break
if not all_beats:
    raise RuntimeError('No QTDB test segments were produced')
CLEAN = np.stack(all_beats).astype(np.float32)
CLEAN_META = pd.DataFrame(rows)
pd.DataFrame(audits).to_csv(OUT / 'test_record_audit.csv', index=False)
np.save(OUT / 'clean_test.npy', CLEAN)
assert CLEAN.shape == (len(CLEAN_META), LENGTH) and CLEAN_META.sample_id.is_unique
print('Clean QTDB test:', CLEAN.shape)

NOISE, MIDPOINT = {}, {}
for artifact, record_name in VALID_ARTIFACTS.items():
    record = wfdb.rdrecord(record_name, pn_dir=PHYSIONET_NST)
    values = record.p_signal.astype(np.float32)
    if int(record.fs) != TARGET_FS:
        values = signal.resample(
            values, int(round(len(values) * TARGET_FS / float(record.fs))), axis=0
        ).astype(np.float32)
    assert values.ndim == 2 and values.shape[1] >= 2 and np.isfinite(values).all()
    NOISE[artifact] = values
    MIDPOINT[artifact] = len(values) // 2
'''


MAKE_TEST_PAIRS = r'''def make_test_pairs(artifact, protocol):
    channel = 1 if protocol == 1 else 0
    midpoint = MIDPOINT[artifact]
    noise = NOISE[artifact][midpoint:, channel]
    patch_count = len(noise) // LENGTH
    if patch_count == 0:
        raise RuntimeError(f'{artifact} test half is shorter than one patch')
    # The same r vector is reused for BW, EM, MA and both protocols.
    rng = np.random.RandomState(SEED)
    levels = rng.randint(20, 200, size=len(CLEAN)).astype(np.float32) / 100.0
    noisy = np.empty_like(CLEAN)
    rows = []
    for index, (clean, level) in enumerate(zip(CLEAN, levels)):
        relative_start = (index % patch_count) * LENGTH
        patch = noise[relative_start:relative_start + LENGTH]
        clean_range, noise_range = float(np.ptp(clean)), float(np.ptp(patch))
        if clean_range <= 0 or noise_range <= 0:
            raise ValueError(f'Zero range: {artifact}/P{protocol}/sample{index}')
        noisy[index] = clean + float(level) * clean_range / noise_range * patch
        row = CLEAN_META.iloc[index].to_dict()
        row.update({
            'artifact': artifact, 'protocol': protocol, 'split': 'test',
            'noise_source_id': f'nstdb/1.0.0/{VALID_ARTIFACTS[artifact]}:second_half',
            'noise_channel': channel + 1,
            'noise_start': midpoint + relative_start,
            'r': float(level), 'seed': SEED,
            'sampling_rate': TARGET_FS,
            'preprocessing_version': PREPROCESSING_VERSION,
        })
        rows.append(row)
    return noisy, pd.DataFrame(rows)


NOISY, MANIFEST = {}, {}
for artifact in ARTIFACTS_TO_EVALUATE:
    for protocol in PROTOCOLS:
        key = (artifact, protocol)
        NOISY[key], MANIFEST[key] = make_test_pairs(artifact, protocol)
manifest = pd.concat(MANIFEST.values(), ignore_index=True)
assert len(manifest) == len(CLEAN) * len(ARTIFACTS_TO_EVALUATE) * len(PROTOCOLS)
assert not manifest.duplicated(['artifact', 'protocol', 'sample_id']).any()
manifest.to_csv(OUT / 'manifest.csv', index=False)
display(manifest.groupby(['artifact', 'protocol', 'noise_channel']).size().rename('N'))
'''


INFERENCE = r'''def synchronize():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def compute_metric_frame(clean, noisy, prediction, metadata, artifact, protocol, k,
                         elapsed_ms, output_start, checkpoint_hash):
    clean, noisy, prediction = [np.asarray(values, dtype=np.float64)
                                for values in (clean, noisy, prediction)]
    error = prediction - clean
    numerator = np.sum(error ** 2, axis=1)
    signal_power = np.sum(clean ** 2, axis=1)
    input_power = np.sum((noisy - clean) ** 2, axis=1)
    repo_denominator = np.sum((prediction - np.mean(clean)) ** 2, axis=1)
    standard_denominator = np.sum((clean - clean.mean(axis=1, keepdims=True)) ** 2, axis=1)
    cosine_denominator = np.linalg.norm(clean, axis=1) * np.linalg.norm(prediction, axis=1)
    def safe_ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_in = 10 * np.log10(safe_ratio(signal_power, input_power))
        snr_out = 10 * np.log10(safe_ratio(signal_power, numerator))
    frame = metadata[['sample_id', 'patient_id', 'record_id', 'artifact', 'r']].copy()
    frame['protocol'] = protocol
    frame['K'] = k
    frame['checkpoint_sha256'] = checkpoint_hash
    frame['output_row'] = np.arange(output_start, output_start + len(clean))
    frame['inference_time_ms'] = elapsed_ms / len(clean)
    frame['ssd'] = numerator
    frame['rmse'] = np.sqrt(np.mean(error ** 2, axis=1))
    frame['mae'] = np.mean(np.abs(error), axis=1)
    frame['mad'] = np.max(np.abs(error), axis=1)
    frame['prd_repo'] = 100 * np.sqrt(safe_ratio(numerator, repo_denominator))
    frame['prd_standard'] = 100 * np.sqrt(safe_ratio(numerator, standard_denominator))
    frame['cosine_similarity'] = safe_ratio(np.sum(clean * prediction, axis=1), cosine_denominator)
    frame['snr_in_db'] = snr_in
    frame['snr_out_db'] = snr_out
    frame['snr_improvement_db'] = snr_out - snr_in
    return frame


checkpoint_hashes = {(row['artifact'], int(row['protocol'])): row['sha256']
                     for row in CHECKPOINT_INFO}
result_frames = []
for artifact in ARTIFACTS_TO_EVALUATE:
    for protocol in PROTOCOLS:
        key = (artifact, protocol)
        model = MODELS[key]
        prediction_maps = {
            k: np.lib.format.open_memmap(
                OUT / f'xhat_{artifact.lower()}_protocol{protocol}_K{k}.npy',
                mode='w+', dtype=np.float32, shape=CLEAN.shape,
            ) for k in K_VALUES
        }
        artifact_seed = {'BW': 11, 'EM': 22, 'MA': 33}[artifact]
        torch.manual_seed(SEED + artifact_seed * 100 + protocol)
        torch.cuda.manual_seed_all(SEED + artifact_seed * 100 + protocol)
        for start in tqdm(range(0, len(CLEAN), BATCH_SIZE), desc=f'{artifact}/P{protocol}'):
            stop = min(start + BATCH_SIZE, len(CLEAN))
            scale = np.maximum(
                np.max(np.abs(CLEAN[start:stop]), axis=1, keepdims=True), 1e-8
            ).astype(np.float32)
            condition = torch.from_numpy((NOISY[key][start:stop] / scale)[:, None, :]).to(DEVICE)
            running_sum = torch.zeros_like(condition)
            synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                for shot in range(1, max(K_VALUES) + 1):
                    output = model.denoising(
                        condition, use_ddim=True, ddim_steps=DDIM_STEPS,
                        ddim_eta=DDIM_ETA, num_shots=1,
                    )
                    if output.shape != condition.shape or not torch.isfinite(output).all():
                        raise RuntimeError(f'Invalid output: {artifact}/P{protocol}/shot{shot}')
                    running_sum += output
                    if shot in K_VALUES:
                        synchronize()
                        elapsed_ms = (time.perf_counter() - tick) * 1000
                        prediction = (running_sum / shot).cpu().numpy()[:, 0, :] * scale
                        prediction_maps[shot][start:stop] = prediction
                        result_frames.append(compute_metric_frame(
                            CLEAN[start:stop], NOISY[key][start:stop], prediction,
                            MANIFEST[key].iloc[start:stop].reset_index(drop=True),
                            artifact, protocol, shot, elapsed_ms, start,
                            checkpoint_hashes[key],
                        ))
        for values in prediction_maps.values():
            values.flush()

per_sample = pd.concat(result_frames, ignore_index=True)
expected_rows = len(CLEAN) * len(ARTIFACTS_TO_EVALUATE) * len(PROTOCOLS) * len(K_VALUES)
assert len(per_sample) == expected_rows
assert not per_sample.duplicated(['artifact', 'protocol', 'K', 'sample_id']).any()
per_sample.to_csv(OUT / 'metrics_per_sample.csv', index=False)
print('Metric rows:', len(per_sample))
'''


SUMMARY = r'''METRICS = ['ssd', 'rmse', 'mae', 'mad', 'prd_repo', 'prd_standard', 'cosine_similarity',
           'snr_in_db', 'snr_out_db', 'snr_improvement_db', 'inference_time_ms']
per_sample['r_group'] = pd.cut(
    per_sample.r, bins=[0.2, 0.6, 1.0, 1.5, 2.001], right=False,
    labels=['0.20-<0.60', '0.60-<1.00', '1.00-<1.50', '1.50-2.00'],
)
assert per_sample.r_group.notna().all()


def summarize(frame, artifact, protocol, k, r_group='ALL'):
    row = {'artifact': artifact, 'protocol': protocol, 'K': k,
           'r_group': r_group, 'N': len(frame)}
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        row[f'{metric}_valid_N'] = len(finite)
        row[f'{metric}_nonfinite_N'] = len(values) - len(finite)
        row[f'{metric}_mean'] = float(np.mean(finite)) if len(finite) else np.nan
        row[f'{metric}_std'] = float(np.std(finite, ddof=0)) if len(finite) else np.nan
    return row


summary_rows = []
for (artifact, protocol, k), frame in per_sample.groupby(['artifact', 'protocol', 'K']):
    summary_rows.append(summarize(frame, artifact, protocol, k))
    for label, part in frame.groupby('r_group', observed=True):
        summary_rows.append(summarize(part, artifact, protocol, k, str(label)))
for (artifact, k), frame in per_sample.groupby(['artifact', 'K']):
    summary_rows.append(summarize(frame, artifact, 'pooled', k))
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / 'summary.csv', index=False)

record_rows = []
for (artifact, protocol, k, record_id), frame in per_sample.groupby(
        ['artifact', 'protocol', 'K', 'record_id']):
    row = summarize(frame, artifact, protocol, k)
    row['record_id'] = record_id
    record_rows.append(row)
pd.DataFrame(record_rows).to_csv(OUT / 'summary_by_record.csv', index=False)

readable = summary[summary.r_group == 'ALL'][['artifact', 'protocol', 'K', 'N']].copy()
for metric in METRICS:
    all_rows = summary[summary.r_group == 'ALL']
    readable[metric] = [f'{mean:.4f} ± {std:.4f}' for mean, std in zip(
        all_rows[f'{metric}_mean'], all_rows[f'{metric}_std'])]
readable.to_csv(OUT / 'summary_mean_std.csv', index=False)
display(readable)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
colors = {'BW': '#4C78A8', 'EM': '#F58518', 'MA': '#54A24B'}
for artifact in ARTIFACTS_TO_EVALUATE:
    part = summary[(summary.artifact == artifact) &
                   (summary.protocol == 'pooled') &
                   (summary.r_group == 'ALL')]
    axes[0].plot(part.K, part.snr_improvement_db_mean, marker='o',
                 color=colors[artifact], label=artifact)
    axes[1].plot(part.K, part.prd_repo_mean, marker='o',
                 color=colors[artifact], label=artifact)
axes[0].set(title='SNR improvement by artifact', xlabel='K', ylabel='ΔSNR (dB)')
axes[1].set(title='PRD_repo by artifact', xlabel='K', ylabel='PRD_repo (%)')
for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend()
fig.tight_layout()
fig.savefig(OUT / 'artifact_comparison.png', dpi=160)
plt.show()

# Extended visual analysis. Error bars below show population standard deviation,
# not confidence intervals.
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#4A4A4A', 'axes.labelcolor': '#222222',
    'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': 0.20,
    'font.size': 10, 'legend.frameon': False,
})
COLORS = {'BW': '#4C78A8', 'EM': '#F58518', 'MA': '#54A24B'}
HATCHES = {1: '', 2: '//'}
DISPLAY_K = max(K_VALUES)
display_frame = per_sample[per_sample.K == DISPLAY_K].copy()
figure_manifest = []


def save_figure(fig, filename, purpose):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches='tight')
    figure_manifest.append({'file': filename, 'K': DISPLAY_K, 'purpose': purpose})
    plt.show()
    plt.close(fig)


# 1) Metric overview by artifact and protocol.
groups = [(artifact, protocol) for artifact in ARTIFACTS_TO_EVALUATE
          for protocol in PROTOCOLS]
group_labels = [f'{artifact}-P{protocol}' for artifact, protocol in groups]
overview_specs = [
    ('snr_improvement_db', 'SNR improvement', 'dB'),
    ('prd_standard', 'Standard PRD', '%'),
    ('rmse', 'RMSE', 'signal unit'),
    ('cosine_similarity', 'Cosine similarity', 'score'),
]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for axis, (metric, title, unit) in zip(axes.ravel(), overview_specs):
    values, spreads = [], []
    for artifact, protocol in groups:
        series = display_frame[(display_frame.artifact == artifact) &
                               (display_frame.protocol == protocol)][metric]
        values.append(series.mean())
        spreads.append(series.std(ddof=0))
    positions = np.arange(len(groups))
    bars = axis.bar(
        positions, values, yerr=spreads, capsize=3,
        color=[COLORS[a] for a, _ in groups], edgecolor='#333333', linewidth=0.6,
    )
    for bar, (_, protocol) in zip(bars, groups):
        bar.set_hatch(HATCHES[protocol])
    if metric == 'snr_improvement_db':
        axis.axhline(0, color='#333333', linewidth=1)
    axis.set(title=f'{title} — mean ± population SD', ylabel=unit,
             xticks=positions, xticklabels=group_labels)
    axis.tick_params(axis='x', rotation=30)
fig.suptitle(f'Artifact/protocol metric overview (K={DISPLAY_K}, N shown in CSV)', y=1.01)
save_figure(fig, 'metric_overview_by_artifact_protocol.png',
            'Mean and population spread of four principal metrics by artifact and protocol.')


# 2) Performance by controlled noise-scaling range r, pooled across protocols.
severity_specs = [
    ('snr_improvement_db', 'SNR improvement', 'dB'),
    ('prd_standard', 'Standard PRD', '%'),
    ('rmse', 'RMSE', 'signal unit'),
    ('cosine_similarity', 'Cosine similarity', 'score'),
]
r_order = [str(value) for value in display_frame.r_group.cat.categories]
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for axis, (metric, title, unit) in zip(axes.ravel(), severity_specs):
    for artifact in ARTIFACTS_TO_EVALUATE:
        part = (display_frame[display_frame.artifact == artifact]
                .groupby('r_group', observed=True)[metric].mean().reindex(r_order))
        axis.plot(r_order, part.values, marker='o', linewidth=2,
                  color=COLORS[artifact], label=artifact)
    if metric == 'snr_improvement_db':
        axis.axhline(0, color='#333333', linewidth=1)
    axis.set(title=f'{title} by r range', ylabel=unit, xlabel='Noise scale r')
    axis.legend(title='Artifact')
    axis.tick_params(axis='x', rotation=20)
fig.suptitle(f'Robustness across noise severity (protocols pooled, K={DISPLAY_K})', y=1.01)
save_figure(fig, 'performance_by_noise_level.png',
            'Mean denoising quality across the four predefined r ranges.')


# 3) Full distributions expose medians, tails, and unstable cases hidden by means.
distribution_specs = [
    ('snr_improvement_db', 'SNR improvement distribution', 'dB'),
    ('prd_standard', 'Standard PRD distribution', '%'),
]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for axis, (metric, title, unit) in zip(axes, distribution_specs):
    arrays = [display_frame[(display_frame.artifact == artifact) &
                            (display_frame.protocol == protocol)][metric]
              .replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
              for artifact, protocol in groups]
    boxplot_options = dict(showfliers=False, patch_artist=True, widths=0.65)
    try:
        boxes = axis.boxplot(arrays, tick_labels=group_labels, **boxplot_options)
    except TypeError:  # Matplotlib < 3.9
        boxes = axis.boxplot(arrays, labels=group_labels, **boxplot_options)
    for box, (artifact, protocol) in zip(boxes['boxes'], groups):
        box.set(facecolor=COLORS[artifact], alpha=0.70,
                edgecolor='#333333', hatch=HATCHES[protocol])
    if metric == 'snr_improvement_db':
        axis.axhline(0, color='#333333', linewidth=1)
    axis.set(title=title + ' (outliers hidden)', ylabel=unit)
    axis.tick_params(axis='x', rotation=30)
save_figure(fig, 'metric_distributions.png',
            'Median and interquartile spread by artifact/protocol; visual outliers are hidden only in the plot.')


# 4) Input versus output SNR. Points above y=x indicate improvement.
artifact_count = len(ARTIFACTS_TO_EVALUATE)
fig, axes = plt.subplots(1, artifact_count, figsize=(6 * artifact_count, 5),
                         squeeze=False, sharex=True, sharey=True)
scatter_sample = display_frame.sample(min(6000, len(display_frame)), random_state=SEED)
finite_snr = scatter_sample[['snr_in_db', 'snr_out_db']].replace(
    [np.inf, -np.inf], np.nan).dropna()
snr_low = float(finite_snr.min().min())
snr_high = float(finite_snr.max().max())
for axis, artifact in zip(axes.ravel(), ARTIFACTS_TO_EVALUATE):
    for protocol, marker in ((1, 'o'), (2, '^')):
        part = scatter_sample[(scatter_sample.artifact == artifact) &
                              (scatter_sample.protocol == protocol)]
        axis.scatter(part.snr_in_db, part.snr_out_db, s=10, alpha=0.22,
                     color=COLORS[artifact], marker=marker, label=f'P{protocol}')
    axis.plot([snr_low, snr_high], [snr_low, snr_high], '--', color='#333333',
              linewidth=1.2, label='No change')
    axis.set(title=artifact, xlabel='Input SNR (dB)', ylabel='Output SNR (dB)',
             xlim=(snr_low, snr_high), ylim=(snr_low, snr_high))
    axis.legend()
fig.suptitle(f'Input–output SNR relationship (deterministic sample, K={DISPLAY_K})', y=1.02)
save_figure(fig, 'snr_input_output_scatter.png',
            'Per-segment relationship between input and output SNR; the diagonal is no improvement.')


# 5) Record-level heatmap detects whether conclusions depend on a few QTDB records.
record_metric = (display_frame.groupby(['record_id', 'artifact', 'protocol'])
                 ['snr_improvement_db'].mean().unstack(['artifact', 'protocol']))
record_columns = [(artifact, protocol) for artifact, protocol in groups
                  if (artifact, protocol) in record_metric.columns]
record_metric = record_metric.reindex(columns=pd.MultiIndex.from_tuples(record_columns))
matrix = record_metric.to_numpy(dtype=float)
limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
limit = max(limit, 1e-6)
fig, axis = plt.subplots(figsize=(max(8, 1.35 * len(record_columns)),
                                  max(6, 0.42 * len(record_metric))))
image = axis.imshow(matrix, aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
axis.grid(False)
axis.set(xticks=np.arange(len(record_columns)),
         xticklabels=[f'{a}-P{p}' for a, p in record_columns],
         yticks=np.arange(len(record_metric)), yticklabels=record_metric.index,
         xlabel='Artifact / protocol', ylabel='QTDB record',
         title=f'Mean SNR improvement by record (dB, K={DISPLAY_K})')
axis.tick_params(axis='x', rotation=30)
fig.colorbar(image, ax=axis, label='Mean ΔSNR (dB)')
save_figure(fig, 'record_snr_improvement_heatmap.png',
            'Record-level mean SNR improvement for checking cross-record consistency.')


# 6) Signal and spectrum examples at low, median, and high r for every experiment.
time_ms = np.arange(LENGTH) / TARGET_FS * 1000.0
for artifact, protocol in groups:
    part = display_frame[(display_frame.artifact == artifact) &
                         (display_frame.protocol == protocol)].copy()
    prediction_path = OUT / f'xhat_{artifact.lower()}_protocol{protocol}_K{DISPLAY_K}.npy'
    predictions = np.load(prediction_path, mmap_mode='r')
    targets = part.r.quantile([0.15, 0.50, 0.85]).to_numpy()
    selected = [part.iloc[int(np.argmin(np.abs(part.r.to_numpy() - target)))]
                for target in targets]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for row_index, sample in enumerate(selected):
        output_row = int(sample.output_row)
        clean = CLEAN[output_row]
        noisy = NOISY[(artifact, protocol)][output_row]
        prediction = np.asarray(predictions[output_row]).copy()
        axis_time, axis_spectrum = axes[row_index]
        axis_time.plot(time_ms, noisy, color='#9E9E9E', linewidth=1.0, alpha=0.85,
                       label='Noisy')
        axis_time.plot(time_ms, clean, color='#111111', linewidth=1.3, label='Clean')
        axis_time.plot(time_ms, prediction, color=COLORS[artifact], linewidth=1.1,
                       label='Denoised')
        axis_time.set(title=(f'r={sample.r:.2f}, ΔSNR={sample.snr_improvement_db:.2f} dB, '
                             f'PRD={sample.prd_standard:.2f}%'),
                      xlabel='Time (ms)', ylabel='Amplitude')
        axis_time.legend(ncol=3, loc='upper right')
        for values, label, color, linewidth in (
                (noisy, 'Noisy', '#9E9E9E', 1.0),
                (clean, 'Clean', '#111111', 1.3),
                (prediction, 'Denoised', COLORS[artifact], 1.1)):
            frequencies, density = signal.welch(values, fs=TARGET_FS, nperseg=256)
            axis_spectrum.plot(frequencies, 10 * np.log10(density + 1e-12),
                               color=color, linewidth=linewidth, label=label)
        axis_spectrum.set(xlim=(0, 100), xlabel='Frequency (Hz)',
                          ylabel='PSD (dB/Hz)', title='Power spectral density')
        axis_spectrum.legend()
    fig.suptitle(f'{artifact} protocol {protocol}: representative ECG restoration (K={DISPLAY_K})',
                 y=1.01)
    save_figure(
        fig, f'signal_examples_{artifact.lower()}_protocol{protocol}_K{DISPLAY_K}.png',
        f'Clean/noisy/denoised waveforms and spectra for {artifact} protocol {protocol}.',
    )
    del predictions

pd.DataFrame(figure_manifest).to_csv(OUT / 'figure_manifest.csv', index=False)
display(pd.DataFrame(figure_manifest))

run_info = {
    'artifacts': list(ARTIFACTS_TO_EVALUATE), 'protocols': list(PROTOCOLS),
    'test_records': list(selected_records), 'N_clean': len(CLEAN),
    'sampling_rate': TARGET_FS, 'segment_length': LENGTH,
    'preprocessing_version': PREPROCESSING_VERSION,
    'model_normalization': MODEL_NORMALIZATION,
    'K': list(K_VALUES), 'batch_size': BATCH_SIZE,
    'ddim_steps': DDIM_STEPS, 'ddim_eta': DDIM_ETA,
    'checkpoint_info': CHECKPOINT_INFO,
    'device': str(DEVICE), 'torch_version': torch.__version__,
    'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
    'wfdb_version': wfdb.__version__, 'python_version': platform.python_version(),
    'full_evaluation': MAX_TEST_RECORDS is None and MAX_TEST_SEGMENTS is None,
}
(OUT / 'run_info.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')

archive_path = OUT.parent / 'qtdb_bw_em_ma_evaluation.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(OUT))
print('Saved:', OUT, '| archive:', archive_path)
'''


kaggle_cells = [
    markdown("""# Kaggle training — QTDB với BW, EM và MA

Notebook huấn luyện sáu model độc lập: ba artifact NSTDB × hai protocol tách kênh/nửa thời gian. BW đã có trong pipeline cũ; notebook này áp dụng cùng một giao thức cho EM và MA. Bật GPU và Internet.

Mặc định mỗi Kaggle version chạy thêm 5 epoch cho từng experiment rồi lưu checkpoint. Gắn Output của version trước bằng **Add Input** để tự động resume. Để tiết kiệm thời gian, có thể đặt `ARTIFACTS_TO_TRAIN=('EM',)` hoặc `('MA',)` và chạy từng artifact riêng."""),
    markdown("## 1. Cài thư viện"), code(COMMON_INSTALL),
    markdown("## 2. Cấu hình tái lập và phiên training"), code(TRAIN_CONFIG),
    markdown("## 3. Kiến trúc HNF U-Net + FiLM + attention"), code(ARCHITECTURE),
    markdown("## 4. Conditional DDPM và loss đa miền"), code(DDPM),
    markdown("## 5. Hàm tạo QTDB clean segments"), code(CLEAN_FUNCTIONS),
    markdown("## 6. Cache helpers"), code(CACHE_HELPERS),
    markdown("## 7. Tạo QTDB train/validation"), code(TRAIN_DATA),
    markdown("## 8. Tải ba artifact NSTDB"), code(LOAD_NOISE),
    markdown("## 9. Dataset cố định theo artifact/protocol"), code(DATASET),
    markdown("## 10. Train/resume sáu experiment"), code(TRAIN_LOOP),
    markdown("## 11. Đồ thị và đóng gói output"), code(TRAIN_PACKAGE),
    markdown("""## Checkpoint contract

Mỗi checkpoint best có tên `best_<artifact>_protocol<1|2>_norm_loss_v2.pth` và chứa artifact, protocol, config, raw state, EMA state, optimizer, scheduler, epoch và validation loss. Notebook Colab chỉ chấp nhận checkpoint đúng contract này."""),
]


colab_cells = [
    markdown("""# Colab evaluation — QTDB với BW, EM và MA

Notebook đánh giá sáu checkpoint trên cùng 14 QTDB test records. Mỗi artifact dùng đúng nửa sau và kênh đối nghịch với phần training của protocol tương ứng. Mặc định chạy full test với DDIM 50 bước, `eta=0`, 1-shot.

Đặt các checkpoint best trong thư mục Google Drive cấu hình ở cell 3, bật GPU và Internet, rồi **Runtime → Run all**."""),
    markdown("## 1. Cài thư viện"), code(COMMON_INSTALL.replace("%pip", "!pip")),
    markdown("## 2. Mount Drive và cấu hình"), code(EVAL_CONFIG),
    markdown("## 3. Kiến trúc HNF U-Net + FiLM + attention"), code(ARCHITECTURE),
    markdown("## 4. Conditional DDPM/DDIM"), code(DDPM),
    markdown("## 5. Nạp và kiểm tra sáu checkpoint EMA"), code(LOAD_MODELS),
    markdown("## 6. Hàm tạo QTDB clean segments"), code(CLEAN_FUNCTIONS),
    markdown("## 7. Tạo test set và tải BW/EM/MA"), code(TEST_DATA),
    markdown("## 8. Tạo noisy pairs cho từng artifact/protocol"), code(MAKE_TEST_PAIRS),
    markdown("## 9. DDIM inference và metric từng segment"), code(INFERENCE),
    markdown("## 10. Tổng hợp, biểu đồ và đóng gói"), code(SUMMARY),
    markdown("""## Output

- `metrics_per_sample.csv`: metric từng segment;
- `summary.csv`, `summary_mean_std.csv`: kết quả theo artifact/protocol/K;
- `summary_by_record.csv`: kết quả theo QTDB record;
- `manifest.csv`: provenance của clean/noise pair;
- `xhat_<artifact>_protocol<P>_K<K>.npy`: tín hiệu khôi phục;
- `metric_overview_by_artifact_protocol.png`: bốn metric chính theo nhiễu/protocol;
- `performance_by_noise_level.png`: độ bền theo các nhóm mức nhiễu `r`;
- `metric_distributions.png`, `snr_input_output_scatter.png`: phân phối và quan hệ SNR;
- `record_snr_improvement_heatmap.png`: độ ổn định qua các QTDB records;
- `signal_examples_<artifact>_protocol<P>_K<K>.png`: clean/noisy/denoised và phổ tần;
- `figure_manifest.csv`, `artifact_comparison.png`, `run_info.json` và file ZIP tổng."""),
]


KAGGLE_OUT.write_text(json.dumps(notebook(kaggle_cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
COLAB_OUT.write_text(json.dumps(notebook(colab_cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(KAGGLE_OUT)
print(COLAB_OUT)
