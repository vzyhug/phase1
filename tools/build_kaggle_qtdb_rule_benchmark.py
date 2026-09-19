"""Build a standalone Kaggle notebook from the repository's model source."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_qtdb_rule_benchmark.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(True)}


def source(path, remove=()):
    value = (ROOT / path).read_text(encoding="utf-8")
    for line in remove:
        value = value.replace(line, "")
    return value


cells = [
    md("""# QTDB BW benchmark — conditional DDPM + HNF U-Net của nghiên cứu

Notebook Kaggle độc lập này dùng **QT Database làm ground truth tạm thời** và đánh giá đúng hai protocol BW của `rule.md`. Kiến trúc lấy nguyên từ `src/models`: HNF đa tỉ lệ, FiLM bridge, attention bottleneck, U-Net 1D và DDPM. Giao diện là 1 kênh vì QT benchmark có 1 kênh; `in_channels=2`, `out_channels=1`. Không áp chuẩn hóa biên độ, resample 500 Hz hay ba loại artifact của pipeline LUDB lên benchmark này.

**Điều kiện chạy:** bật Internet trong Kaggle để đọc PhysioNet; đính kèm hai checkpoint có tên `qtdb_1ch_noise_type_1_best.pth` và `qtdb_1ch_noise_type_2_best.pth` bằng Add Input. Chạy tất cả cell từ đầu. Mặc định dùng đủ 14 test record và K = 1, 3, 5, 10. Kết quả ghi vào `/kaggle/working/qtdb_rule_eval/`.

**Phạm vi:** Đây là benchmark BW một kênh. Notebook dùng checkpoint đã huấn luyện, không train lại. Checkpoint dạng state dict thuần không chứa epoch, validation score hoặc train seed; notebook xác minh trọng số/kiến trúc nhưng không suy đoán lịch sử huấn luyện. Nếu các checkpoint chưa được chọn bằng validation, cần chọn lại trước khi công bố số test.
"""),
    md("## 1. Cài thư viện và khóa cấu hình"),
    code("""%pip -q install 'wfdb>=4.1,<5' 'scipy>=1.10' 'pandas>=2.0' 'matplotlib>=3.7' 'pyyaml>=6.0'\n"""),
    code("""import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import torch
import wfdb
from IPython.display import display

SEED = 1234
TEST_RECORDS = (
    'sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853',
    'sel16420', 'sel16795', 'sele0106', 'sele0121', 'sel32',
    'sel49', 'sel14046', 'sel15814',
)
TARGET_FS = 360
LENGTH = 512
MAX_UNPADDED = 496
INSERT_OFFSET = 16
BATCH_SIZE = 50
K_VALUES = (1, 3, 5, 10)
DDIM_STEPS = 50
DDIM_ETA = 0.0
PREPROCESSING_VERSION = 'qtdb_pu1_p_and_beat_40ms_reflect100ms_360hz_endpoints_pad16_v2'
PHYSIONET_QT = 'qtdb/1.0.0'
PHYSIONET_NST = 'nstdb/1.0.0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = Path('/kaggle/working/qtdb_rule_eval') if Path('/kaggle/working').exists() else Path('qtdb_rule_eval')
OUT.mkdir(parents=True, exist_ok=True)

# Chỉ sửa hai giới hạn này để smoke test; để None khi báo cáo benchmark.
MAX_TEST_RECORDS = None
MAX_TEST_SEGMENTS = None

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print('Device:', DEVICE, '| output:', OUT.resolve())
"""),
    md("## 2. Kiến trúc gốc của nghiên cứu — mã được chép trực tiếp từ `src` khi tạo notebook"),
    code("import torch.nn as nn\nimport torch.nn.functional as F\nimport math\n" +
         source("src/models/blocks/hnf_block.py") + "\n" +
         source("src/models/blocks/bridge_block.py") + "\n" +
         source("src/models/blocks/attention_1d.py") + "\n" +
         source("src/models/unet_1d.py", ("from .blocks import HNFBlock, BridgeBlock, SelfAttention1D\n",))),
    md("### DDPM và DDIM đúng theo mã của nghiên cứu"),
    code("from functools import partial\nfrom inspect import isfunction\nfrom tqdm.auto import tqdm\n" +
         source("src/models/main_model.py", ("from tqdm import tqdm\n",))),
    md("## 3. Nạp hai checkpoint, kiểm tra đúng toàn bộ tensor"),
    code("""CONFIG = {
    'train': {'feats': 80, 'batch_size': 96, 'epochs': 400, 'lr': 1e-4,
              'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
}
CHECKPOINT_NAMES = {
    1: 'qtdb_1ch_noise_type_1_best.pth',
    2: 'qtdb_1ch_noise_type_2_best.pth',
}

def locate_checkpoint(filename):
    roots = [Path('/kaggle/input'), Path('checkpoints'), Path('.')]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(path for path in root.rglob(filename) if path.is_file())
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'Expected exactly one {filename}; found {len(candidates)}: {candidates}. '
            'Attach the matching QTDB one-channel checkpoint in Kaggle Add Input.'
        )
    return candidates[0]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_model(protocol):
    path = locate_checkpoint(CHECKPOINT_NAMES[protocol])
    base = UNet1D(in_channels=2, base_channels=80, emb_dim=128, out_channels=1)
    model = DDPM(base, CONFIG, DEVICE).to(DEVICE)
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    state = payload.get('state_dict', payload)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    probe = torch.zeros(1, 1, LENGTH, device=DEVICE)
    with torch.inference_mode():
        result = model.model(probe, probe, torch.ones(1, 1, device=DEVICE))
    assert result.shape == probe.shape and torch.isfinite(result).all()
    info = {'protocol': protocol, 'path': str(path), 'sha256': sha256(path),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'checkpoint_epoch': payload.get('epoch') if isinstance(payload, dict) else None,
            'checkpoint_validation': payload.get('best_validation') if isinstance(payload, dict) else None}
    return model, info

MODELS = {}
CHECKPOINT_INFO = []
for protocol in (1, 2):
    MODELS[protocol], info = load_model(protocol)
    CHECKPOINT_INFO.append(info)
    print(f'Protocol {protocol}: strict load OK; SHA256={info["sha256"][:16]}...')
display(pd.DataFrame(CHECKPOINT_INFO))
"""),
    md("## 4. QTDB ground truth: P onset, 360 Hz, 512 mẫu"),
    code("""BEAT_SYMBOLS = set('NLRBAaJSVrFejnE/fQ?')

def p_onsets(annotation):
    symbols = np.asarray(annotation.symbol)
    samples = np.asarray(annotation.sample, dtype=np.int64)
    return np.asarray([int(samples[i]) for i, symbol in enumerate(symbols)
                       if symbol == '(' and 'p' in symbols[i + 1:i + 4]], dtype=np.int64)

def r_locations(annotation):
    return np.asarray([int(sample) for sample, symbol in zip(annotation.sample, annotation.symbol)
                       if symbol in BEAT_SYMBOLS], dtype=np.int64)

def resample_reflect(segment, original_fs):
    pad = min(max(1, int(round(0.1 * original_fs))), len(segment) - 1)
    padded = np.pad(segment, (pad, pad), mode='reflect')
    new_length = int(round(len(padded) * TARGET_FS / original_fs))
    resampled = signal.resample(padded, new_length).astype(np.float32)
    target_pad = int(round(pad * TARGET_FS / original_fs))
    expected = int(round(len(segment) * TARGET_FS / original_fs))
    return resampled[target_pad:target_pad + expected]

def clean_segments_for_record(name):
    record = wfdb.rdrecord(name, pn_dir=PHYSIONET_QT, channels=[0])
    # QTDB does not provide .atr for every record (e.g. sel30).
    # .pu1 provides P boundaries and beat fiducials throughout each record.
    pu = wfdb.rdann(name, 'pu1', pn_dir=PHYSIONET_QT)
    fs = float(record.fs)
    ecg = record.p_signal[:, 0].astype(np.float32)
    starts = p_onsets(pu) - int(round(0.04 * fs))
    starts = starts[(starts >= 0) & (starts < len(ecg))]
    peaks = r_locations(pu)
    beats, rows = [], []
    audit = {'record_id': name, 'p_onsets': len(starts),
             'candidate_intervals': max(0, len(starts) - 1),
             'rejected_short_interval': 0, 'rejected_multiple_r': 0,
             'rejected_length': 0, 'rejected_nonfinite_or_flat': 0}
    resampled_lengths = []
    for segment_index, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if end <= start + 2:
            audit['rejected_short_interval'] += 1
            continue
        r_count = int(np.count_nonzero((peaks >= start) & (peaks < end)))
        if r_count > 1:
            audit['rejected_multiple_r'] += 1
            continue
        segment = resample_reflect(ecg[start:end], fs)
        resampled_lengths.append(len(segment))
        if not 2 <= len(segment) <= MAX_UNPADDED:
            audit['rejected_length'] += 1
            continue
        segment = segment - 0.5 * (segment[0] + segment[-1])
        clean = np.zeros(LENGTH, dtype=np.float32)
        clean[INSERT_OFFSET:INSERT_OFFSET + len(segment)] = segment
        if not np.all(np.isfinite(clean)) or np.ptp(clean) <= 0:
            audit['rejected_nonfinite_or_flat'] += 1
            continue
        beats.append(clean)
        rows.append({'sample_id': f'{name}:{segment_index}', 'patient_id': name,
                     'record_id': name, 'clean_segment_id': f'{name}:{segment_index}',
                     'start_sample': int(start), 'end_sample': int(end),
                     'unpadded_length': len(segment), 'r_count': r_count,
                     'annotation_source': 'pu1'})
    audit['min_resampled_length'] = min(resampled_lengths) if resampled_lengths else np.nan
    audit['max_resampled_length'] = max(resampled_lengths) if resampled_lengths else np.nan
    audit['accepted_segments'] = len(beats)
    return beats, rows, audit

selected_records = TEST_RECORDS[:MAX_TEST_RECORDS] if MAX_TEST_RECORDS else TEST_RECORDS
all_beats, clean_rows, test_audit_rows = [], [], []
for record_name in tqdm(selected_records, desc='QTDB test records'):
    beats, rows, audit = clean_segments_for_record(record_name)
    test_audit_rows.append(audit)
    all_beats.extend(beats)
    clean_rows.extend(rows)
    print(record_name, len(beats))
    if MAX_TEST_SEGMENTS and len(all_beats) >= MAX_TEST_SEGMENTS:
        all_beats = all_beats[:MAX_TEST_SEGMENTS]
        clean_rows = clean_rows[:MAX_TEST_SEGMENTS]
        break
pd.DataFrame(test_audit_rows).to_csv(OUT / 'test_record_audit.csv', index=False)
if not all_beats:
    raise RuntimeError('No valid QTDB test segments after rule.md filtering; inspect test_record_audit.csv')
CLEAN = np.stack(all_beats).astype(np.float32)
CLEAN_META = pd.DataFrame(clean_rows)
assert CLEAN.shape == (len(CLEAN_META), LENGTH)
assert CLEAN_META.sample_id.is_unique
assert set(CLEAN_META.record_id).issubset(TEST_RECORDS)
np.save(OUT / 'clean_test.npy', CLEAN)
print('Clean test segments:', len(CLEAN), '| shape:', CLEAN.shape)
"""),
    md("## 5. NSTDB BW: hai protocol và manifest cố định"),
    code("""noise_record = wfdb.rdrecord('bw', pn_dir=PHYSIONET_NST)
BW = noise_record.p_signal.astype(np.float32)
if int(noise_record.fs) != TARGET_FS:
    BW = signal.resample(BW, int(round(len(BW) * TARGET_FS / float(noise_record.fs))), axis=0).astype(np.float32)
assert BW.ndim == 2 and BW.shape[1] >= 2 and np.isfinite(BW).all()
MIDPOINT = len(BW) // 2
NOISE_LEVELS = np.arange(20, 200, dtype=np.int16) / 100.0

def make_protocol_pairs(protocol):
    channel = 1 if protocol == 1 else 0  # Zero-based: P1 tests channel 2; P2 tests channel 1.
    noise = BW[MIDPOINT:, channel]
    patches = len(noise) // LENGTH
    assert patches > 0
    rng = np.random.RandomState(SEED)
    r_values = rng.randint(20, 200, size=len(CLEAN)) / 100.0
    noisy = np.empty_like(CLEAN)
    rows = []
    for index, (clean, r) in enumerate(zip(CLEAN, r_values)):
        relative_start = (index % patches) * LENGTH
        patch = noise[relative_start:relative_start + LENGTH]
        clean_range, noise_range = np.ptp(clean), np.ptp(patch)
        if clean_range <= 0 or noise_range <= 0:
            raise ValueError(f'Zero amplitude range at sample {index}, protocol {protocol}')
        alpha = float(r) * float(clean_range) / float(noise_range)
        noisy[index] = clean + alpha * patch
        row = CLEAN_META.iloc[index].to_dict()
        row.update({'protocol': protocol, 'split': 'test', 'noise_type': 'BW',
                    'noise_source_id': 'nstdb/1.0.0/bw:second_half',
                    'noise_channel': channel + 1,
                    'noise_start': MIDPOINT + relative_start,
                    'r': float(r), 'sampling_rate': TARGET_FS,
                    'preprocessing_version': PREPROCESSING_VERSION, 'seed': SEED})
        rows.append(row)
    return noisy, pd.DataFrame(rows)

NOISY, MANIFEST = {}, {}
for protocol in (1, 2):
    NOISY[protocol], MANIFEST[protocol] = make_protocol_pairs(protocol)
    np.save(OUT / f'noisy_protocol{protocol}.npy', NOISY[protocol])
manifest = pd.concat([MANIFEST[1], MANIFEST[2]], ignore_index=True)
assert len(manifest) == 2 * len(CLEAN)
assert not manifest.duplicated(['sample_id', 'protocol']).any()
assert manifest.groupby('protocol').size().eq(len(CLEAN)).all()
assert np.isfinite(CLEAN).all() and all(np.isfinite(x).all() for x in NOISY.values())
assert manifest.r.between(0.2, 1.99).all()
manifest.to_csv(OUT / 'manifest.csv', index=False)
display(manifest.groupby(['protocol', 'noise_channel']).size().rename('N').reset_index())
"""),
    md("## 6. Suy luận DDIM 50 bước; K lượt Gaussian độc lập trên cùng input"),
    code("""def synchronize():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

def metric_rows(clean, noisy, pred, meta, protocol, k, batch_time_ms, batch_start):
    clean, noisy, pred = [np.asarray(a, dtype=np.float64) for a in (clean, noisy, pred)]
    diff = pred - clean
    numerator = np.sum(diff ** 2, axis=1)
    signal_power = np.sum(clean ** 2, axis=1)
    input_power = np.sum((noisy - clean) ** 2, axis=1)
    repo_denominator = np.sum((pred - np.mean(clean)) ** 2, axis=1)
    standard_denominator = np.sum((clean - clean.mean(axis=1, keepdims=True)) ** 2, axis=1)
    cosine_denominator = np.linalg.norm(clean, axis=1) * np.linalg.norm(pred, axis=1)
    def safe_ratio(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_in = 10 * np.log10(safe_ratio(signal_power, input_power))
        snr_out = 10 * np.log10(safe_ratio(signal_power, numerator))
        values = {
            'ssd': numerator,
            'mad': np.max(np.abs(diff), axis=1),
            'prd_repo': 100 * np.sqrt(safe_ratio(numerator, repo_denominator)),
            'prd_standard': 100 * np.sqrt(safe_ratio(numerator, standard_denominator)),
            'cosine_similarity': safe_ratio(np.sum(clean * pred, axis=1), cosine_denominator),
            'snr_in_db': snr_in, 'snr_out_db': snr_out,
            'snr_improvement_db': snr_out - snr_in,
        }
    frame = meta[['sample_id', 'patient_id', 'record_id', 'noise_type', 'r']].copy()
    frame['experiment_id'] = 'qtdb_bw_rule_v1'
    frame['model_id'] = 'research_hnf_unet_ddpm_1ch'
    frame['checkpoint_id'] = CHECKPOINT_INFO[protocol - 1]['sha256']
    frame['train_seed'] = np.nan  # Plain state_dict does not prove training seed.
    frame['inference_seed'] = SEED + protocol
    frame['protocol'] = protocol
    frame['K'] = k
    frame['output_row'] = np.arange(batch_start, batch_start + len(clean))
    frame['output_file'] = f'xhat_protocol{protocol}_K{k}.npy'
    frame['inference_time_ms'] = batch_time_ms / len(clean)
    for key, vector in values.items():
        frame[key] = vector
    return frame

RESULT_FRAMES = []
for protocol in (1, 2):
    model = MODELS[protocol]
    predictions = {k: np.lib.format.open_memmap(
        OUT / f'xhat_protocol{protocol}_K{k}.npy', mode='w+',
        dtype=np.float32, shape=CLEAN.shape) for k in K_VALUES}
    torch.manual_seed(SEED + protocol)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(SEED + protocol)
    for start in tqdm(range(0, len(CLEAN), BATCH_SIZE), desc=f'Protocol {protocol}'):
        stop = min(start + BATCH_SIZE, len(CLEAN))
        condition = torch.from_numpy(NOISY[protocol][start:stop, None, :]).to(DEVICE)
        assert condition.shape == (stop - start, 1, LENGTH)
        running_sum = torch.zeros_like(condition)
        synchronize()
        tick = time.perf_counter()
        with torch.inference_mode():
            for shot in range(1, max(K_VALUES) + 1):
                output = model.denoising(condition, use_ddim=True,
                                         ddim_steps=DDIM_STEPS, ddim_eta=DDIM_ETA,
                                         num_shots=1)
                assert output.shape == condition.shape and torch.isfinite(output).all()
                running_sum += output
                if shot in K_VALUES:
                    synchronize()
                    elapsed_ms = (time.perf_counter() - tick) * 1000
                    average = (running_sum / shot).cpu().numpy()[:, 0, :]
                    predictions[shot][start:stop] = average
                    RESULT_FRAMES.append(metric_rows(
                        CLEAN[start:stop], NOISY[protocol][start:stop], average,
                        MANIFEST[protocol].iloc[start:stop].reset_index(drop=True),
                        protocol, shot, elapsed_ms, start))
    for array in predictions.values():
        array.flush()

per_sample = pd.concat(RESULT_FRAMES, ignore_index=True)
assert len(per_sample) == 2 * len(CLEAN) * len(K_VALUES)
per_sample.to_csv(OUT / 'metrics_per_sample.csv', index=False)
print('Per-sample rows:', len(per_sample))
"""),
    md("## 7. Kiểm toán và tổng hợp mean ± std theo protocol, pooled và nhóm r"),
    code("""METRICS = ['ssd', 'mad', 'prd_repo', 'prd_standard', 'cosine_similarity',
           'snr_in_db', 'snr_out_db', 'snr_improvement_db', 'inference_time_ms']
assert not per_sample.duplicated(['protocol', 'K', 'sample_id']).any()
assert np.allclose(per_sample.snr_improvement_db,
                   per_sample.snr_out_db - per_sample.snr_in_db, equal_nan=True)
per_sample['r_group'] = pd.cut(
    per_sample.r, bins=[0.2, 0.6, 1.0, 1.5, 2.001], right=False,
    labels=['0.20–<0.60', '0.60–<1.00', '1.00–<1.50', '1.50–2.00'])
assert per_sample.r_group.notna().all()
assert per_sample.groupby(['protocol', 'K'], observed=True).r_group.count().eq(len(CLEAN)).all()

def summarize(frame, protocol_label, group_label):
    result = {'model_id': 'research_hnf_unet_ddpm_1ch',
              'protocol': protocol_label, 'K': int(frame.K.iloc[0]),
              'r_group': group_label, 'N': len(frame)}
    for metric in METRICS:
        vector = frame[metric].to_numpy(dtype=np.float64)
        finite = vector[np.isfinite(vector)]
        result[f'{metric}_valid_N'] = len(finite)
        result[f'{metric}_nonfinite_N'] = len(vector) - len(finite)
        result[f'{metric}_mean'] = float(np.mean(finite)) if len(finite) else np.nan
        result[f'{metric}_std'] = float(np.std(finite, ddof=0)) if len(finite) else np.nan
    return result

summary_rows = []
for k in K_VALUES:
    subset_k = per_sample[per_sample.K == k]
    for protocol in (1, 2, 'pooled'):
        frame = subset_k if protocol == 'pooled' else subset_k[subset_k.protocol == protocol]
        summary_rows.append(summarize(frame, protocol, 'ALL'))
        for label, part in frame.groupby('r_group', observed=True):
            summary_rows.append(summarize(part, protocol, str(label)))
summary = pd.DataFrame(summary_rows)
assert summary[summary.r_group == 'ALL'].groupby('K').N.sum().eq(4 * len(CLEAN)).all()
for (protocol, k), frame in summary.groupby(['protocol', 'K']):
    assert frame.loc[frame.r_group != 'ALL', 'N'].sum() == frame.loc[frame.r_group == 'ALL', 'N'].iloc[0]
# Record means keep the independent QTDB record visible separately from segment-weighted tables.
record_rows = []
for (protocol, k, record_id), frame in per_sample.groupby(['protocol', 'K', 'record_id']):
    row = summarize(frame, protocol, 'ALL')
    row['record_id'] = record_id
    record_rows.append(row)
record_summary = pd.DataFrame(record_rows)
per_sample.to_csv(OUT / 'metrics_per_sample.csv', index=False)
summary.to_csv(OUT / 'summary.csv', index=False)
record_summary.to_csv(OUT / 'summary_by_record.csv', index=False)
readable = summary[['protocol', 'K', 'r_group', 'N']].copy()
for metric in METRICS:
    readable[metric] = [f'{mean:.4f} ± {std:.4f}'
                        for mean, std in zip(summary[f'{metric}_mean'], summary[f'{metric}_std'])]
readable.to_csv(OUT / 'summary_mean_std.csv', index=False)
display(readable.loc[readable.r_group == 'ALL'])
print('Grouped by r (segment-weighted):')
display(readable.loc[readable.r_group != 'ALL',
                     ['protocol', 'K', 'r_group', 'N', 'prd_repo', 'snr_improvement_db']])
display(summary.loc[summary.r_group == 'ALL',
                    ['protocol', 'K', 'N'] + [f'{m}_nonfinite_N' for m in METRICS]])
"""),
    md("## 8. Kiểm tra trực quan và lưu điều kiện thí nghiệm"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for protocol in (1, 2):
    part = summary[(summary.protocol == protocol) & (summary.r_group == 'ALL')]
    axes[0].plot(part.K, part.snr_improvement_db_mean, marker='o', label=f'Protocol {protocol}')
    axes[1].plot(part.K, part.prd_repo_mean, marker='o', label=f'Protocol {protocol}')
axes[0].set(title='Cải thiện SNR theo K — QTDB/BW test', xlabel='K (số lần suy mẫu)', ylabel='ΔSNR (dB)')
axes[1].set(title='PRD_repo theo K — QTDB/BW test', xlabel='K (số lần suy mẫu)', ylabel='PRD_repo (%)')
for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'summary_by_K.png', dpi=160)
plt.show()

index = 0
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(CLEAN[index], label='QTDB clean', color='black')
ax.plot(NOISY[1][index], label='BW noisy (protocol 1)', alpha=0.65)
for k in (1, 10):
    estimate = np.load(OUT / f'xhat_protocol1_K{k}.npy', mmap_mode='r')[index]
    ax.plot(estimate, label=f'DDIM K={k}', alpha=0.85)
ax.set(xlabel='Mẫu tại 360 Hz', ylabel='mV', title=f'Mẫu {CLEAN_META.sample_id.iloc[index]} — protocol 1')
ax.legend(ncol=4)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / 'waveform_example.png', dpi=160)
plt.show()

run_info = {
    'source_clean': PHYSIONET_QT, 'source_noise': PHYSIONET_NST + '/bw',
    'test_records': list(selected_records), 'full_benchmark': MAX_TEST_RECORDS is None and MAX_TEST_SEGMENTS is None,
    'N_clean': len(CLEAN), 'N_per_protocol': len(CLEAN),
    'sampling_rate': TARGET_FS, 'segment_length': LENGTH,
    'preprocessing_version': PREPROCESSING_VERSION,
    'r_definition': 'peak-to-peak scaled BW / peak-to-peak clean; discrete 0.20..1.99',
    'noise_pair_seed': SEED, 'inference_seeds': {'1': SEED + 1, '2': SEED + 2},
    'K': list(K_VALUES), 'batch_size': BATCH_SIZE,
    'sampler': 'research DDPM.ddim_sample_loop', 'ddim_steps': DDIM_STEPS, 'eta': DDIM_ETA,
    'diffusion': CONFIG['diffusion'], 'model_base_features': 80,
    'model_embedding_dim': 128, 'checkpoint_info': CHECKPOINT_INFO,
    'training_configuration_from_research': CONFIG['train'],
    'training_history_verified_from_plain_checkpoint': False,
    'device': str(DEVICE), 'torch_version': torch.__version__,
    'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
    'wfdb_version': wfdb.__version__, 'python_version': platform.python_version(),
    'std_definition': 'population ddof=0 over finite per-segment values',
    'nonfinite_policy': 'keep all rows; aggregate finite metric values and report nonfinite_N per metric',
}
(OUT / 'run_info.json').write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved:', sorted(path.name for path in OUT.iterdir()))
"""),
    md("""## 9. Diễn giải khi hoàn tất chạy

- `summary.csv` có bảng riêng protocol 1, protocol 2, pooled và bốn khoảng r không giao nhau. `N` là số đoạn, không phải số bệnh nhân độc lập. `std` dùng `ddof=0`.
- `summary_by_record.csv` lưu thống kê riêng từng QT record để tránh nhầm độ phân tán đoạn với độ bất định giữa record.
- `prd_repo` dùng mean của **toàn batch sạch 50 đoạn** và mẫu số là tín hiệu đầu ra, đúng định nghĩa repo; `prd_standard` dùng ECG sạch riêng mỗi đoạn.
- `metrics_per_sample.csv` liên kết dự đoán với `output_file` và `output_row`; `manifest.csv` lưu nguồn BW và chỉ số bắt đầu patch.
- Kết quả chỉ áp dụng cho BW tổng hợp từ QTDB. Trước khi công bố như benchmark chính thức, đối chiếu provenance chọn checkpoint bằng validation; state dict thuần không chứng minh epoch/seed huấn luyện.
"""),
]

# The benchmark must train on the specified QTDB/BW split. Existing repository
# checkpoints have no auditable training provenance and are never used here.
cells[0] = md("""# QTDB BW benchmark — huấn luyện và đánh giá kiến trúc của nghiên cứu

Notebook Kaggle độc lập này dùng QTDB làm ECG sạch tạm thời, giữ **HNF U-Net + FiLM + attention + DDPM** và loss đa miền đúng mã `src` của nghiên cứu. Giao diện đổi sang một kênh vì benchmark QTDB có một kênh. Dữ liệu và cách chấm theo `rule.md`: 14 test record, P onset, 360 Hz, hai protocol BW, r đỉnh–đỉnh, K = 1/3/5/10, PRD_repo và PRD_standard.

**Notebook huấn luyện mới từ đầu**, riêng protocol 1 và 2, đủ 400 epoch mỗi protocol, chọn checkpoint bằng validation 70/30 của các record QTDB còn lại. Hai checkpoint cũ trong repo không được nạp. Test chỉ được mở sau khi cả hai lần huấn luyện hoàn tất. Bật Internet và GPU trong Kaggle. Checkpoint `last` được lưu sau **mỗi epoch**, checkpoint `best` được cập nhật khi validation tốt hơn. Nếu có checkpoint trước đó, notebook tự resume cả model, optimizer, scheduler và EMA.

Kết quả test và manifest được lưu trong `/kaggle/working/qtdb_rule_eval/`. Không gọi một phiên huấn luyện chưa đủ 400 epoch là benchmark hoàn chỉnh.
""")

preprocessing_source = ''.join(cells[11]['source'])
split_marker = 'selected_records = TEST_RECORDS[:MAX_TEST_RECORDS]'
preprocessing_functions, test_loading_source = preprocessing_source.split(split_marker, 1)
test_loading_source = split_marker + test_loading_source

noise_source = ''.join(cells[13]['source'])
noise_marker = 'def make_protocol_pairs(protocol):'
noise_loading_source, test_pairing_source = noise_source.split(noise_marker, 1)
test_pairing_source = noise_marker + test_pairing_source
noise_loading_source = """import shutil

def cached_input(filename):
    local = OUT / filename
    if local.is_file():
        return local
    root = Path('/kaggle/input')
    matches = list(root.rglob(filename)) if root.exists() else []
    if len(matches) > 1:
        raise RuntimeError(f'Multiple cached {filename} files; attach only the latest output version')
    if matches:
        shutil.copy2(matches[0], local)
        return local
    return None

bw_cache = cached_input('bw_360.npy')
if bw_cache is not None:
    BW = np.load(bw_cache)
    print('Loaded BW cache:', bw_cache)
else:
    noise_record = wfdb.rdrecord('bw', pn_dir=PHYSIONET_NST)
    BW = noise_record.p_signal.astype(np.float32)
    if int(noise_record.fs) != TARGET_FS:
        BW = signal.resample(BW, int(round(len(BW) * TARGET_FS / float(noise_record.fs))), axis=0).astype(np.float32)
    np.save(OUT / 'bw_360.npy', BW)
assert BW.ndim == 2 and BW.shape[1] >= 2 and np.isfinite(BW).all()
MIDPOINT = len(BW) // 2
NOISE_LEVELS = np.arange(20, 200, dtype=np.int16) / 100.0
"""

train_setup = code("""from torch.utils.data import DataLoader, TensorDataset

# Các tham số huấn luyện chính giữ đúng configs/base.yaml và src/training/utils.py.
CONFIG = {
    'train': {'feats': 80, 'batch_size': 96, 'epochs': 400, 'lr': 1e-4,
              'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
}
TRAIN_DIR = OUT / 'train'
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_RECORD_LIMIT = None  # None for the official run.

clean_cache = cached_input('train_clean_full.npy') if TRAIN_RECORD_LIMIT is None else None
meta_cache = cached_input('train_metadata_full.csv') if TRAIN_RECORD_LIMIT is None else None
audit_cache = cached_input('train_record_audit.csv') if TRAIN_RECORD_LIMIT is None else None
if any(path is not None for path in (clean_cache, meta_cache, audit_cache)) and not all(
        path is not None for path in (clean_cache, meta_cache, audit_cache)):
    raise RuntimeError('Incomplete QTDB training cache: need clean, metadata and record audit files')
if clean_cache is not None:
    TRAIN_CLEAN = np.load(clean_cache)
    TRAIN_META = pd.read_csv(meta_cache)
    TRAIN_AUDIT = pd.read_csv(audit_cache)
    TRAIN_RECORDS = tuple(TRAIN_AUDIT.record_id)
    print('Loaded QTDB training cache:', clean_cache)
else:
    records = wfdb.get_record_list('qtdb')
    TRAIN_RECORDS = tuple(sorted(set(records) - set(TEST_RECORDS)))
    if TRAIN_RECORD_LIMIT is not None:
        TRAIN_RECORDS = TRAIN_RECORDS[:TRAIN_RECORD_LIMIT]
    train_beats, train_rows, train_audit_rows = [], [], []
    for name in tqdm(TRAIN_RECORDS, desc='QTDB train/validation records'):
        beats, rows, audit = clean_segments_for_record(name)
        train_audit_rows.append(audit)
        if not beats:
            print(f'{name}: 0 eligible segments; see train_record_audit.csv')
            continue
        train_beats.extend(beats)
        train_rows.extend(rows)
    TRAIN_AUDIT = pd.DataFrame(train_audit_rows)
    TRAIN_AUDIT.to_csv(OUT / 'train_record_audit.csv', index=False)
    if not train_beats:
        raise RuntimeError('No valid QTDB training segments; inspect train_record_audit.csv')
    TRAIN_CLEAN = np.stack(train_beats).astype(np.float32)
    TRAIN_META = pd.DataFrame(train_rows)
    if TRAIN_RECORD_LIMIT is None:
        np.save(OUT / 'train_clean_full.npy', TRAIN_CLEAN)
        TRAIN_META.to_csv(OUT / 'train_metadata_full.csv', index=False)
assert TRAIN_RECORDS and not set(TRAIN_RECORDS) & set(TEST_RECORDS)
assert tuple(TRAIN_AUDIT.record_id) == TRAIN_RECORDS
assert TRAIN_AUDIT.accepted_segments.sum() == len(TRAIN_CLEAN)
assert len(TRAIN_CLEAN) == len(TRAIN_META)
assert TRAIN_CLEAN.ndim == 2 and TRAIN_CLEAN.shape[1] == LENGTH
assert np.isfinite(TRAIN_CLEAN).all() and TRAIN_META.sample_id.is_unique
assert not set(TRAIN_META.record_id) & set(TEST_RECORDS)

# Rule 3.1: split 70/30 of segments from non-test QTDB records; fixed seed.
indices = np.random.RandomState(SEED).permutation(len(TRAIN_CLEAN))
cut = int(0.70 * len(indices))
TRAIN_INDICES, VAL_INDICES = indices[:cut], indices[cut:]
assert len(TRAIN_INDICES) and len(VAL_INDICES)
pd.DataFrame({'sample_id': TRAIN_META.sample_id,
              'record_id': TRAIN_META.record_id,
              'split': np.where(np.isin(np.arange(len(TRAIN_CLEAN)), TRAIN_INDICES),
                                'train', 'validation')}).to_csv(OUT / 'train_validation_split.csv', index=False)
print('Records:', len(TRAIN_RECORDS), '| train:', len(TRAIN_INDICES), '| validation:', len(VAL_INDICES))
""")

train_pairing = code("""def fixed_noisy_pairs(clean_array, noise, seed):
    # The same sequential 512-sample BW patch rule as test, with fixed r per clean beat.
    patch_count = len(noise) // LENGTH
    if patch_count == 0:
        raise ValueError('BW training half is shorter than one patch')
    levels = np.random.RandomState(seed).randint(20, 200, len(clean_array)) / 100.0
    noisy_array = np.empty_like(clean_array)
    for i, (clean, r) in enumerate(zip(clean_array, levels)):
        patch = noise[(i % patch_count) * LENGTH:(i % patch_count + 1) * LENGTH]
        clean_range, noise_range = float(np.ptp(clean)), float(np.ptp(patch))
        if clean_range <= 0 or noise_range <= 0:
            raise ValueError(f'Zero amplitude range for training sample {i}')
        noisy_array[i] = clean + float(r) * clean_range / noise_range * patch
    return noisy_array

TRAIN_NOISY = {}
for protocol in (1, 2):
    train_channel = 0 if protocol == 1 else 1
    train_noise = BW[:MIDPOINT, train_channel]
    TRAIN_NOISY[protocol] = fixed_noisy_pairs(TRAIN_CLEAN, train_noise, SEED)
    assert np.isfinite(TRAIN_NOISY[protocol]).all()
assert BW[:MIDPOINT].shape[0] > LENGTH and BW[MIDPOINT:].shape[0] > LENGTH
print('Train noise: P1 first half channel 1; P2 first half channel 2')
""")

train_functions = code("""def training_loader(protocol, indices, shuffle, drop_last, epoch=0):
    clean = torch.from_numpy(TRAIN_CLEAN[indices, None, :])
    noisy = torch.from_numpy(TRAIN_NOISY[protocol][indices, None, :])
    generator = torch.Generator().manual_seed(SEED + protocol * 100_000 + epoch)
    return DataLoader(TensorDataset(clean, noisy), batch_size=CONFIG['train']['batch_size'],
                      shuffle=shuffle, drop_last=drop_last, generator=generator,
                      num_workers=0, pin_memory=DEVICE.type == 'cuda')

def checkpoint_candidates(protocol):
    name = f'last_protocol{protocol}.pth'
    candidates = [TRAIN_DIR / name]
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        candidates += list(kaggle_input.rglob(name))
    return [p for p in candidates if p.is_file()]

def restore_random_state(payload):
    if 'torch_rng_state' in payload:
        torch.set_rng_state(payload['torch_rng_state'])
    if DEVICE.type == 'cuda' and payload.get('cuda_rng_state') is not None:
        torch.cuda.set_rng_state(payload['cuda_rng_state'])

def save_state(model, optimizer, scheduler, ema, protocol, epoch, best_val, name):
    payload = {'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(), 'ema_state_dict': ema.state_dict(),
               'protocol': protocol,
               'epoch': epoch, 'best_validation_loss': best_val, 'config': CONFIG,
               'train_records': TRAIN_RECORDS, 'test_records': TEST_RECORDS,
               'torch_rng_state': torch.get_rng_state(),
               'cuda_rng_state': torch.cuda.get_rng_state() if DEVICE.type == 'cuda' else None}
    target = TRAIN_DIR / name
    temporary = target.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(target)

def validation_loss(model, loader, protocol):
    # Same validation noise draws each epoch; restore training RNG afterwards.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if DEVICE.type == 'cuda' else None
    torch.manual_seed(SEED + 10_000 + protocol)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(SEED + 10_000 + protocol)
    model.eval()
    weighted_sum, examples = 0.0, 0
    with torch.no_grad():
        for clean, noisy in loader:
            clean, noisy = clean.to(DEVICE), noisy.to(DEVICE)
            loss = model(clean, noisy)
            weighted_sum += float(loss) * len(clean)
            examples += len(clean)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    return weighted_sum / examples

def train_protocol(protocol):
    model = DDPM(UNet1D(2, CONFIG['train']['feats'], 128, 1), CONFIG, DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['train']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=1.0)
    ema = EMA(0.9)
    ema.register(model)
    val_loader = training_loader(protocol, VAL_INDICES, False, False)
    if len(TRAIN_INDICES) < CONFIG['train']['batch_size']:
        raise ValueError('Training set smaller than batch size 96')
    candidates = checkpoint_candidates(protocol)
    if candidates:
        payloads = [(torch.load(path, map_location='cpu', weights_only=False), path) for path in candidates]
        payload, path = max(payloads, key=lambda item: int(item[0]['epoch']))
        if path != TRAIN_DIR / f'last_protocol{protocol}.pth':
            shutil.copy2(path, TRAIN_DIR / f'last_protocol{protocol}.pth')
            for filename in (f'best_protocol{protocol}.pth', f'training_protocol{protocol}.csv'):
                previous = path.parent / filename
                if previous.is_file():
                    shutil.copy2(previous, TRAIN_DIR / filename)
        assert payload['protocol'] == protocol and payload['config'] == CONFIG
        assert tuple(payload['train_records']) == TRAIN_RECORDS
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        ema.load_state_dict({name: tensor.to(DEVICE)
                             for name, tensor in payload['ema_state_dict'].items()})
        restore_random_state(payload)
        start_epoch = int(payload['epoch']) + 1
        best_val = float(payload['best_validation_loss'])
        print(f'Protocol {protocol}: resume from {path}, epoch {start_epoch}')
    else:
        start_epoch, best_val = 0, float('inf')
        print(f'Protocol {protocol}: new training run')
    end_epoch = CONFIG['train']['epochs']
    log_path = TRAIN_DIR / f'training_protocol{protocol}.csv'
    completed_epoch = start_epoch
    for epoch in range(start_epoch, end_epoch):
        train_loader = training_loader(protocol, TRAIN_INDICES, True, True, epoch=epoch)
        model.train()
        total, count = 0.0, 0
        for clean, noisy in tqdm(train_loader, desc=f'P{protocol} epoch {epoch + 1}', leave=False):
            clean, noisy = clean.to(DEVICE), noisy.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = model(clean, noisy)  # src.models.main_model.DDPM.p_losses, unchanged.
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss, protocol {protocol}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            total += float(loss.detach())
            count += 1
        scheduler.step()
        val = validation_loss(model, val_loader, protocol)
        if not np.isfinite(val):
            raise FloatingPointError(f'Nonfinite validation loss, protocol {protocol}, epoch {epoch}')
        if val < best_val:
            best_val = val
            save_state(model, optimizer, scheduler, ema, protocol, epoch, best_val,
                       f'best_protocol{protocol}.pth')
        save_state(model, optimizer, scheduler, ema, protocol, epoch, best_val,
                   f'last_protocol{protocol}.pth')
        pd.DataFrame([{'protocol': protocol, 'epoch': epoch + 1,
                       'train_loss': total / count, 'validation_loss': val,
                       'best_validation_loss': best_val}]).to_csv(
                           log_path, mode='a', header=not log_path.exists(), index=False)
        completed_epoch = epoch + 1
        print(f'P{protocol} epoch {epoch + 1}/{CONFIG["train"]["epochs"]}: '
              f'train={total/count:.6f}, val={val:.6f}, best={best_val:.6f}')
    return completed_epoch
""")

train_run = code("""COMPLETED_EPOCHS = {}
for protocol in (1, 2):
    COMPLETED_EPOCHS[protocol] = train_protocol(protocol)
    print(f'Protocol {protocol}: completed {COMPLETED_EPOCHS[protocol]}/{CONFIG["train"]["epochs"]} epochs')

# Test is opened only after both training protocols finish.
READY_FOR_TEST = (TRAIN_RECORD_LIMIT is None and
                  all(COMPLETED_EPOCHS[p] >= CONFIG['train']['epochs'] for p in (1, 2)))
if READY_FOR_TEST:
    print('Both protocols reached 400 epochs; validation-selected checkpoints are locked.')
else:
    print('Training is incomplete; test remains unopened.')
""")

load_best = code("""def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

MODELS, CHECKPOINT_INFO = {}, []
for protocol in (1, 2):
    path = TRAIN_DIR / f'best_protocol{protocol}.pth'
    if not path.is_file():
        matches = list(Path('/kaggle/input').rglob(path.name)) if Path('/kaggle/input').exists() else []
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one validation-best checkpoint: {path.name}')
        path = matches[0]
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['protocol'] == protocol and payload['config'] == CONFIG
    assert tuple(payload['train_records']) == TRAIN_RECORDS
    model = DDPM(UNet1D(2, 80, 128, 1), CONFIG, DEVICE).to(DEVICE)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    MODELS[protocol] = model
    info = {'protocol': protocol, 'path': str(path), 'sha256': sha256(path),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'checkpoint_epoch': int(payload['epoch']) + 1,
            'checkpoint_validation': float(payload['best_validation_loss'])}
    CHECKPOINT_INFO.append(info)
    print(f'Protocol {protocol}: selected epoch {info["checkpoint_epoch"]}; '
          f'val loss={info["checkpoint_validation"]:.6f}; strict weights OK')
display(pd.DataFrame(CHECKPOINT_INFO))
""")

cells = [
    cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], cells[6], cells[7],
    md('## 3. Hàm chuẩn bị ECG sạch theo benchmark'), code(preprocessing_functions),
    md('## 4. Tải BW và khóa nửa thời gian cho hai protocol'), code(noise_loading_source),
    md('## 5. QTDB train/validation và cặp BW nửa đầu'), train_setup, train_pairing,
    md('## 6. Huấn luyện riêng hai protocol, chọn best bằng validation'), train_functions, train_run,
    md('## 7. Nạp checkpoint best sau khi đủ 400 epoch'), load_best,
    md('## 8. Mở 14 QTDB test records'), code(test_loading_source),
    md('## 9. Tạo cặp BW test từ nửa sau và manifest'), code(test_pairing_source),
    cells[14], cells[15], cells[16], cells[17], cells[18], cells[19], cells[20],
]

# The last metadata cell now records verified training provenance.
metadata_cell = ''.join(cells[-2]['source'])
metadata_cell = metadata_cell.replace(
    "'training_history_verified_from_plain_checkpoint': False,",
    "'training_history_verified_from_checkpoint': True,\n"
    "    'train_records': list(TRAIN_RECORDS),\n"
    "    'train_N': len(TRAIN_INDICES), 'validation_N': len(VAL_INDICES),"
)
metadata_cell = metadata_cell.replace(
    "'full_benchmark': MAX_TEST_RECORDS is None and MAX_TEST_SEGMENTS is None,",
    "'full_benchmark': TRAIN_RECORD_LIMIT is None and MAX_TEST_RECORDS is None and MAX_TEST_SEGMENTS is None,"
)
cells[-2] = code(metadata_cell)
for cell_index, number in ((24, 10), (26, 11), (28, 12), (30, 13)):
    heading = ''.join(cells[cell_index]['source'])
    old_number = heading.split(' ', 2)[1]
    heading = heading.replace(old_number, f'{number}.', 1)
    cells[cell_index] = md(heading)
takeaways = ''.join(cells[-1]['source']).replace(
    'Trước khi công bố như benchmark chính thức, đối chiếu provenance chọn checkpoint bằng validation; state dict thuần không chứng minh epoch/seed huấn luyện.',
    'Checkpoint được huấn luyện trong notebook và chọn bằng validation; xem `train/training_protocol*.csv` cùng `run_info.json` để kiểm toán epoch, seed và cấu hình.'
)
takeaways += '\nNếu `READY_FOR_TEST=False`, không có bảng test vì huấn luyện chưa đủ điều kiện đánh giá.\n'
cells[-1] = md(takeaways)
for cell_index in (19, 21, 23, 25, 27, 29):
    body = ''.join(cells[cell_index]['source'])
    cells[cell_index] = code('if READY_FOR_TEST:\n' + ''.join('    ' + line for line in body.splitlines(True)))

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUT)
