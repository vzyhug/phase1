"""Build a Kaggle notebook for BW/EM/MA cross-noise generalization.

The validated Colab evaluator remains the source of truth for the network,
diffusion sampler, QTDB segmentation, and clean/noise construction.  This
builder replaces only the orchestration needed to evaluate every specialist
model on every test artifact (3 x 3), while matching protocol 1 with protocol
1 and protocol 2 with protocol 2.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "colab_evaluate_qtdb_bw_em_ma.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_cross_noise_generalization.ipynb"


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


CONFIG = r'''import hashlib
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
MODEL_ARTIFACTS = ('BW', 'EM', 'MA')
TEST_ARTIFACTS = ('BW', 'EM', 'MA')
PROTOCOLS = (1, 2)
# Start with 1-shot. Add 3, 5, 10 only after the full 3x3 smoke test passes.
K_VALUES = (1,)
BATCH_SIZE = 50
DDIM_STEPS = 50
DDIM_ETA = 0.0

# Attach one Kaggle dataset containing the six protocol checkpoints.
CHECKPOINT_ROOT = Path('/kaggle/input')
OUT = Path('/kaggle/working/qtdb_cross_noise_generalization')
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
    raise RuntimeError('Enable GPU in Kaggle: Settings -> Accelerator -> GPU.')

# Use small values only for a smoke test. Keep None for the official run.
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

if not set(MODEL_ARTIFACTS).issubset(VALID_ARTIFACTS):
    raise ValueError('MODEL_ARTIFACTS contains an unknown artifact')
if not set(TEST_ARTIFACTS).issubset(VALID_ARTIFACTS):
    raise ValueError('TEST_ARTIFACTS contains an unknown artifact')
if not set(PROTOCOLS).issubset({1, 2}):
    raise ValueError('PROTOCOLS must contain only 1 and/or 2')

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print('GPU:', torch.cuda.get_device_name(0))
print('Checkpoint root:', CHECKPOINT_ROOT, '| output:', OUT)
'''


LOAD_MODELS = r'''def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


MODELS, CHECKPOINT_INFO = {}, []
for model_artifact in MODEL_ARTIFACTS:
    for protocol in PROTOCOLS:
        filename = f'best_{model_artifact.lower()}_protocol{protocol}_{CHECKPOINT_TAG}.pth'
        matches = sorted(CHECKPOINT_ROOT.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(f'Need exactly one {filename}; found {matches}')
        path = matches[0]
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('artifact') != model_artifact or int(payload.get('protocol', -1)) != protocol:
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
            probe_output = model.model(probe, probe, torch.ones(1, 1, device=DEVICE))
        assert probe_output.shape == probe.shape and torch.isfinite(probe_output).all()
        MODELS[(model_artifact, protocol)] = model
        CHECKPOINT_INFO.append({
            'model_artifact': model_artifact,
            'protocol': protocol,
            'path': str(path),
            'sha256': sha256(path),
            'selected_epoch': int(payload['epoch']) + 1,
            'best_validation_loss': float(payload['best_validation_loss']),
            'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        })

checkpoint_manifest = pd.DataFrame(CHECKPOINT_INFO)
checkpoint_manifest.to_csv(OUT / 'checkpoint_manifest.csv', index=False)
display(checkpoint_manifest)
'''


MAKE_TEST_PAIRS = r'''def make_test_pairs(test_artifact, protocol):
    # P1 trains on first-half channel 1 and tests on second-half channel 2.
    # P2 trains on first-half channel 2 and tests on second-half channel 1.
    channel = 1 if protocol == 1 else 0
    midpoint = MIDPOINT[test_artifact]
    noise = NOISE[test_artifact][midpoint:, channel]
    patch_count = len(noise) // LENGTH
    if patch_count == 0:
        raise RuntimeError(f'{test_artifact} test half is shorter than one patch')
    # Reuse the same r vector for all artifacts/protocols for paired comparisons.
    rng = np.random.RandomState(SEED)
    levels = rng.randint(20, 200, size=len(CLEAN)).astype(np.float32) / 100.0
    noisy = np.empty_like(CLEAN)
    rows = []
    for index, (clean, level) in enumerate(zip(CLEAN, levels)):
        relative_start = (index % patch_count) * LENGTH
        patch = noise[relative_start:relative_start + LENGTH]
        clean_range, noise_range = float(np.ptp(clean)), float(np.ptp(patch))
        if clean_range <= 0 or noise_range <= 0:
            raise ValueError(f'Zero range: {test_artifact}/P{protocol}/sample{index}')
        noisy[index] = clean + float(level) * clean_range / noise_range * patch
        row = CLEAN_META.iloc[index].to_dict()
        row.update({
            'test_artifact': test_artifact,
            'protocol': protocol,
            'split': 'test',
            'noise_source_id': f'nstdb/1.0.0/{VALID_ARTIFACTS[test_artifact]}:second_half',
            'noise_channel': channel + 1,
            'noise_start': midpoint + relative_start,
            'r': float(level),
            'seed': SEED,
            'sampling_rate': TARGET_FS,
            'preprocessing_version': PREPROCESSING_VERSION,
        })
        rows.append(row)
    return noisy, pd.DataFrame(rows)


NOISY, MANIFEST = {}, {}
for test_artifact in TEST_ARTIFACTS:
    for protocol in PROTOCOLS:
        key = (test_artifact, protocol)
        NOISY[key], MANIFEST[key] = make_test_pairs(test_artifact, protocol)

manifest = pd.concat(MANIFEST.values(), ignore_index=True)
assert len(manifest) == len(CLEAN) * len(TEST_ARTIFACTS) * len(PROTOCOLS)
assert not manifest.duplicated(['test_artifact', 'protocol', 'sample_id']).any()
manifest.to_csv(OUT / 'test_pair_manifest.csv', index=False)
display(manifest.groupby(['test_artifact', 'protocol', 'noise_channel']).size().rename('N'))
'''


INFERENCE = r'''def synchronize():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def compute_metric_frame(clean, noisy, prediction, metadata, model_artifact,
                         test_artifact, protocol, k, elapsed_ms, output_start,
                         checkpoint_hash):
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
    frame = metadata[['sample_id', 'patient_id', 'record_id', 'test_artifact', 'r']].copy()
    frame.insert(3, 'model_artifact', model_artifact)
    frame['in_domain'] = model_artifact == test_artifact
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


checkpoint_hashes = {(row['model_artifact'], int(row['protocol'])): row['sha256']
                     for row in CHECKPOINT_INFO}
result_frames = []
test_seed_offset = {'BW': 11, 'EM': 22, 'MA': 33}

for model_artifact in MODEL_ARTIFACTS:
    for test_artifact in TEST_ARTIFACTS:
        for protocol in PROTOCOLS:
            model_key = (model_artifact, protocol)
            test_key = (test_artifact, protocol)
            model = MODELS[model_key]
            prediction_maps = {
                k: np.lib.format.open_memmap(
                    OUT / (f'xhat_model-{model_artifact.lower()}_test-{test_artifact.lower()}'
                           f'_protocol{protocol}_K{k}.npy'),
                    mode='w+', dtype=np.float32, shape=CLEAN.shape,
                ) for k in K_VALUES
            }
            # Common random numbers: for a fixed test artifact/protocol every model
            # starts from the same diffusion RNG state.
            inference_seed = SEED + test_seed_offset[test_artifact] * 100 + protocol
            torch.manual_seed(inference_seed)
            torch.cuda.manual_seed_all(inference_seed)
            description = f'model={model_artifact}, test={test_artifact}, P{protocol}'
            for start in tqdm(range(0, len(CLEAN), BATCH_SIZE), desc=description):
                stop = min(start + BATCH_SIZE, len(CLEAN))
                scale = np.maximum(
                    np.max(np.abs(CLEAN[start:stop]), axis=1, keepdims=True), 1e-8
                ).astype(np.float32)
                condition = torch.from_numpy(
                    (NOISY[test_key][start:stop] / scale)[:, None, :]
                ).to(DEVICE)
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
                            raise RuntimeError(f'Invalid output: {description}, shot={shot}')
                        running_sum += output
                        if shot in K_VALUES:
                            synchronize()
                            elapsed_ms = (time.perf_counter() - tick) * 1000
                            prediction = (running_sum / shot).cpu().numpy()[:, 0, :] * scale
                            prediction_maps[shot][start:stop] = prediction
                            result_frames.append(compute_metric_frame(
                                CLEAN[start:stop], NOISY[test_key][start:stop], prediction,
                                MANIFEST[test_key].iloc[start:stop].reset_index(drop=True),
                                model_artifact, test_artifact, protocol, shot,
                                elapsed_ms, start, checkpoint_hashes[model_key],
                            ))
            for values in prediction_maps.values():
                values.flush()

per_sample = pd.concat(result_frames, ignore_index=True)
expected_rows = (len(CLEAN) * len(MODEL_ARTIFACTS) * len(TEST_ARTIFACTS)
                 * len(PROTOCOLS) * len(K_VALUES))
assert len(per_sample) == expected_rows
assert not per_sample.duplicated(
    ['model_artifact', 'test_artifact', 'protocol', 'K', 'sample_id']
).any()
per_sample.to_csv(OUT / 'metrics_per_sample.csv', index=False)
print('Metric rows:', len(per_sample), '| expected:', expected_rows)
'''


SUMMARY = r'''METRICS = [
    'ssd', 'rmse', 'mae', 'mad', 'prd_repo', 'prd_standard',
    'cosine_similarity', 'snr_in_db', 'snr_out_db',
    'snr_improvement_db', 'inference_time_ms',
]
per_sample['r_group'] = pd.cut(
    per_sample.r,
    bins=[0.2, 0.6, 1.0, 1.5, 2.001],
    right=False,
    labels=['0.20-<0.60', '0.60-<1.00', '1.00-<1.50', '1.50-2.00'],
)
assert per_sample.r_group.notna().all()


def metric_stats(frame, prefix=''):
    result = {}
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[f'{prefix}{metric}_valid_N'] = len(finite)
        result[f'{prefix}{metric}_nonfinite_N'] = len(values) - len(finite)
        result[f'{prefix}{metric}_mean'] = float(np.mean(finite)) if len(finite) else np.nan
        result[f'{prefix}{metric}_std'] = float(np.std(finite, ddof=0)) if len(finite) else np.nan
    return result


summary_rows = []
group_columns = ['model_artifact', 'test_artifact', 'protocol', 'K']
for keys, frame in per_sample.groupby(group_columns):
    model_artifact, test_artifact, protocol, k = keys
    base = {
        'model_artifact': model_artifact,
        'test_artifact': test_artifact,
        'protocol': protocol,
        'K': k,
        'r_group': 'ALL',
        'N': len(frame),
        'in_domain': model_artifact == test_artifact,
    }
    summary_rows.append({**base, **metric_stats(frame)})
    for r_group, part in frame.groupby('r_group', observed=True):
        summary_rows.append({
            **base, 'r_group': str(r_group), 'N': len(part), **metric_stats(part)
        })

# Protocol-pooled rows retain equal protocol weighting because both protocols have
# exactly the same QTDB samples.
for keys, frame in per_sample.groupby(['model_artifact', 'test_artifact', 'K']):
    model_artifact, test_artifact, k = keys
    summary_rows.append({
        'model_artifact': model_artifact,
        'test_artifact': test_artifact,
        'protocol': 'pooled',
        'K': k,
        'r_group': 'ALL',
        'N': len(frame),
        'in_domain': model_artifact == test_artifact,
        **metric_stats(frame),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / 'summary.csv', index=False)

record_rows = []
for keys, frame in per_sample.groupby(
        ['model_artifact', 'test_artifact', 'protocol', 'K', 'record_id']):
    model_artifact, test_artifact, protocol, k, record_id = keys
    record_rows.append({
        'model_artifact': model_artifact,
        'test_artifact': test_artifact,
        'protocol': protocol,
        'K': k,
        'record_id': record_id,
        'N': len(frame),
        **metric_stats(frame),
    })
pd.DataFrame(record_rows).to_csv(OUT / 'summary_by_record.csv', index=False)

DISPLAY_K = max(K_VALUES)
display_frame = per_sample[per_sample.K == DISPLAY_K].copy()

# Exact 3x3 matrices, pooled over the two protocols. These are the primary
# generalization results; rows are training domains and columns are test domains.
matrix_rows = []
for (model_artifact, test_artifact), frame in display_frame.groupby(
        ['model_artifact', 'test_artifact']):
    row = {
        'model_artifact': model_artifact,
        'test_artifact': test_artifact,
        'K': DISPLAY_K,
        'N': len(frame),
        'in_domain': model_artifact == test_artifact,
    }
    for metric in METRICS:
        values = frame[metric].replace([np.inf, -np.inf], np.nan).dropna()
        row[f'{metric}_mean'] = values.mean()
        row[f'{metric}_std'] = values.std(ddof=0)
    matrix_rows.append(row)
cross_noise_matrix = pd.DataFrame(matrix_rows)
cross_noise_matrix.to_csv(OUT / 'cross_noise_matrix.csv', index=False)
display(cross_noise_matrix)

# Model-level OOD report. OOD macro first averages within each unseen artifact,
# then averages the two artifact means so every artifact has equal weight.
generalization_rows = []
for model_artifact in MODEL_ARTIFACTS:
    model_frame = display_frame[display_frame.model_artifact == model_artifact]
    in_domain = model_frame[model_frame.test_artifact == model_artifact]
    ood = model_frame[model_frame.test_artifact != model_artifact]
    ood_by_artifact = ood.groupby('test_artifact')[
        ['snr_improvement_db', 'prd_standard', 'cosine_similarity']
    ].mean()
    row = {
        'model_artifact': model_artifact,
        'K': DISPLAY_K,
        'in_domain_N': len(in_domain),
        'ood_N': len(ood),
        'snr_improvement_in_domain_db': in_domain.snr_improvement_db.mean(),
        'snr_improvement_ood_macro_db': ood_by_artifact.snr_improvement_db.mean(),
        'snr_improvement_ood_worst_db': ood_by_artifact.snr_improvement_db.min(),
        'prd_standard_in_domain_pct': in_domain.prd_standard.mean(),
        'prd_standard_ood_macro_pct': ood_by_artifact.prd_standard.mean(),
        'prd_standard_ood_worst_pct': ood_by_artifact.prd_standard.max(),
        'cosine_in_domain': in_domain.cosine_similarity.mean(),
        'cosine_ood_macro': ood_by_artifact.cosine_similarity.mean(),
        'cosine_ood_worst': ood_by_artifact.cosine_similarity.min(),
    }
    row['snr_generalization_drop_db'] = (
        row['snr_improvement_in_domain_db'] - row['snr_improvement_ood_macro_db']
    )
    row['prd_generalization_increase_pct_point'] = (
        row['prd_standard_ood_macro_pct'] - row['prd_standard_in_domain_pct']
    )
    row['cosine_generalization_drop'] = row['cosine_in_domain'] - row['cosine_ood_macro']
    for test_artifact in TEST_ARTIFACTS:
        part = model_frame[model_frame.test_artifact == test_artifact]
        row[f'snr_improvement_test_{test_artifact}_db'] = part.snr_improvement_db.mean()
    generalization_rows.append(row)

generalization = pd.DataFrame(generalization_rows)
generalization.to_csv(OUT / 'generalization_by_model.csv', index=False)
display(generalization)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#4A4A4A',
    'axes.labelcolor': '#222222',
    'axes.titleweight': 'bold',
    'axes.grid': False,
    'font.size': 10,
})
ORDER = list(MODEL_ARTIFACTS)


def plot_matrix(metric, title, unit, cmap, filename):
    matrix = (cross_noise_matrix.pivot(
        index='model_artifact', columns='test_artifact', values=f'{metric}_mean'
    ).reindex(index=ORDER, columns=list(TEST_ARTIFACTS)))
    values = matrix.to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(6.8, 5.5))
    image = axis.imshow(values, cmap=cmap, aspect='equal')
    axis.set(
        xticks=np.arange(len(TEST_ARTIFACTS)),
        xticklabels=TEST_ARTIFACTS,
        yticks=np.arange(len(ORDER)),
        yticklabels=ORDER,
        xlabel='Test noise',
        ylabel='Model training noise',
        title=f'{title} (protocols pooled, K={DISPLAY_K})',
    )
    axis.grid(False)
    threshold = np.nanmedian(values)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            color = 'white' if value < threshold else '#111111'
            axis.text(column_index, row_index, f'{value:.3f}', ha='center',
                      va='center', color=color, fontweight='bold')
    fig.colorbar(image, ax=axis, label=unit)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)


plot_matrix('snr_improvement_db', 'Cross-noise SNR improvement', 'dB',
            'cividis', 'cross_noise_snr_improvement_heatmap.png')
plot_matrix('prd_standard', 'Cross-noise standard PRD', '%',
            'magma_r', 'cross_noise_prd_heatmap.png')
plot_matrix('cosine_similarity', 'Cross-noise cosine similarity', 'score',
            'viridis', 'cross_noise_cosine_heatmap.png')

# In-domain vs OOD macro comparison. Exact gaps remain in the CSV.
positions = np.arange(len(generalization))
width = 0.36
fig, axis = plt.subplots(figsize=(8.2, 5.2))
axis.bar(positions - width / 2, generalization.snr_improvement_in_domain_db,
         width, color='#4C78A8', label='In-domain')
axis.bar(positions + width / 2, generalization.snr_improvement_ood_macro_db,
         width, color='#F2CF5B', edgecolor='#333333', label='OOD macro')
axis.axhline(0, color='#333333', linewidth=1)
axis.set(
    xticks=positions,
    xticklabels=generalization.model_artifact,
    xlabel='Model training noise',
    ylabel='Mean SNR improvement (dB)',
    title=f'In-domain vs unseen-noise generalization (K={DISPLAY_K})',
)
axis.grid(axis='y', alpha=0.22)
axis.legend()
fig.tight_layout()
fig.savefig(OUT / 'in_domain_vs_ood_snr.png', dpi=180, bbox_inches='tight')
plt.show()
plt.close(fig)

run_info = {
    'model_artifacts': list(MODEL_ARTIFACTS),
    'test_artifacts': list(TEST_ARTIFACTS),
    'protocols': list(PROTOCOLS),
    'test_records': list(selected_records),
    'N_clean': len(CLEAN),
    'sampling_rate': TARGET_FS,
    'segment_length': LENGTH,
    'preprocessing_version': PREPROCESSING_VERSION,
    'model_normalization': MODEL_NORMALIZATION,
    'K': list(K_VALUES),
    'batch_size': BATCH_SIZE,
    'ddim_steps': DDIM_STEPS,
    'ddim_eta': DDIM_ETA,
    'common_random_numbers_across_models': True,
    'checkpoint_info': CHECKPOINT_INFO,
    'device': str(DEVICE),
    'torch_version': torch.__version__,
    'numpy_version': np.__version__,
    'scipy_version': scipy.__version__,
    'wfdb_version': wfdb.__version__,
    'python_version': platform.python_version(),
    'full_evaluation': MAX_TEST_RECORDS is None and MAX_TEST_SEGMENTS is None,
}
(OUT / 'run_info.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')

archive_path = OUT.parent / 'qtdb_cross_noise_generalization.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(OUT))
print('Saved:', OUT, '| archive:', archive_path)
'''


TITLE = """# Cross-noise generalization — QTDB × NSTDB BW/EM/MA

Notebook đánh giá chéo toàn bộ ma trận **model huấn luyện theo BW/EM/MA × dữ liệu kiểm tra BW/EM/MA**. Hai protocol được giữ riêng và ghép đúng protocol giữa model và test để bảo toàn quy tắc tách kênh/nửa thời gian của noise.

Mục tiêu chính là đo hiệu năng đúng miền, hiệu năng trên hai loại nhiễu chưa gặp, mức suy giảm OOD và trường hợp OOD xấu nhất. Mọi model dùng cùng QTDB segments, cùng noisy input, cùng mức `r`, và cùng trạng thái RNG cho từng test noise/protocol.

**Kaggle:** bật GPU + Internet, dùng **Add Input** để gắn dataset chứa sáu file `best_<bw|em|ma>_protocol<1|2>_norm_loss_v2.pth`, sau đó chạy **Run all**. Mặc định là full test, DDIM 50 bước, `eta=0`, 1-shot. Nếu hiện chỉ có đúng ba checkpoint của một protocol, đổi `PROTOCOLS = (1,)` hoặc `(2,)` cho phù hợp.
"""


OUTPUT = """## Output chính

- `metrics_per_sample.csv`: metric từng segment cho toàn bộ 3×3×2 protocol;
- `summary.csv`: mean/SD theo model, test noise, protocol, K và mức `r`;
- `cross_noise_matrix.csv`: ma trận 3×3 gộp hai protocol;
- `generalization_by_model.csv`: in-domain, OOD macro, OOD worst-case và generalization gap;
- `summary_by_record.csv`: kiểm tra độ ổn định theo QTDB record;
- `checkpoint_manifest.csv`, `test_pair_manifest.csv`, `run_info.json`: provenance;
- ba heatmap SNR/PRD/cosine và biểu đồ in-domain so với OOD;
- `qtdb_cross_noise_generalization.zip`: toàn bộ kết quả.

Giá trị gap dương trong `snr_generalization_drop_db`, `prd_generalization_increase_pct_point` hoặc `cosine_generalization_drop` biểu thị hiệu năng xấu đi khi chuyển sang nhiễu ngoài miền.
"""


def build() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    output = copy.deepcopy(source)
    cells = output["cells"]
    if len(cells) != 22:
        raise RuntimeError(f"Unexpected source notebook structure: {len(cells)} cells")

    set_source(cells[0], TITLE)
    set_source(cells[1], "## 1. Cài thư viện\n")
    install = "".join(cells[2]["source"]).replace("!pip", "%pip")
    set_source(cells[2], install)
    set_source(cells[3], "## 2. Cấu hình ma trận đánh giá chéo\n")
    set_source(cells[4], CONFIG)
    set_source(cells[9], "## 5. Nạp và kiểm tra checkpoint EMA theo miền train\n")
    set_source(cells[10], LOAD_MODELS)
    set_source(cells[13], "## 7. Tạo QTDB test set và tải BW/EM/MA\n")
    set_source(cells[15], "## 8. Tạo noisy input dùng chung cho mọi model\n")
    set_source(cells[16], MAKE_TEST_PAIRS)
    set_source(cells[17], "## 9. DDIM inference cho ma trận model × test noise\n")
    set_source(cells[18], INFERENCE)
    set_source(cells[19], "## 10. Ma trận tổng quát hóa, OOD gap và biểu đồ\n")
    set_source(cells[20], SUMMARY)
    set_source(cells[21], OUTPUT)

    output["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    TARGET.write_text(
        json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    build()
