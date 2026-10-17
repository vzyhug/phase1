"""Build the Kaggle zero-shot RMN benchmark for QTDB BW/EM/MA checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "notebooks" / "colab_evaluate_qtdb_bw_em_ma.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_qtdb_zero_shot_rmn_benchmark.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": source.splitlines(True),
    }


def source_after_heading(notebook: dict, heading: str) -> str:
    cells = notebook["cells"]
    for index, cell in enumerate(cells[:-1]):
        if cell.get("cell_type") == "markdown" and heading in "".join(cell.get("source", [])):
            return "".join(cells[index + 1].get("source", []))
    raise RuntimeError(f"Missing model source heading: {heading}")


model_notebook = json.loads(MODEL_SOURCE.read_text(encoding="utf-8"))
ARCHITECTURE = source_after_heading(model_notebook, "Kiến trúc HNF U-Net")
DDPM = source_after_heading(model_notebook, "Conditional DDPM/DDIM")


INSTALL = "!pip -q install 'wfdb>=4.1,<5' 'PyWavelets>=1.5' 'prettytable>=3.9'\n"


SETUP = r'''from pathlib import Path
import gc
import hashlib
import json
import pickle
import platform
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
try:
    from IPython.display import display
except ImportError:
    display = print

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run one artifact per Kaggle version to stay within the execution limit.
MODEL_ARTIFACTS_TO_EVALUATE = ('BW',)
NOISE_VERSIONS = (1, 2)  # verified mapping: P1=nv1, P2=nv2
RUN_INFERENCE = True
RUN_COMPARISON = True
REQUIRE_ALL_ZERO_SHOT_MODELS = False

# None means the official full 15,535-segment test set. Use 128 only for smoke tests.
MAX_TEST_SEGMENTS = None
INFERENCE_BATCH_SIZE = 50
DDIM_STEPS = 50
DDIM_ETA = 0.0
K = 1
SAVE_PREDICTIONS = False

if RUN_INFERENCE and DEVICE.type != 'cuda':
    raise RuntimeError('Enable GPU in Kaggle before zero-shot diffusion inference.')

INPUT_ROOT = Path('/kaggle/input')
TEMP_ROOT = Path('/kaggle/temp/qtdb_zero_shot_rmn')
DATA_DIR = TEMP_ROOT / 'data' / '4_RMN'
REPO_DIR = TEMP_ROOT / 'fully_gated_DAE'
OUT = Path('/kaggle/working/qtdb_zero_shot_rmn')
METRIC_DIR = OUT / 'metrics'
REPORT_DIR = OUT / 'reports'
PREDICTION_DIR = OUT / 'predictions'
for directory in (DATA_DIR, METRIC_DIR, REPORT_DIR, PREDICTION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TARGET_FS = 360
LENGTH = 512
PREPROCESSING_VERSION = 'qtdb_pu1_p_and_beat_40ms_reflect100ms_360hz_endpoints_pad16_v2'
MODEL_NORMALIZATION = 'per_segment_clean_maxabs_shared_with_noisy_v1'
LOSS_VERSION = 'l1_mean_relative_stft_snr_weighted_continuous_alpha_v1'
CHECKPOINT_TAG = 'norm_loss_v2'
CONFIG = {
    'train': {'feats': 80, 'batch_size': 96, 'epochs': 100, 'lr': 1e-4,
              'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
    'model_normalization': MODEL_NORMALIZATION,
    'loss_version': LOSS_VERSION,
    'preprocessing_version': PREPROCESSING_VERSION,
}

FGDAE_REPO_URL = 'https://github.com/AhmedAShaheen/fully_gated_DAE.git'
FGDAE_REPO_COMMIT = '70e768e'
VALID_ARTIFACTS = ('BW', 'EM', 'MA')
if not set(MODEL_ARTIFACTS_TO_EVALUATE).issubset(VALID_ARTIFACTS):
    raise ValueError(MODEL_ARTIFACTS_TO_EVALUATE)
if set(NOISE_VERSIONS) - {1, 2}:
    raise ValueError(NOISE_VERSIONS)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print('Device:', torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU (aggregation only)')
print('Model artifacts:', MODEL_ARTIFACTS_TO_EVALUATE, '| RMN versions:', NOISE_VERSIONS)
'''


DATA_PREP = r'''def attached_dataset_directory():
    parents = sorted({path.parent for path in INPUT_ROOT.rglob('Dataset_1.pkl')
                      if (path.parent / 'Dataset_2.pkl').is_file()})
    if not parents:
        return None
    # Identical duplicate pairs are still ambiguous at the path level; keep one input only.
    if len(parents) != 1:
        raise RuntimeError('Multiple Dataset_1/2 directories found; keep one: ' + str(parents))
    return parents[0]


def clone_repository():
    if not REPO_DIR.exists():
        subprocess.run(['git', 'clone', FGDAE_REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(['git', 'checkout', FGDAE_REPO_COMMIT], cwd=REPO_DIR, check=True)
    prep = REPO_DIR / 'Data_Preparation' / 'data_preparation_overlapped.py'
    lines = prep.read_text(encoding='utf-8').splitlines()
    patched = []
    for line in lines:
        if 'test_name = re.search' in line:
            indent = line[:len(line) - len(line.lstrip())]
            line = indent + "test_name = str(signal_name).replace('\\\\', '/').split('/')[-1]"
        patched.append(line)
    prep.write_text('\n'.join(patched) + '\n', encoding='utf-8')


def download_zip(url, destination, attempts=5):
    if destination.is_file() and zipfile.is_zipfile(destination):
        return
    partial = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 ECG-benchmark/1.0'})
            with urllib.request.urlopen(request, timeout=180) as response, open(partial, 'wb') as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            with zipfile.ZipFile(partial) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile('Archive integrity test failed')
            partial.replace(destination)
            return
        except Exception as error:
            if partial.exists():
                partial.unlink()
            if attempt == attempts:
                raise RuntimeError(
                    f'Cannot download {url}. Attach Dataset_1.pkl and Dataset_2.pkl as Kaggle Input.'
                ) from error
            time.sleep(min(30, 3 * 2 ** (attempt - 1)))


def ensure_datasets():
    source = attached_dataset_directory()
    if source is not None:
        for version in (1, 2):
            shutil.copy2(source / f'Dataset_{version}.pkl', DATA_DIR / f'Dataset_{version}.pkl')
        print('Reused RMN datasets from:', source)
        return
    print('No attached RMN datasets; generating the official QTDB RMN datasets.')
    clone_repository()
    repo_data = REPO_DIR / 'data'
    repo_output = repo_data / '4_RMN'
    repo_output.mkdir(parents=True, exist_ok=True)
    downloads = {
        'qt-database-1.0.0.zip': 'https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip',
        'mit-bih-noise-stress-test-database-1.0.0.zip':
            'https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip',
    }
    for name, url in downloads.items():
        archive_path = REPO_DIR / name
        download_zip(url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(repo_data)
    subprocess.run([sys.executable, 'generate_data.py', '--Data', 'RMN1'],
                   cwd=REPO_DIR, check=True)
    for version in (1, 2):
        shutil.copy2(repo_output / f'Dataset_{version}.pkl', DATA_DIR / f'Dataset_{version}.pkl')


if RUN_INFERENCE:
    ensure_datasets()
    DATASET_PATHS = {version: DATA_DIR / f'Dataset_{version}.pkl' for version in (1, 2)}
else:
    DATASET_PATHS = {}
    print('Aggregation-only mode: RMN datasets are not required.')
for version, path in DATASET_PATHS.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f'nv{version}: {path.stat().st_size / 1024**2:.1f} MiB')
'''


DATA_AUDIT = r'''def load_dataset(version):
    with open(DATASET_PATHS[version], 'rb') as handle:
        payload = pickle.load(handle)
    if len(payload) != 7:
        raise ValueError(f'nv{version}: expected 7 items, got {len(payload)}')
    return payload


dataset_rows = []
for version in DATASET_PATHS:
    x_train, y_train, x_test, y_test, noise_scale, noise_mask, subject_names = load_dataset(version)
    if x_train.shape != y_train.shape or x_test.shape != y_test.shape:
        raise ValueError(f'nv{version}: noisy/clean mismatch')
    if x_test.shape[1:] != (512, 1):
        raise ValueError(f'nv{version}: expected (N,512,1), got {x_test.shape}')
    if not (len(x_test) == len(noise_scale) == len(noise_mask) == len(subject_names)):
        raise ValueError(f'nv{version}: metadata length mismatch')
    dataset_rows.append({
        'noise_version': f'nv{version}', 'train_segments': len(x_train),
        'test_segments': len(x_test), 'shape': str(x_test.shape),
        'dtype': str(x_test.dtype),
        'clean_test_sha256': hashlib.sha256(np.asarray(y_test).tobytes()).hexdigest(),
        'noisy_test_sha256': hashlib.sha256(np.asarray(x_test).tobytes()).hexdigest(),
    })
    del x_train, y_train, x_test, y_test, noise_scale, noise_mask, subject_names
    gc.collect()
dataset_audit = pd.DataFrame(dataset_rows)
dataset_audit.to_csv(REPORT_DIR / 'dataset_audit.csv', index=False)
display(dataset_audit)
'''


CHECKPOINTS = r'''def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(artifact, protocol):
    filename = f'best_{artifact.lower()}_protocol{protocol}_{CHECKPOINT_TAG}.pth'
    matches = sorted(INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f'Need exactly one {filename}; found {matches}')
    return matches[0]


checkpoint_rows = []
if RUN_INFERENCE:
    for artifact in MODEL_ARTIFACTS_TO_EVALUATE:
        for version in NOISE_VERSIONS:
            path = resolve_checkpoint(artifact, version)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload.get('artifact') != artifact or int(payload.get('protocol', -1)) != version:
                raise RuntimeError(f'Checkpoint identity mismatch: {path}')
            if payload.get('config') != CONFIG:
                raise RuntimeError(f'Checkpoint configuration mismatch: {path}')
            if 'ema_model_state_dict' not in payload:
                raise RuntimeError(f'EMA inference weights missing: {path}')
            checkpoint_rows.append({
                'train_artifact': artifact, 'protocol': version, 'noise_version': f'nv{version}',
                'selected_epoch': int(payload['epoch']) + 1,
                'best_validation_loss': float(payload['best_validation_loss']),
                'sha256': sha256(path), 'path': str(path),
            })
else:
    print('Aggregation-only mode: checkpoints are not required.')
checkpoint_manifest = pd.DataFrame(checkpoint_rows)
checkpoint_manifest.to_csv(REPORT_DIR / 'checkpoint_manifest.csv', index=False)
display(checkpoint_manifest)
'''


METRICS = r'''def artifact_labels(mask):
    names = np.array(['BW', 'MA', 'EM'])
    labels = []
    for row in np.asarray(mask).astype(bool):
        active = names[row]
        labels.append('Clean' if len(active) == 0 else '+'.join(active))
    return labels


def calculate_metrics(clean, noisy, prediction, eps=1e-12):
    clean, noisy, prediction = [np.asarray(value, dtype=np.float64).reshape(len(value), -1)
                                for value in (clean, noisy, prediction)]
    error = prediction - clean
    input_error = noisy - clean
    squared = error ** 2
    ssd = squared.sum(axis=1)
    signal_power = (clean ** 2).sum(axis=1)
    input_power = (input_error ** 2).sum(axis=1)
    centered = clean - clean.mean(axis=1, keepdims=True)
    standard_denominator = (centered ** 2).sum(axis=1)
    repo_denominator = ((prediction - clean.mean()) ** 2).sum(axis=1)
    norm_product = np.linalg.norm(clean, axis=1) * np.linalg.norm(prediction, axis=1)
    snr_in = 10 * np.log10(np.maximum(signal_power, eps) / np.maximum(input_power, eps))
    snr_out = 10 * np.log10(np.maximum(signal_power, eps) / np.maximum(ssd, eps))
    return pd.DataFrame({
        'SSD': ssd,
        'MAD_mV': np.abs(error).max(axis=1),
        'PRD_paper_pct': 100 * np.sqrt(ssd / np.maximum(standard_denominator, eps)),
        'PRD_repo_pct': 100 * np.sqrt(ssd / np.maximum(repo_denominator, eps)),
        'CosSim': (clean * prediction).sum(axis=1) / np.maximum(norm_product, eps),
        'RMSE_mV': np.sqrt(squared.mean(axis=1)),
        'MAE_mV': np.abs(error).mean(axis=1),
        'SNR_in_dB': snr_in,
        'SNR_dB': snr_out,
        'SNR_improvement_dB': snr_out - snr_in,
    })
'''


INFERENCE = r'''def build_model():
    return ResearchLossDDPM(UNet1D(2, 80, 128, 1), CONFIG, DEVICE).to(DEVICE)


def test_count(array):
    return len(array) if MAX_TEST_SEGMENTS is None else min(len(array), int(MAX_TEST_SEGMENTS))


run_rows = []
if RUN_INFERENCE:
    for artifact in MODEL_ARTIFACTS_TO_EVALUATE:
        output_frames = []
        for version in NOISE_VERSIONS:
            checkpoint_path = resolve_checkpoint(artifact, version)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = build_model()
            model.load_state_dict(checkpoint['ema_model_state_dict'], strict=True)
            model.eval()

            payload = load_dataset(version)
            x_test, y_test, noise_scale, noise_mask, subject_names = payload[2:]
            count = test_count(x_test)
            noisy = np.asarray(x_test[:count], dtype=np.float32)[:, :, 0]
            clean = np.asarray(y_test[:count], dtype=np.float32)[:, :, 0]
            scale = np.maximum(np.max(np.abs(clean), axis=1, keepdims=True), 1e-8).astype(np.float32)

            predictions = np.lib.format.open_memmap(
                PREDICTION_DIR / f'xhat_model_{artifact.lower()}_rmn_nv{version}_K{K}.npy',
                mode='w+', dtype=np.float32, shape=clean.shape,
            ) if SAVE_PREDICTIONS else np.empty_like(clean)

            artifact_seed = {'BW': 11, 'EM': 22, 'MA': 33}[artifact]
            torch.manual_seed(SEED + artifact_seed * 100 + version)
            torch.cuda.manual_seed_all(SEED + artifact_seed * 100 + version)
            started = time.perf_counter()
            for start in tqdm(range(0, count, INFERENCE_BATCH_SIZE),
                              desc=f'DDPM-{artifact} zero-shot RMN nv{version}'):
                stop = min(start + INFERENCE_BATCH_SIZE, count)
                condition = torch.from_numpy((noisy[start:stop] / scale[start:stop])[:, None, :]).to(DEVICE)
                with torch.inference_mode():
                    output = model.denoising(condition, use_ddim=True, ddim_steps=DDIM_STEPS,
                                             ddim_eta=DDIM_ETA, num_shots=K)
                if output.shape != condition.shape or not torch.isfinite(output).all():
                    raise RuntimeError(f'Invalid output {artifact}/nv{version}/{start}')
                predictions[start:stop] = output.cpu().numpy()[:, 0, :] * scale[start:stop]
            elapsed = time.perf_counter() - started
            if hasattr(predictions, 'flush'):
                predictions.flush()

            metric_frame = calculate_metrics(clean, noisy, predictions)
            metric_frame.insert(0, 'checkpoint_sha256', sha256(checkpoint_path))
            metric_frame.insert(0, 'segment_index', np.arange(count))
            metric_frame.insert(0, 'subject', list(subject_names)[:count])
            labels = artifact_labels(np.asarray(noise_mask)[:count])
            metric_frame.insert(0, 'artifact', labels)
            metric_frame.insert(0, 'test_artifact', labels)
            metric_frame.insert(0, 'noise_scale', np.asarray(noise_scale)[:count])
            metric_frame.insert(0, 'protocol', version)
            metric_frame.insert(0, 'noise_version', f'nv{version}')
            metric_frame.insert(0, 'train_artifact', artifact)
            metric_frame.insert(0, 'model', f'DDPM-{artifact} zero-shot RMN')
            output_frames.append(metric_frame)
            run_rows.append({
                'model': f'DDPM-{artifact} zero-shot RMN', 'train_artifact': artifact,
                'noise_version': f'nv{version}', 'protocol': version, 'segments': count,
                'inference_seconds': elapsed, 'segments_per_second': count / elapsed,
                'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
                'full_test': MAX_TEST_SEGMENTS is None,
            })
            del model, checkpoint, payload, x_test, y_test, noisy, clean, scale, predictions
            torch.cuda.empty_cache()
            gc.collect()
        artifact_output = pd.concat(output_frames, ignore_index=True)
        artifact_output.to_csv(METRIC_DIR / f'zero_shot_metrics_ddpm_{artifact.lower()}.csv', index=False)
        print(artifact, 'metric rows:', len(artifact_output))
    pd.DataFrame(run_rows).to_csv(REPORT_DIR / 'inference_runs.csv', index=False)
else:
    print('Inference disabled; attached metric CSVs will be aggregated.')
'''


AGGREGATE = r'''def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


metric_candidates = list(INPUT_ROOT.rglob('zero_shot_metrics_ddpm_*.csv'))
metric_candidates.extend(METRIC_DIR.glob('zero_shot_metrics_ddpm_*.csv'))
selected_files = {}
for path in metric_candidates:
    name = path.name
    digest = file_hash(path)
    if name in selected_files and selected_files[name][0] != digest:
        raise RuntimeError(f'Conflicting versions of {name}: {selected_files[name][1]} and {path}')
    selected_files[name] = (digest, path)

if not selected_files:
    raise FileNotFoundError('No zero_shot_metrics_ddpm_*.csv files found')
zero_shot = pd.concat([pd.read_csv(item[1]) for item in selected_files.values()], ignore_index=True)
identity = ['model', 'noise_version', 'segment_index']
if zero_shot.duplicated(identity).any():
    raise RuntimeError('Duplicate zero-shot model/version/segment rows')
available_artifacts = set(zero_shot.train_artifact.unique())
if REQUIRE_ALL_ZERO_SHOT_MODELS and available_artifacts != set(VALID_ARTIFACTS):
    raise RuntimeError(f'Need BW, EM, MA metrics; found {sorted(available_artifacts)}')
zero_shot.to_csv(REPORT_DIR / 'zero_shot_metrics_all.csv', index=False)
print('Available zero-shot models:', sorted(available_artifacts), '| rows:', len(zero_shot))

COMMON_METRICS = ['SSD', 'MAD_mV', 'PRD_paper_pct', 'PRD_repo_pct',
                  'CosSim', 'RMSE_mV', 'MAE_mV', 'SNR_dB']


def summary_table(frame, group_columns):
    means = frame.groupby(group_columns)[COMMON_METRICS].mean().add_suffix('_mean')
    stds = frame.groupby(group_columns)[COMMON_METRICS].std(ddof=0).add_suffix('_std')
    counts = frame.groupby(group_columns).size().rename('N')
    return pd.concat([counts, means, stds], axis=1).reset_index()


full_summary = summary_table(zero_shot, ['model', 'train_artifact'])
matching = zero_shot[zero_shot.test_artifact == zero_shot.train_artifact].copy()
subset_summary = summary_table(matching, ['model', 'train_artifact', 'test_artifact'])
composition_summary = summary_table(zero_shot, ['model', 'train_artifact', 'test_artifact'])
full_summary.to_csv(REPORT_DIR / 'zero_shot_summary_full_rmn.csv', index=False)
subset_summary.to_csv(REPORT_DIR / 'zero_shot_summary_matching_subset.csv', index=False)
composition_summary.to_csv(REPORT_DIR / 'zero_shot_summary_by_composition.csv', index=False)
display(full_summary)
display(subset_summary)
'''


BASELINE_COMPARISON = r'''baseline_paths = sorted(INPUT_ROOT.rglob('joint_metrics_per_segment.csv'))
if RUN_COMPARISON and baseline_paths:
    digests = {file_hash(path): path for path in baseline_paths}
    if len(digests) != 1:
        raise RuntimeError(f'Conflicting joint_metrics_per_segment.csv files: {baseline_paths}')
    baseline = pd.read_csv(next(iter(digests.values())))
    baseline = baseline[baseline.model != 'MyDDPM'].copy()
    required = {'model', 'noise_version', 'segment_index', 'artifact', *COMMON_METRICS}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f'Baseline evidence missing columns: {sorted(missing)}')

    expected_keys = zero_shot[['noise_version', 'segment_index']].drop_duplicates()
    for model_name, model_frame in baseline.groupby('model'):
        observed = model_frame[['noise_version', 'segment_index']].drop_duplicates()
        if len(observed.merge(expected_keys)) != len(expected_keys):
            raise RuntimeError(f'Baseline {model_name} does not cover every zero-shot test row')

    common_zero = zero_shot[['model', 'noise_version', 'segment_index', 'artifact', *COMMON_METRICS]].copy()
    full_comparison = pd.concat([baseline[['model', 'noise_version', 'segment_index', 'artifact', *COMMON_METRICS]],
                                 common_zero], ignore_index=True)
    full_comparison.to_csv(REPORT_DIR / 'comparison_metrics_per_segment.csv', index=False)
    comparison_summary = summary_table(full_comparison, ['model'])
    comparison_summary.to_csv(REPORT_DIR / 'comparison_summary_full_rmn.csv', index=False)

    subset_frames = []
    for artifact in sorted(available_artifacts):
        zero_part = common_zero[(zero_shot.train_artifact == artifact).to_numpy() &
                                (common_zero.artifact == artifact)]
        baseline_part = baseline[baseline.artifact == artifact][
            ['model', 'noise_version', 'segment_index', 'artifact', *COMMON_METRICS]
        ].copy()
        zero_part = zero_part.copy()
        baseline_part['evaluation_subset'] = artifact
        zero_part['evaluation_subset'] = artifact
        subset_frames.extend([baseline_part, zero_part])
    subset_comparison = pd.concat(subset_frames, ignore_index=True)
    subset_comparison.to_csv(REPORT_DIR / 'comparison_matching_subset_per_segment.csv', index=False)
    subset_comparison_summary = summary_table(subset_comparison, ['evaluation_subset', 'model'])
    subset_comparison_summary.to_csv(REPORT_DIR / 'comparison_matching_subset_summary.csv', index=False)
    display(subset_comparison_summary)
else:
    baseline = None
    print('Baseline comparison skipped. Add joint_metrics_per_segment.csv as Kaggle Input to enable it.')
'''


PLOTS = r'''COLORS = {'BW': '#4C78A8', 'EM': '#F58518', 'MA': '#54A24B'}
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.grid': True, 'grid.alpha': 0.2, 'font.size': 10,
                     'axes.titleweight': 'bold'})

fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, 0.55 * len(full_summary))))
for axis, metric, title, direction in (
        (axes[0], 'PRD_paper_pct_mean', 'Full RMN: standard PRD (%)', 'lower'),
        (axes[1], 'SNR_dB_mean', 'Full RMN: output SNR (dB)', 'higher')):
    ordered = full_summary.sort_values(metric, ascending=(direction == 'lower'))
    positions = np.arange(len(ordered))
    axis.barh(positions, ordered[metric], color=[COLORS[a] for a in ordered.train_artifact])
    axis.set(yticks=positions, yticklabels=ordered.model, xlabel=f'{title} — {direction} is better')
    axis.invert_yaxis()
fig.suptitle('QTDB single-artifact DDPM checkpoints — zero-shot RMN', y=1.02)
fig.tight_layout()
fig.savefig(REPORT_DIR / 'zero_shot_full_rmn_comparison.png', dpi=180, bbox_inches='tight')
plt.show()

compositions = ['Clean', 'BW', 'MA', 'BW+MA', 'EM', 'BW+EM', 'MA+EM', 'BW+MA+EM']
matrix = composition_summary.pivot(index='model', columns='test_artifact',
                                   values='SNR_dB_mean').reindex(columns=compositions)
fig, axis = plt.subplots(figsize=(13, max(3.5, 0.8 * len(matrix))))
image = axis.imshow(matrix.to_numpy(), aspect='auto', cmap='YlGnBu')
axis.grid(False)
axis.set(xticks=np.arange(len(matrix.columns)), xticklabels=matrix.columns,
         yticks=np.arange(len(matrix.index)), yticklabels=matrix.index,
         xlabel='RMN artifact composition', ylabel='Zero-shot model',
         title='Mean output SNR by RMN artifact composition')
axis.tick_params(axis='x', rotation=30)
for row in range(len(matrix.index)):
    for column in range(len(matrix.columns)):
        value = matrix.iloc[row, column]
        if np.isfinite(value):
            axis.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(image, ax=axis, label='Mean SNR-out (dB)')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'zero_shot_snr_by_rmn_composition.png', dpi=180, bbox_inches='tight')
plt.show()

if baseline is not None:
    focus_models = ['FGDAE q=1', 'FGDAE q=2', 'FGDAE q=3', 'FGDAE q=4'] + sorted(zero_shot.model.unique())
    plot_data = subset_comparison_summary[subset_comparison_summary.model.isin(focus_models)].copy()
    artifacts = sorted(plot_data.evaluation_subset.unique())
    fig, axes = plt.subplots(1, len(artifacts), figsize=(6 * len(artifacts), 5), squeeze=False)
    for axis, artifact in zip(axes.ravel(), artifacts):
        part = plot_data[plot_data.evaluation_subset == artifact].sort_values('PRD_paper_pct_mean')
        positions = np.arange(len(part))
        colors = [COLORS[artifact] if name.startswith('DDPM-') else '#7A8793' for name in part.model]
        axis.barh(positions, part.PRD_paper_pct_mean, color=colors)
        axis.set(yticks=positions, yticklabels=part.model,
                 xlabel='Mean standard PRD (%) — lower is better', title=f'Test subset: {artifact}')
        axis.invert_yaxis()
    fig.suptitle('Artifact-specific subset: zero-shot DDPM vs FGDAE', y=1.02)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'zero_shot_vs_fgdae_matching_subsets.png', dpi=180, bbox_inches='tight')
    plt.show()
'''


PACKAGE = r'''run_info = {
    'model_artifacts_requested': list(MODEL_ARTIFACTS_TO_EVALUATE),
    'model_artifacts_available': sorted(available_artifacts),
    'noise_versions': list(NOISE_VERSIONS),
    'protocol_mapping': {'nv1': 'P1', 'nv2': 'P2'},
    'K': K, 'ddim_steps': DDIM_STEPS, 'ddim_eta': DDIM_ETA,
    'max_test_segments': MAX_TEST_SEGMENTS,
    'full_test': MAX_TEST_SEGMENTS is None,
    'normalization': MODEL_NORMALIZATION,
    'checkpoint_manifest': checkpoint_rows if RUN_INFERENCE else [],
    'dataset_audit': dataset_rows,
    'baseline_comparison_available': baseline is not None,
    'interpretation': 'Transfer/zero-shot RMN only; not an equal-training-distribution benchmark.',
    'device': str(DEVICE), 'torch_version': torch.__version__,
    'python_version': platform.python_version(),
}
(REPORT_DIR / 'run_info.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')

archive_path = OUT.parent / 'qtdb_zero_shot_rmn_results.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(OUT))
print('Saved:', archive_path)
'''


cells = [
    markdown("""# QTDB single-noise checkpoints → zero-shot RMN benchmark

Notebook đánh giá các checkpoint DDPM-BW, DDPM-EM và DDPM-MA hiện tại trên test set Random Mixed Noise chính thức của `fully_gated_DAE`.

Kết quả phải được gọi là **zero-shot/transfer RMN**, không phải so sánh cùng điều kiện huấn luyện với FGDAE. Notebook cũng tạo artifact-specific subset benchmark trên các hàng RMN thuần BW, EM hoặc MA.

### Kaggle Inputs

- Hai checkpoint `best_<artifact>_protocol1/2_norm_loss_v2.pth` cho artifact đang chạy.
- `Dataset_1.pkl` và `Dataset_2.pkl` nếu đã có; nếu thiếu, notebook sẽ tạo lại từ QTDB/NSTDB và cần bật Internet.
- Tùy chọn: `joint_metrics_per_segment.csv` để so sánh trên đúng segment với FGDAE và các baseline.

Mặc định chạy một artifact để tránh hết thời gian. Lần lượt đặt `('BW',)`, `('EM',)` và `('MA',)`. Sau ba phiên, chạy một phiên tổng hợp với `RUN_INFERENCE=False`, `REQUIRE_ALL_ZERO_SHOT_MODELS=True` và thêm ba output làm Input."""),
    markdown("## 1. Cài thư viện"), code(INSTALL),
    markdown("## 2. Cấu hình"), code(SETUP),
    markdown("## 3. Chuẩn bị RMN Dataset 1/2"), code(DATA_PREP),
    markdown("## 4. Audit dataset"), code(DATA_AUDIT),
    markdown("## 5. Kiến trúc checkpoint hiện tại"), code(ARCHITECTURE),
    markdown("## 6. Conditional DDPM/DDIM"), code(DDPM),
    markdown("## 7. Kiểm tra checkpoint và mapping P1/nv1, P2/nv2"), code(CHECKPOINTS),
    markdown("## 8. Metric chung"), code(METRICS),
    markdown("## 9. Zero-shot RMN inference"), code(INFERENCE),
    markdown("## 10. Ghép các phiên BW/EM/MA"), code(AGGREGATE),
    markdown("## 11. Artifact-specific subset benchmark với baseline"), code(BASELINE_COMPARISON),
    markdown("## 12. Biểu đồ"), code(PLOTS),
    markdown("## 13. Đóng gói"), code(PACKAGE),
    markdown("""## Cách diễn giải

- `zero_shot_summary_full_rmn.csv`: khả năng của từng model đơn nhiễu trên toàn bộ RMN, gồm cả nhiễu hỗn hợp và Clean.
- `zero_shot_summary_matching_subset.csv`: DDPM-BW chỉ trên hàng BW, DDPM-EM chỉ trên hàng EM, DDPM-MA chỉ trên hàng MA.
- `comparison_matching_subset_summary.csv`: so sánh zero-shot checkpoint với FGDAE/baseline trên đúng cùng segment, nếu đã thêm `joint_metrics_per_segment.csv`.
- `zero_shot_snr_by_rmn_composition.png`: độ tổng quát qua tám thành phần artifact RMN.

Không dùng bảng này để khẳng định model nào tốt hơn dưới cùng điều kiện train. Benchmark công bằng chính thức vẫn cần DDPM được huấn luyện trên RMN Dataset 1/2."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET}")
