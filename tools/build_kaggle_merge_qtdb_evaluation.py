"""Build a CPU-only Kaggle notebook that merges BW/EM/MA evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "kaggle_merge_qtdb_bw_em_ma_evaluation.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


SETUP = r'''import hashlib
import io
import json
import platform
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

INPUT_ROOT = Path('/kaggle/input')
OUT = Path('/kaggle/working/qtdb_bw_em_ma_final_evaluation')
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_ARTIFACTS = ('BW', 'EM', 'MA')
EXPECTED_PROTOCOLS = (1, 2)
TARGET_K = 1
SEED = 1234

COLORS = {'BW': '#4C78A8', 'EM': '#F58518', 'MA': '#54A24B'}
HATCHES = {1: '', 2: '//'}
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#4A4A4A', 'axes.labelcolor': '#222222',
    'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': 0.20,
    'font.size': 10, 'legend.frameon': False,
})

print('Input:', INPUT_ROOT)
print('Output:', OUT)
'''


LOAD_RESULTS = r'''def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def csv_candidate(raw, location, kind):
    frame = pd.read_csv(io.BytesIO(raw))
    if 'artifact' not in frame.columns:
        return None
    artifacts = tuple(sorted(frame.artifact.dropna().astype(str).str.upper().unique()))
    if len(artifacts) != 1 or artifacts[0] not in EXPECTED_ARTIFACTS:
        return None
    frame['artifact'] = frame['artifact'].astype(str).str.upper()
    return {
        'artifact': artifacts[0], 'frame': frame, 'location': location,
        'kind': kind, 'sha256': sha256_bytes(raw), 'bytes': len(raw),
    }


candidates = []
for path in sorted(INPUT_ROOT.rglob('metrics_per_sample.csv')):
    candidate = csv_candidate(path.read_bytes(), str(path), 'file')
    if candidate:
        candidates.append(candidate)

for archive_path in sorted(INPUT_ROOT.rglob('*.zip')):
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if Path(member).name != 'metrics_per_sample.csv':
                    continue
                candidate = csv_candidate(
                    archive.read(member), f'{archive_path}::{member}', 'zip'
                )
                if candidate:
                    candidates.append(candidate)
    except zipfile.BadZipFile:
        print('Skip invalid ZIP:', archive_path)

selected, source_rows = {}, []
for artifact in EXPECTED_ARTIFACTS:
    artifact_candidates = [item for item in candidates if item['artifact'] == artifact]
    unique = {}
    for item in artifact_candidates:
        unique.setdefault(item['sha256'], item)
    if not unique:
        raise FileNotFoundError(
            f'Missing {artifact} metrics_per_sample.csv in {INPUT_ROOT}. '
            f'Add the {artifact} evaluation output or ZIP as Kaggle Input.'
        )
    if len(unique) > 1:
        locations = [item['location'] for item in unique.values()]
        raise RuntimeError(
            f'Found conflicting {artifact} result versions. Keep one version only: {locations}'
        )
    item = next(iter(unique.values()))
    selected[artifact] = item['frame']
    source_rows.append({key: item[key] for key in ('artifact', 'location', 'kind', 'sha256', 'bytes')})

source_manifest = pd.DataFrame(source_rows)
source_manifest.to_csv(OUT / 'source_manifest.csv', index=False)
display(source_manifest)
'''


VALIDATE_AND_MERGE = r'''REQUIRED_COLUMNS = {
    'sample_id', 'record_id', 'artifact', 'protocol', 'K', 'r',
    'ssd', 'mad', 'prd_repo', 'prd_standard', 'cosine_similarity',
    'snr_in_db', 'snr_out_db', 'snr_improvement_db', 'inference_time_ms',
}

for artifact, frame in selected.items():
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f'{artifact} is missing columns: {sorted(missing)}')
    frame['protocol'] = pd.to_numeric(frame['protocol'], errors='raise').astype(int)
    frame['K'] = pd.to_numeric(frame['K'], errors='raise').astype(int)
    frame['r'] = pd.to_numeric(frame['r'], errors='raise')
    if set(frame.protocol.unique()) != set(EXPECTED_PROTOCOLS):
        raise ValueError(f'{artifact}: expected protocols {EXPECTED_PROTOCOLS}, got {sorted(frame.protocol.unique())}')
    if TARGET_K not in set(frame.K.unique()):
        raise ValueError(f'{artifact}: K={TARGET_K} is unavailable; found {sorted(frame.K.unique())}')
    selected[artifact] = frame[frame.K == TARGET_K].copy()
    if selected[artifact].duplicated(['artifact', 'protocol', 'K', 'sample_id']).any():
        raise ValueError(f'{artifact}: duplicate sample rows')

# All artifacts must describe the same paired test samples and r values.
reference = None
for artifact in EXPECTED_ARTIFACTS:
    keys = (selected[artifact][['protocol', 'K', 'sample_id', 'record_id', 'r']]
            .sort_values(['protocol', 'K', 'sample_id']).reset_index(drop=True))
    if reference is None:
        reference = keys
    else:
        if not keys.drop(columns='r').equals(reference.drop(columns='r')):
            raise ValueError(f'{artifact}: test sample identities do not match BW')
        if not np.allclose(keys.r.to_numpy(), reference.r.to_numpy(), rtol=0, atol=1e-8):
            raise ValueError(f'{artifact}: r values do not match BW')

per_sample = pd.concat([selected[a] for a in EXPECTED_ARTIFACTS], ignore_index=True)
if 'rmse' not in per_sample.columns:
    per_sample['rmse'] = np.sqrt(per_sample['ssd'] / 512.0)
per_sample['r_group'] = pd.cut(
    per_sample.r, bins=[0.2, 0.6, 1.0, 1.5, 2.001], right=False,
    labels=['0.20-<0.60', '0.60-<1.00', '1.00-<1.50', '1.50-2.00'],
)
if per_sample.r_group.isna().any():
    raise ValueError('Some r values fall outside the expected range [0.20, 2.00]')

per_sample.to_csv(OUT / 'metrics_per_sample_all.csv', index=False)
audit = (per_sample.groupby(['artifact', 'protocol', 'K'])
         .agg(N=('sample_id', 'size'), records=('record_id', 'nunique'),
              r_min=('r', 'min'), r_max=('r', 'max'))
         .reset_index())
display(audit)
'''


SUMMARIES = r'''METRICS = [
    metric for metric in (
        'ssd', 'rmse', 'mae', 'mad', 'prd_repo', 'prd_standard',
        'cosine_similarity', 'snr_in_db', 'snr_out_db',
        'snr_improvement_db', 'inference_time_ms',
    ) if metric in per_sample.columns
]


def summarize(frame, artifact, protocol, r_group='ALL'):
    row = {'artifact': artifact, 'protocol': protocol, 'K': TARGET_K,
           'r_group': r_group, 'N': len(frame)}
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        row[f'{metric}_valid_N'] = len(finite)
        row[f'{metric}_nonfinite_N'] = len(values) - len(finite)
        row[f'{metric}_mean'] = float(np.mean(finite)) if len(finite) else np.nan
        row[f'{metric}_std'] = float(np.std(finite, ddof=0)) if len(finite) else np.nan
        row[f'{metric}_median'] = float(np.median(finite)) if len(finite) else np.nan
    return row


summary_rows = []
for (artifact, protocol), frame in per_sample.groupby(['artifact', 'protocol']):
    summary_rows.append(summarize(frame, artifact, protocol))
    for label, part in frame.groupby('r_group', observed=True):
        summary_rows.append(summarize(part, artifact, protocol, str(label)))
for artifact, frame in per_sample.groupby('artifact'):
    summary_rows.append(summarize(frame, artifact, 'pooled'))
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / 'summary_all.csv', index=False)

record_rows = []
for (artifact, protocol, record_id), frame in per_sample.groupby(
        ['artifact', 'protocol', 'record_id']):
    row = summarize(frame, artifact, protocol)
    row['record_id'] = record_id
    record_rows.append(row)
pd.DataFrame(record_rows).to_csv(OUT / 'summary_by_record_all.csv', index=False)

all_rows = summary[summary.r_group == 'ALL'].copy()
readable = all_rows[['artifact', 'protocol', 'K', 'N']].copy()
for metric in METRICS:
    readable[metric] = [f'{mean:.4f} ± {std:.4f}' for mean, std in zip(
        all_rows[f'{metric}_mean'], all_rows[f'{metric}_std'])]
readable.to_csv(OUT / 'summary_mean_std_all.csv', index=False)

ranking = summary[(summary.protocol == 'pooled') & (summary.r_group == 'ALL')][
    ['artifact', 'N', 'snr_improvement_db_mean', 'prd_standard_mean',
     'rmse_mean', 'cosine_similarity_mean']
].copy()
ranking['rank_delta_snr'] = ranking.snr_improvement_db_mean.rank(ascending=False, method='min').astype(int)
ranking['rank_prd'] = ranking.prd_standard_mean.rank(ascending=True, method='min').astype(int)
ranking['rank_rmse'] = ranking.rmse_mean.rank(ascending=True, method='min').astype(int)
ranking['rank_cosine'] = ranking.cosine_similarity_mean.rank(ascending=False, method='min').astype(int)
ranking.to_csv(OUT / 'artifact_ranking.csv', index=False)
display(readable)
display(ranking.sort_values('rank_delta_snr'))
'''


PLOT_HELPERS = r'''figure_rows = []


def save_figure(fig, filename, purpose):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches='tight')
    figure_rows.append({'file': filename, 'K': TARGET_K, 'purpose': purpose})
    plt.show()
    plt.close(fig)


groups = [(artifact, protocol) for artifact in EXPECTED_ARTIFACTS
          for protocol in EXPECTED_PROTOCOLS]
group_labels = [f'{artifact}-P{protocol}' for artifact, protocol in groups]
'''


PLOT_OVERVIEW = r'''specs = [
    ('snr_improvement_db', 'SNR improvement', 'dB'),
    ('prd_standard', 'Standard PRD', '%'),
    ('rmse', 'RMSE', 'signal unit'),
    ('cosine_similarity', 'Cosine similarity', 'score'),
]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for axis, (metric, title, unit) in zip(axes.ravel(), specs):
    values, spreads = [], []
    for artifact, protocol in groups:
        series = per_sample[(per_sample.artifact == artifact) &
                            (per_sample.protocol == protocol)][metric]
        values.append(series.mean())
        spreads.append(series.std(ddof=0))
    positions = np.arange(len(groups))
    bars = axis.bar(positions, values, yerr=spreads, capsize=3,
                    color=[COLORS[a] for a, _ in groups],
                    edgecolor='#333333', linewidth=0.6)
    for bar, (_, protocol) in zip(bars, groups):
        bar.set_hatch(HATCHES[protocol])
    if metric == 'snr_improvement_db':
        axis.axhline(0, color='#333333', linewidth=1)
    axis.set(title=f'{title} — mean ± population SD', ylabel=unit,
             xticks=positions, xticklabels=group_labels)
    axis.tick_params(axis='x', rotation=30)
fig.suptitle(f'Final BW/EM/MA comparison (K={TARGET_K})', y=1.01)
save_figure(fig, 'final_metric_overview.png',
            'Principal metrics for all artifacts and protocols; error bars are population SD.')
'''


PLOT_SEVERITY_PROTOCOL = r'''r_order = [str(value) for value in per_sample.r_group.cat.categories]
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for axis, (metric, title, unit) in zip(axes.ravel(), specs):
    for artifact in EXPECTED_ARTIFACTS:
        part = (per_sample[per_sample.artifact == artifact]
                .groupby('r_group', observed=True)[metric].mean().reindex(r_order))
        axis.plot(r_order, part.values, marker='o', linewidth=2,
                  color=COLORS[artifact], label=artifact)
    if metric == 'snr_improvement_db':
        axis.axhline(0, color='#333333', linewidth=1)
    axis.set(title=f'{title} by r range', xlabel='Noise scale r', ylabel=unit)
    axis.tick_params(axis='x', rotation=20)
    axis.legend(title='Artifact')
fig.suptitle(f'Robustness across noise severity (protocols pooled, K={TARGET_K})', y=1.01)
save_figure(fig, 'final_performance_by_noise_level.png',
            'Mean performance across controlled r ranges, pooling the two protocols.')

protocol_specs = [('snr_improvement_db', 'SNR improvement', 'dB'),
                  ('prd_standard', 'Standard PRD', '%')]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for axis, (metric, title, unit) in zip(axes, protocol_specs):
    for artifact in EXPECTED_ARTIFACTS:
        means = [per_sample[(per_sample.artifact == artifact) &
                            (per_sample.protocol == protocol)][metric].mean()
                 for protocol in EXPECTED_PROTOCOLS]
        axis.plot(EXPECTED_PROTOCOLS, means, marker='o', linewidth=2.2,
                  color=COLORS[artifact], label=artifact)
    axis.set(title=f'{title}: protocol sensitivity', xlabel='Protocol', ylabel=unit,
             xticks=EXPECTED_PROTOCOLS)
    axis.legend()
save_figure(fig, 'final_protocol_comparison.png',
            'Change in mean performance between protocol 1 and protocol 2.')
'''


PLOT_DISTRIBUTIONS = r'''fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for axis, (metric, title, unit) in zip(axes, (
        ('snr_improvement_db', 'SNR improvement distribution', 'dB'),
        ('prd_standard', 'Standard PRD distribution', '%'))):
    arrays = [per_sample[(per_sample.artifact == artifact) &
                         (per_sample.protocol == protocol)][metric]
              .replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
              for artifact, protocol in groups]
    options = dict(showfliers=False, patch_artist=True, widths=0.65)
    try:
        boxes = axis.boxplot(arrays, tick_labels=group_labels, **options)
    except TypeError:
        boxes = axis.boxplot(arrays, labels=group_labels, **options)
    for box, (artifact, protocol) in zip(boxes['boxes'], groups):
        box.set(facecolor=COLORS[artifact], alpha=0.70,
                edgecolor='#333333', hatch=HATCHES[protocol])
    if metric == 'snr_improvement_db':
        axis.axhline(0, color='#333333', linewidth=1)
    axis.set(title=title + ' (visual outliers hidden)', ylabel=unit)
    axis.tick_params(axis='x', rotation=30)
save_figure(fig, 'final_metric_distributions.png',
            'Median and interquartile spread; all observations remain in the CSV files.')

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
scatter = per_sample.sample(min(9000, len(per_sample)), random_state=SEED)
finite = scatter[['snr_in_db', 'snr_out_db']].replace([np.inf, -np.inf], np.nan).dropna()
low, high = float(finite.min().min()), float(finite.max().max())
for axis, artifact in zip(axes, EXPECTED_ARTIFACTS):
    for protocol, marker in ((1, 'o'), (2, '^')):
        part = scatter[(scatter.artifact == artifact) & (scatter.protocol == protocol)]
        axis.scatter(part.snr_in_db, part.snr_out_db, s=10, alpha=0.22,
                     color=COLORS[artifact], marker=marker, label=f'P{protocol}')
    axis.plot([low, high], [low, high], '--', color='#333333', linewidth=1.2,
              label='No change')
    axis.set(title=artifact, xlabel='Input SNR (dB)', ylabel='Output SNR (dB)',
             xlim=(low, high), ylim=(low, high))
    axis.legend()
fig.suptitle(f'Input–output SNR relationship (K={TARGET_K})', y=1.02)
save_figure(fig, 'final_snr_input_output_scatter.png',
            'Per-segment input/output SNR; points above the diagonal improved.')
'''


PLOT_HEATMAP = r'''record_metric = (per_sample.groupby(['record_id', 'artifact', 'protocol'])
                 ['snr_improvement_db'].mean().unstack(['artifact', 'protocol']))
columns = [(artifact, protocol) for artifact, protocol in groups
           if (artifact, protocol) in record_metric.columns]
record_metric = record_metric.reindex(columns=pd.MultiIndex.from_tuples(columns))
matrix = record_metric.to_numpy(dtype=float)
limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
limit = max(limit, 1e-6)
fig, axis = plt.subplots(figsize=(max(9, 1.35 * len(columns)),
                                  max(6, 0.42 * len(record_metric))))
image = axis.imshow(matrix, aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
axis.grid(False)
axis.set(xticks=np.arange(len(columns)),
         xticklabels=[f'{a}-P{p}' for a, p in columns],
         yticks=np.arange(len(record_metric)), yticklabels=record_metric.index,
         xlabel='Artifact / protocol', ylabel='QTDB record',
         title=f'Mean SNR improvement by record (dB, K={TARGET_K})')
axis.tick_params(axis='x', rotation=30)
fig.colorbar(image, ax=axis, label='Mean ΔSNR (dB)')
save_figure(fig, 'final_record_snr_heatmap.png',
            'Cross-record consistency for every artifact/protocol experiment.')
'''


COLLECT_SIGNAL_IMAGES = r'''SIGNAL_DIR = OUT / 'individual_signal_figures'
SIGNAL_DIR.mkdir(exist_ok=True)
signal_assets = {}

for path in sorted(INPUT_ROOT.rglob('signal_examples_*.png')):
    data = path.read_bytes()
    signal_assets.setdefault(path.name, (data, str(path)))

for archive_path in sorted(INPUT_ROOT.rglob('*.zip')):
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name.startswith('signal_examples_') and name.endswith('.png'):
                    signal_assets.setdefault(name, (archive.read(member), f'{archive_path}::{member}'))
    except zipfile.BadZipFile:
        pass

for name, (data, _) in signal_assets.items():
    (SIGNAL_DIR / name).write_bytes(data)

expected_names = [f'signal_examples_{artifact.lower()}_protocol{protocol}_K{TARGET_K}.png'
                  for artifact in EXPECTED_ARTIFACTS for protocol in EXPECTED_PROTOCOLS]
available = [name for name in expected_names if (SIGNAL_DIR / name).is_file()]
if available:
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    for axis, name in zip(axes.ravel(), expected_names):
        path = SIGNAL_DIR / name
        if path.is_file():
            axis.imshow(plt.imread(path))
            axis.set_title(name.replace('signal_examples_', '').replace('.png', ''))
        else:
            axis.text(0.5, 0.5, f'Missing\n{name}', ha='center', va='center')
        axis.axis('off')
    save_figure(fig, 'final_signal_examples_montage.png',
                'Montage of the six artifact/protocol waveform and spectrum figures.')
else:
    print('No signal_examples_*.png found. Metric synthesis remains complete.')
'''


PACKAGE = r'''pd.DataFrame(figure_rows).to_csv(OUT / 'figure_manifest.csv', index=False)

run_info = {
    'artifacts': list(EXPECTED_ARTIFACTS),
    'protocols': list(EXPECTED_PROTOCOLS),
    'K': TARGET_K,
    'rows': len(per_sample),
    'records': int(per_sample.record_id.nunique()),
    'metrics': METRICS,
    'sources': source_rows,
    'python_version': platform.python_version(),
    'pandas_version': pd.__version__,
    'numpy_version': np.__version__,
}
(OUT / 'run_info.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')

archive_path = OUT.parent / 'qtdb_bw_em_ma_final_evaluation.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(OUT))

print('Final rows:', len(per_sample))
print('Figures:', len(figure_rows))
print('Saved:', archive_path)
'''


cells = [
    markdown("""# Final evaluation — tổng hợp QTDB BW, EM và MA

Notebook này **không chạy lại mô hình**. Nó ghép ba bộ kết quả đánh giá riêng của BW, EM và MA thành một evaluation cuối cùng, xác minh tính tương thích, tái tính bảng tổng hợp và tạo lại biểu đồ so sánh chung.

## Cách dùng trên Kaggle

1. Tạo notebook mới và upload notebook này.
2. Dùng **Add Input** để thêm ba output/dataset BW, EM và MA. Có thể thêm thư mục kết quả hoặc ba file ZIP.
3. Không cần GPU và không cần Internet.
4. Chạy **Save & Run All**.
5. Tải `qtdb_bw_em_ma_final_evaluation.zip` trong Output.

Notebook tự bỏ qua bản file/ZIP trùng nội dung. Nếu phát hiện hai phiên khác nhau cho cùng một artifact, notebook sẽ dừng để tránh âm thầm trộn sai kết quả."""),
    markdown("## 1. Cấu hình và thư viện"), code(SETUP),
    markdown("## 2. Phát hiện ba bộ kết quả"), code(LOAD_RESULTS),
    markdown("## 3. Kiểm tra tính tương thích và ghép dữ liệu"), code(VALIDATE_AND_MERGE),
    markdown("## 4. Tính lại bảng tổng hợp"), code(SUMMARIES),
    markdown("## 5. Thiết lập biểu đồ"), code(PLOT_HELPERS),
    markdown("## 6. So sánh metric tổng thể"), code(PLOT_OVERVIEW),
    markdown("## 7. So sánh theo mức nhiễu và protocol"), code(PLOT_SEVERITY_PROTOCOL),
    markdown("## 8. Phân phối và quan hệ SNR"), code(PLOT_DISTRIBUTIONS),
    markdown("## 9. Độ ổn định theo QTDB record"), code(PLOT_HEATMAP),
    markdown("## 10. Thu thập hình tín hiệu và tạo montage"), code(COLLECT_SIGNAL_IMAGES),
    markdown("## 11. Đóng gói final evaluation"), code(PACKAGE),
    markdown("""## Kết quả cuối

- `metrics_per_sample_all.csv`: toàn bộ metric BW/EM/MA ở mức segment;
- `summary_all.csv`, `summary_mean_std_all.csv`: kết quả chung và `mean ± SD`;
- `summary_by_record_all.csv`: kết quả theo QTDB record;
- `artifact_ranking.csv`: xếp hạng mô tả theo ΔSNR, PRD, RMSE và cosine;
- `source_manifest.csv`: nguồn và SHA-256 của ba bảng đầu vào;
- `final_metric_overview.png`: so sánh BW/EM/MA × protocol;
- `final_performance_by_noise_level.png`: độ bền theo mức `r`;
- `final_protocol_comparison.png`: độ nhạy giữa protocol 1 và 2;
- `final_metric_distributions.png`: phân phối ΔSNR và PRD;
- `final_snr_input_output_scatter.png`: quan hệ SNR vào–ra;
- `final_record_snr_heatmap.png`: tính ổn định theo record;
- `final_signal_examples_montage.png`: montage hình tín hiệu nếu input có các PNG;
- `qtdb_bw_em_ma_final_evaluation.zip`: gói kết quả cuối cùng."""),
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
