"""Portable Figure 5 helpers, embedded verbatim in the Colab notebook.

No model training, no fabricated experimental waveforms, no automatic download.
"""

import hashlib
import json
import math
import platform
import random
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import torch
import wfdb
from IPython.display import Image, display


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_array(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode('ascii'))
    digest.update(str(values.dtype).encode('ascii'))
    digest.update(values.tobytes())
    return digest.hexdigest()


def wfdb_read_with_retry(operation, *args, **kwargs):
    for attempt in range(5):
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            transient = any(token in str(exc).lower() for token in (
                '429', '500 error', '502', '503', '504', 'bad gateway',
                'timeout', 'timed out', 'connection error', 'connection reset',
                'temporarily unavailable',
            ))
            if not transient or attempt == 4:
                raise
            delay = min(2 ** (attempt + 1), 16)
            print(f'PhysioNet: retry {attempt + 1}/4 sau {delay}s: {exc}')
            time.sleep(delay)


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_file():
        return path
    local = Path('checkpoints/mix') / path.name
    if local.is_file():
        return local
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError(f'Không thấy checkpoint: {path}') from None
    print('Chọn best_u4_mix7_protocol1_mixed512_100ep_v1.pth; không cần tải src.')
    uploaded = files.upload()
    if path.is_file():
        return path
    matches = [Path(name) for name in uploaded if Path(name).name == path.name]
    if len(matches) != 1:
        raise FileNotFoundError(f'Cần đúng checkpoint {path.name}; sửa CHECKPOINT_PATH nếu đổi tên.')
    return matches[0]


def load_u4_checkpoint(path, weights_kind, device):
    # Only load the tensor/basic-container checkpoint format used by this project.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'ablation_id': 'U4', 'artifact': 'MIX7', 'protocol': 1,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f'Checkpoint không đúng {name}={value}: {payload.get(name)}')
    config = payload['config']
    checks = {
        'model_normalization': 'per_segment_clean_maxabs_shared_with_noisy_v1',
        'preprocessing_version': 'qtdb_pu1_p_and_beat_40ms_reflect100ms_360hz_endpoints_pad16_v2',
        'data_protocol': 'qtdb_nstdb_balanced_7_nonempty_masks_512_v1',
        'loss_version': 'l1_mean_relative_stft_snr_weighted_continuous_alpha_v1',
    }
    for name, value in checks.items():
        if config.get(name) != value:
            raise ValueError(f'Khác giao thức {name}: {config.get(name)}')
    if config.get('architecture') != {
        'id': 'U4', 'feature_block': 'hnf_multiscale_halfnorm_residual',
        'timestep': 'sinusoidal_film_scale_shift', 'attention': 'self_attention_4head',
    } or config['train']['feats'] != 80:
        raise ValueError('Cấu hình kiến trúc không phải U4 gốc.')
    if not set(payload['train_records']).isdisjoint(payload['test_records']):
        raise ValueError('Train/test records bị trùng.')
    if weights_kind not in ('ema', 'raw'):
        raise ValueError("WEIGHTS_KIND phải là 'ema' hoặc 'raw'")
    state_key = 'ema_model_state_dict' if weights_kind == 'ema' else 'state_dict'
    if state_key not in payload:
        raise KeyError(f'Checkpoint thiếu {state_key}; không tự chuyển loại trọng số.')
    model = ResearchLossDDPM(UNet1D(2, 80, 128, 1), config, device).to(device)
    model.load_state_dict(payload[state_key], strict=True)
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    if count != payload['parameter_count']:
        raise ValueError('Số tham số không khớp checkpoint.')
    provenance = {
        'checkpoint_name': Path(path).name, 'checkpoint_sha256': sha256_file(path),
        'weights': weights_kind, 'checkpoint_epoch_one_based': int(payload['epoch']) + 1,
        'best_validation_loss_raw_model': float(payload['best_validation_loss']),
        'parameter_count': count, 'config': config,
        'train_records': list(payload['train_records']),
        'test_records': list(payload['test_records']),
        'training_data_fingerprint': payload['data_fingerprint'],
        'torch': torch.__version__, 'numpy': np.__version__,
        'wfdb': wfdb.__version__, 'python': platform.python_version(),
        'device': str(device),
    }
    print(f"U4 MIX7 | epoch {provenance['checkpoint_epoch_one_based']} | "
          f"{weights_kind.upper()} | {count:,} tham số | {device}")
    print('Best validation loss (raw model):', provenance['best_validation_loss_raw_model'])
    return model, provenance


def prepare_pilot_data(provenance, record_names, max_beats_per_record, cache_dir):
    if not record_names or len(set(record_names)) != len(record_names):
        raise ValueError('Cần các record test không trùng nhau.')
    if not set(record_names).issubset(provenance['test_records']):
        raise ValueError('Chỉ được chọn record test giữ riêng trong checkpoint.')
    if max_beats_per_record < 1:
        raise ValueError('MAX_BEATS_PER_RECORD phải >= 1.')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean_parts, metadata, audits = [], [], []
    for record_name in record_names:
        cache = cache_dir / f'{record_name}_clean512_v2.npz'
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                beats = saved['clean']
                rows = json.loads(saved['metadata_json'].item())
                audit = json.loads(saved['audit_json'].item())
        else:
            # This is the same extraction function as in the U4 training notebook.
            beats, rows, audit = clean_segments_for_record(record_name)
            if not beats:
                raise RuntimeError(f'{record_name}: không có đoạn hợp lệ; không bỏ qua record.')
            beats = np.stack(beats).astype(np.float32)
            np.savez_compressed(cache, clean=beats, metadata_json=json.dumps(rows),
                                audit_json=json.dumps(audit))
        if beats.shape != (len(rows), LENGTH) or not np.isfinite(beats).all():
            raise ValueError(f'Cache clean không hợp lệ: {cache}')
        if any(row['record_id'] != record_name for row in rows):
            raise ValueError(f'Cache sai record: {cache}')
        take = min(max_beats_per_record, len(beats))
        # Fixed first accepted beats, selected BEFORE inference, not by model quality.
        clean_parts.append(beats[:take])
        metadata.extend(rows[:take])
        audits.append({**audit, 'selected_for_pilot': take})
        print(f'{record_name}: dùng {take}/{len(beats)} đoạn hợp lệ đầu tiên')
    clean = np.concatenate(clean_parts).astype(np.float32)
    meta = pd.DataFrame(metadata)
    if not meta.sample_id.is_unique:
        raise ValueError('sample_id clean bị trùng.')
    noise = {}
    for artifact in ('BW', 'EM', 'MA'):
        cache = cache_dir / f'{artifact.lower()}_360.npy'
        if cache.is_file():
            values = np.load(cache, allow_pickle=False)
        else:
            record = wfdb_read_with_retry(wfdb.rdrecord, artifact.lower(), pn_dir=PHYSIONET_NST)
            values = record.p_signal.astype(np.float32)
            if int(record.fs) != TARGET_FS:
                values = signal.resample(values, int(round(
                    len(values) * TARGET_FS / float(record.fs))), axis=0).astype(np.float32)
            np.save(cache, values)
        if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
            raise ValueError(f'NSTDB không hợp lệ: {artifact}')
        actual_hash = sha_array(values)
        expected_hash = provenance['training_data_fingerprint'][f'{artifact.lower()}_noise']
        if actual_hash != expected_hash:
            raise ValueError(f'{artifact}: fingerprint khác NSTDB khi train; kiểm tra cache/nguồn dữ liệu.')
        noise[artifact] = values
    return clean, meta, noise, pd.DataFrame(audits)


def make_pilot_pairs(clean_raw, clean_meta, noise, groups, level_r):
    allowed = ('BW', 'EM', 'MA', 'BW+EM', 'BW+MA', 'EM+MA', 'BW+EM+MA')
    if not groups or len(set(groups)) != len(groups) or not set(groups).issubset(allowed):
        raise ValueError('NOISE_GROUPS phải là các tổ hợp khác nhau thuộc MIX7.')
    if not np.isfinite(level_r) or not 0.2 <= level_r <= 1.99:
        raise ValueError('NOISE_LEVEL_R phải nằm trong [0.2, 1.99]. Đây không phải dB.')
    clean_list, noisy_list, rows = [], [], []
    for group in groups:
        for index, raw in enumerate(clean_raw):
            patches, starts = [], {}
            for artifact in group.split('+'):
                values = noise[artifact]
                midpoint = len(values) // 2
                # Protocol 1: opposite channel and second half, never the train patches.
                heldout = values[midpoint:, 1]
                count = len(heldout) // LENGTH
                if not count:
                    raise ValueError(f'{artifact}: nửa test quá ngắn.')
                start = (index % count) * LENGTH
                patches.append(heldout[start:start + LENGTH])
                starts[artifact] = midpoint + start
            mixture = np.sum(np.stack(patches), axis=0)
            if np.ptp(raw) <= 0 or np.ptp(mixture) <= 0:
                raise ValueError('Clean/nhiễu có biên độ bằng 0; không bỏ qua mẫu.')
            noisy_raw = raw + float(level_r) * float(np.ptp(raw)) / float(np.ptp(mixture)) * mixture
            scale = max(float(np.max(np.abs(raw))), 1e-8)
            clean_list.append((raw / scale).astype(np.float32))
            noisy_list.append((noisy_raw / scale).astype(np.float32))
            row = clean_meta.iloc[index].to_dict()
            row.update({
                'pair_id': f"{group}|{row['sample_id']}", 'noise_group': group,
                'split': 'test_pilot', 'protocol': 1, 'noise_channel_zero_based': 1,
                'noise_half': 'second', 'noise_start_json': json.dumps(starts, sort_keys=True),
                'noise_level_r': float(level_r), 'clean_maxabs_scale': scale,
                'sampling_rate': TARGET_FS, 'insert_offset': INSERT_OFFSET,
            })
            rows.append(row)
    clean, noisy = np.stack(clean_list), np.stack(noisy_list)
    manifest = pd.DataFrame(rows)
    if not manifest.pair_id.is_unique or clean.shape != noisy.shape:
        raise ValueError('Các cặp bị trùng hoặc khác shape.')
    if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError('Có giá trị không hữu hạn trong input.')
    return clean, noisy, manifest


def prediction_metrics(clean, noisy, prediction, manifest):
    if clean.shape != noisy.shape or clean.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Prediction không hữu hạn hoặc không khớp input.')
    rows = []
    for i, meta in manifest.iterrows():
        row = meta.to_dict()
        regions = {'full512': slice(None), 'valid': slice(
            int(meta.insert_offset), int(meta.insert_offset + meta.unpadded_length))}
        for name, region in regions.items():
            x, y, p = [a[i, region].astype(np.float64) for a in (clean, noisy, prediction)]
            energy, err_in, err_out = np.sum(x*x), np.sum((x-y)**2), np.sum((x-p)**2)
            floor = 1e-12
            row[f'rmse_{name}'] = float(np.sqrt(np.mean((x-p)**2)))
            row[f'snr_in_db_{name}'] = float(10*np.log10(max(energy, floor)/max(err_in, floor)))
            row[f'snr_out_db_{name}'] = float(10*np.log10(max(energy, floor)/max(err_out, floor)))
            row[f'delta_snr_db_{name}'] = row[f'snr_out_db_{name}'] - row[f'snr_in_db_{name}']
        rows.append(row)
    return pd.DataFrame(rows)


def infer_pilot(model, clean, noisy, manifest, provenance, batch_size, steps, eta, shots, seed, device):
    if not 1 <= steps <= model.num_steps or batch_size < 1 or shots < 1 or eta < 0:
        raise ValueError('Cấu hình sampler không hợp lệ.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    outputs = []
    tick = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(noisy), batch_size):
            batch = torch.from_numpy(noisy[start:start + batch_size, None, :]).to(device)
            pred = model.denoising(batch, use_ddim=True, ddim_steps=steps,
                                   ddim_eta=eta, num_shots=shots)
            if pred.shape != batch.shape or not torch.isfinite(pred).all():
                raise ValueError(f'Prediction không hợp lệ tại batch {start}.')
            outputs.append(pred[:, 0].cpu().numpy())
            print(f'DDIM: {min(start + batch_size, len(noisy))}/{len(noisy)} cặp')
    prediction = np.concatenate(outputs).astype(np.float32)
    tau = [int(np.floor((model.num_steps / steps**2) * i**2)) for i in range(steps+1)]
    provenance.update({
        'sampler': 'original_U4_quadratic_tau_DDIM', 'ddim_steps': steps,
        'ddim_eta': eta, 'shots': shots, 'nfe_per_sample': steps*shots,
        'tau': tau, 'inference_seed': seed, 'inference_batch_size': batch_size,
        'determinism_note': 'Fixed seed and batch/order; CPU/GPU or version changes may differ.',
        'pilot_pair_count': len(noisy), 'pilot_clean_count': manifest.sample_id.nunique(),
        'selected_records': list(manifest.record_id.unique()),
        'selection_rule': 'First accepted beats per specified held-out record, before inference.',
        'noise_level_r': float(manifest.noise_level_r.iloc[0]),
        'pilot_wall_seconds': time.perf_counter()-tick,
        'scope': 'qualitative held-out pilot, NOT the full-test benchmark',
        'clean_sha256': sha_array(clean), 'noisy_sha256': sha_array(noisy),
        'prediction_sha256': sha_array(prediction),
    })
    return prediction


def make_waveform_figure(clean, noisy, prediction, metrics, sample_id, groups, grayscale=False):
    indices = []
    for group in groups:
        found = metrics.index[(metrics.sample_id == sample_id) & (metrics.noise_group == group)]
        if len(found) != 1:
            raise ValueError(f'Cần đúng một cặp: {sample_id}/{group}')
        indices.append(int(found[0]))
    fig, axes = plt.subplots(len(indices), 3, figsize=(10.2, 2.35*len(indices)),
                             squeeze=False, gridspec_kw={'width_ratios': [2.1, 1.15, 1.15]},
                             layout='constrained')
    colors = {'clean': '#222222', 'noisy': '#9b9b9b' if grayscale else '#b78d31',
              'pred': '#555555' if grayscale else '#156c9c'}
    valid_arrays, errors = [], []
    for i in indices:
        row = metrics.loc[i]
        sl = slice(int(row.insert_offset), int(row.insert_offset + row.unpadded_length))
        valid_arrays.extend([a[i, sl] for a in (clean, noisy, prediction)])
        errors.append(prediction[i, sl]-clean[i, sl])
    low = min(float(a.min()) for a in valid_arrays)
    high = max(float(a.max()) for a in valid_arrays)
    margin = max((high-low)*0.08, 0.05)
    error_lim = max(max(float(np.max(np.abs(e))) for e in errors)*1.1, 0.05)
    for r, i in enumerate(indices):
        row = metrics.loc[i]
        offset, count = int(row.insert_offset), int(row.unpadded_length)
        sl = slice(offset, offset+count)
        t = np.arange(count)/float(row.sampling_rate)
        x, y, p = [a[i, sl] for a in (clean, noisy, prediction)]
        # Zoom is anchored to reference absolute peak, NOT to the nicest prediction.
        anchor = int(np.argmax(np.abs(x)))
        half = int(round(0.12*row.sampling_rate))
        a, b = max(0, anchor-half), min(count, anchor+half+1)
        for ax in axes[r, :2]:
            ax.plot(t, y, color=colors['noisy'], lw=0.85, alpha=0.85, label='Noisy')
            ax.plot(t, x, color=colors['clean'], lw=1.1, label='Reference')
            ax.plot(t, p, color=colors['pred'], lw=1.0, ls='--', label='Restored')
            ax.set_ylim(low-margin, high+margin)
        axes[r, 0].axvspan(t[a], t[b-1], color='#777777', alpha=0.1, lw=0)
        axes[r, 1].set_xlim(t[a], t[b-1])
        axes[r, 2].plot(t, p-x, color=colors['pred'], lw=1)
        axes[r, 2].axhline(0, color='#666666', lw=0.65, ls=':')
        axes[r, 2].set_ylim(-error_lim, error_lim)
        axes[r, 0].set_ylabel(f"{row.noise_group}\nNormalized amplitude", fontsize=9)
        axes[r, 0].set_title(
            f"SNR in/out: {row.snr_in_db_valid:.2f}/{row.snr_out_db_valid:.2f} dB",
            fontsize=9, loc='left')
        axes[r, 1].set_title('Reference-peak zoom (±120 ms)', fontsize=8)
        axes[r, 2].set_title(f"Error: restored − reference\nRMSE = {row.rmse_valid:.4f}", fontsize=9)
        for ax in axes[r]:
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.grid(alpha=0.15)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)
    axes[0, 0].legend(loc='upper right', fontsize=8, frameon=False, ncol=3)
    fig.suptitle(f"Held-out pilot | {sample_id} | r={metrics.loc[indices[0], 'noise_level_r']:.2f}\n"
                 'Valid region only (padding removed); normalized amplitude, not mV', fontsize=10)
    return fig


def show_figure(fig, dpi=150):
    stream = BytesIO()
    fig.savefig(stream, format='png', dpi=dpi, facecolor='white')
    display(Image(data=stream.getvalue()))
    plt.close(fig)


def export_results(directory, fig, clean, noisy, prediction, metrics, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / '5.png', dpi=600, facecolor='white')
    fig.savefig(directory / '5.svg', facecolor='white')
    np.savez_compressed(directory / 'fig05_predictions.npz', clean=clean,
                        noisy=noisy, denoised=prediction,
                        pair_id=metrics.pair_id.to_numpy(dtype=str))
    metrics.to_csv(directory / 'fig05_metrics_manifest.csv', index=False)
    (directory / 'fig05_provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Đã lưu (không tự tải xuống):', directory.resolve())
