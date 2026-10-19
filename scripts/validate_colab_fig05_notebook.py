"""Offline Figure 5 checks. Fixtures are QA-only, NOT experimental predictions.

Run with the project .venv Python. No PhysioNet reads or Colab execution.
"""

import ast
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks/colab_draw_fig05_results.ipynb'
TRAIN = ROOT / 'notebooks/kaggle_ablation_u4_all_noise_512_100ep.ipynb'


def source_classes(source):
    return {node.name: ast.dump(node) for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    notebook = json.loads(NB.read_text(encoding='utf-8'))
    training = json.loads(TRAIN.read_text(encoding='utf-8'))
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for source in cells:
        compile(source, '<cell>', 'exec')
    network_source = next(s for s in cells if 'class HNFBlock' in s)
    diffusion_source = next(s for s in cells if 'class ResearchLossDDPM' in s)
    preprocessing = next(s for s in cells if s.startswith('BEAT_SYMBOLS'))
    assert source_classes(network_source) == source_classes(''.join(training['cells'][6]['source']))
    assert diffusion_source == ''.join(training['cells'][8]['source'])
    assert preprocessing == ''.join(training['cells'][10]['source'])
    assert 'EXPORT_FILES = False' in '\n'.join(cells)
    assert 'files.download' not in '\n'.join(cells)
    assert 'files.upload()' in '\n'.join(cells)

    spec = importlib.util.spec_from_file_location('fig05_runtime', ROOT / 'scripts/fig05_ecg_results.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.LENGTH, runtime.TARGET_FS, runtime.INSERT_OFFSET = 512, 360, 16
    runtime.MAX_UNPADDED = 496
    exec(network_source, runtime.__dict__)
    exec(diffusion_source, runtime.__dict__)
    checkpoint = ROOT / 'checkpoints/mix/best_u4_mix7_protocol1_mixed512_100ep_v1.pth'
    model, provenance = runtime.load_u4_checkpoint(checkpoint, 'ema', torch.device('cpu'))
    assert provenance['checkpoint_epoch_one_based'] == 98
    assert provenance['parameter_count'] == 11506721
    # Shape-only checkpoint smoke test, not a denoising-quality experiment.
    with torch.inference_mode():
        dummy = torch.zeros(1, 1, 512)
        out = model.model(dummy, dummy, torch.ones(1, 1))
        assert out.shape == dummy.shape and torch.isfinite(out).all()
    with patch.object(runtime.torch, 'load', return_value={
            'ablation_id': 'U4', 'artifact': 'BW', 'protocol': 1}):
        try:
            runtime.load_u4_checkpoint(checkpoint, 'ema', torch.device('cpu'))
        except ValueError:
            pass
        else:
            raise AssertionError('Specialist checkpoint incorrectly accepted')

    # Fully synthetic unit-test fixture, never published as experimental output.
    rng = np.random.RandomState(5)
    raw = np.zeros((2, 512), dtype=np.float32)
    raw[0, 16:316] = np.sin(np.arange(300)/18)
    raw[1, 16:266] = np.sin(np.arange(250)/16)
    clean_meta = pd.DataFrame([
        {'sample_id': f'QA_ONLY:{i}', 'record_id': 'QA_ONLY', 'unpadded_length': n}
        for i, n in enumerate((300, 250))])
    noise = {g: rng.randn(4096, 2).astype(np.float32) for g in ('BW', 'EM', 'MA')}
    groups = ('BW', 'EM', 'MA', 'BW+EM', 'BW+MA', 'EM+MA', 'BW+EM+MA')
    clean, noisy, manifest = runtime.make_pilot_pairs(raw, clean_meta, noise, groups, 1.0)
    assert clean.shape == noisy.shape == (14, 512)
    assert set(manifest.noise_channel_zero_based) == {1}
    assert set(manifest.noise_half) == {'second'}
    assert manifest.pair_id.is_unique
    for _, row in manifest.iterrows():
        assert all(v >= 2048 for v in json.loads(row.noise_start_json).values())
    for i in range(len(noisy)):
        assert np.isclose(np.ptp(noisy[i]-clean[i])/np.ptp(clean[i]), 1.0, rtol=1e-5)
    # Check mixture composition independently, including source-channel and half split.
    mixture = sum(noise[g][2048:2560, 1] for g in ('BW', 'EM', 'MA'))
    expected = raw[0] + np.ptp(raw[0])/np.ptp(mixture)*mixture
    assert np.allclose(noisy[12], expected/max(float(np.abs(raw[0]).max()), 1e-8), atol=1e-6)
    metrics = runtime.prediction_metrics(clean, noisy, clean.copy(), manifest)
    assert np.all(metrics.rmse_valid == 0) and np.all(metrics.rmse_full512 == 0)
    altered = clean.copy()
    altered[:, 0] += 1  # outside valid region
    metrics2 = runtime.prediction_metrics(clean, noisy, altered, manifest)
    assert np.all(metrics2.rmse_valid == 0)
    assert np.allclose(metrics2.rmse_full512, 1/np.sqrt(512))
    try:
        runtime.make_pilot_pairs(raw, clean_meta, noise, ('UNKNOWN',), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown noise mask accepted')
    try:
        runtime.prepare_pilot_data(provenance, ('sel100',), 8, Path('DO_NOT_CREATE'))
    except ValueError:
        pass
    else:
        raise AssertionError('Training record incorrectly accepted for pilot')

    qa = Path(tempfile.mkdtemp(prefix='fig05-qa-'))
    for gray in (False, True):
        # Deliberately not a model prediction; fixture image explicitly labelled.
        fixture_prediction = clean + 0.1*(noisy-clean)
        fixture_metrics = runtime.prediction_metrics(clean, noisy, fixture_prediction, manifest)
        figure = runtime.make_waveform_figure(clean, noisy, fixture_prediction, fixture_metrics,
                    'QA_ONLY:0', ('BW', 'EM', 'MA', 'BW+EM+MA'), gray)
        figure.suptitle('QA FIXTURE ONLY — NOT MODEL RESULTS', fontsize=11)
        figure.canvas.draw()
        axes = np.asarray(figure.axes).reshape(4, 3)
        assert len({tuple(ax.get_ylim()) for ax in axes[:, 0]}) == 1
        assert len({tuple(ax.get_ylim()) for ax in axes[:, 2]}) == 1
        assert all(len(ax.lines) == 3 for ax in axes[:, 0])
        assert np.isclose(axes[0, 0].lines[0].get_xdata()[-1], 299/360)
        figure.savefig(qa / f'fixture_{"gray" if gray else "color"}.png', dpi=140)
        plt.close(figure)
    # Real export tested on the labelled fixture into temp only; no PDF authoring.
    figure = runtime.make_waveform_figure(clean, noisy, clean, metrics,
                                           'QA_ONLY:0', ('BW',), False)
    with redirect_stdout(StringIO()):
        runtime.export_results(qa / 'export', figure, clean, noisy, clean, metrics,
                               {'scope': 'QA fixture, NOT research results'})
    plt.close(figure)
    with np.load(qa / 'export/fig05_predictions.npz', allow_pickle=False) as saved:
        assert saved['pair_id'].tolist() == manifest.pair_id.tolist()
        assert np.array_equal(saved['denoised'], clean)
    print('PASS: syntax, exact U4 source/sampler/preprocessing, real EMA strict load + shape,')
    print('held-out/noise split, sum-then-scale MIX7, metrics regions, plotting and export fixtures.')
    print('Fixture images for layout QA only:', qa)
    print('NOT RUN: real PhysioNet data preparation / DDIM quality / Colab end-to-end.')


if __name__ == '__main__':
    main()
