"""Execute Figure 2's notebook offline and verify the preview-first contract.

Run with Matplotlib, IPython, Pillow and nbformat installed. --save-outputs keeps
the locally executed inline preview in the deliverable. No PDF/SVG is generated
by these checks; export calls are intercepted. QA PNGs go in a temporary folder.
"""

import argparse
import base64
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import nbformat
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/colab_draw_fig02_architecture.ipynb'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-outputs', action='store_true')
    options = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    qa_dir = Path(tempfile.mkdtemp(prefix='fig02-notebook-qa-'))
    scope, images = {}, []
    current_outputs = []
    original_savefig = Figure.savefig

    def display_image(item):
        assert isinstance(item.data, bytes)
        with PILImage.open(BytesIO(item.data)) as preview:
            assert preview.width > preview.height  # Compact, symmetric U-shaped network.
            preview.verify()
        images.append(item.data)
        current_outputs.append(nbformat.v4.new_output(
            'display_data', data={'image/png': base64.b64encode(item.data).decode('ascii')},
            metadata={}))

    def memory_save(self, target, *args, **kwargs):
        assert isinstance(target, BytesIO), 'Default run must not export files'
        return original_savefig(self, target, *args, **kwargs)

    count = 0
    with patch('IPython.display.display', display_image), \
            patch.object(Figure, 'savefig', memory_save), \
            patch.object(Path, 'mkdir', side_effect=AssertionError('Default run must not create output folders')):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            count += 1
            current_outputs = []
            output = StringIO()
            with redirect_stdout(output):
                exec(compile(cell.source, f'{NOTEBOOK.name}:cell{index}', 'exec'), scope)
            if output.getvalue():
                current_outputs.insert(0, nbformat.v4.new_output(
                    'stream', name='stdout', text=output.getvalue()))
            cell.outputs = current_outputs
            cell.execution_count = count

    assert len(images) == 1
    assert scope['EXPORT_FILES'] is False and scope['DOWNLOAD_ZIP'] is False
    assert scope['MODEL_SPEC'] == {
        'id': 'U4', 'length': 512, 'input_channels': 2, 'output_channels': 1,
        'base_channels': 80, 'embedding_dim': 128, 'attention_heads': 4,
    }
    nodes = scope['fig']._architecture_nodes
    assert len(nodes) == 7  # E1-E3, D1-D3, and the central E4+attention stage.
    for stage in range(1, 4):
        left, right = nodes[f'e{stage}']['center'], nodes[f'd{stage}']['center']
        assert abs(left[0] + right[0] - 22) < 1e-8, 'U arms must be mirrored'
        assert left[1] == right[1], 'Skip connections must stay horizontal'
    base = nodes['e4_attention']['center']
    assert base[0] == 11 and base[1] < nodes['e3']['center'][1]
    (qa_dir / 'preview_color.png').write_bytes(images[0])

    # Export wiring: verify names, dimensions, PNG DPI without writing PDFs.
    saved = []

    def capture_export(self, target, **kwargs):
        saved.append((Path(target).name, tuple(self.get_size_inches()), kwargs))

    class EmptyArchive:
        def __init__(self, *args, **kwargs):
            self.entries = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def write(self, path, arcname):
            self.entries.append(arcname)

    with patch.object(Figure, 'savefig', capture_export), \
            patch.object(Path, 'mkdir'), patch.object(Path, 'write_text'), \
            patch.object(Path, 'stat', return_value=type('Stat', (), {'st_size': 0})()), \
            patch.object(scope['zipfile'], 'ZipFile', EmptyArchive), \
            redirect_stdout(StringIO()):
        scope['export_architecture']()
    assert [entry[0] for entry in saved] == ['2.pdf', '2.svg', '2.png']
    assert saved[-1][2]['dpi'] == 600
    assert all(abs(entry[1][0] - 7.16) < 1e-8 for entry in saved)

    # Rerun all definition cells with the grayscale option before the preview.
    gray_scope = {}
    for cell in notebook.cells:
        if cell.cell_type != 'code' or 'fig = make_architecture_figure()' in cell.source:
            continue
        if 'EXPORT_FILES = False' in cell.source:
            continue
        source = cell.source.replace('GRAYSCALE = False', 'GRAYSCALE = True')
        with redirect_stdout(StringIO()):
            exec(compile(source, 'grayscale_preflight', 'exec'), gray_scope)
    with redirect_stdout(StringIO()):
        figure = gray_scope['make_architecture_figure']()
    figure.savefig(qa_dir / 'preview_grayscale.png', dpi=160)
    plt.close(figure)

    assert scope['SIGNAL_MODE'] == 'illustration' and scope['SIGNALS_NPZ'] is None
    assert scope['fig']._signal_mode == 'illustration'
    assert len(scope['fig'].axes) == 4  # Main diagram + y, x_s, and illustrative Gaussian.
    first, second = scope['io_signals'](), scope['io_signals']()
    np = scope['np']
    for key in ('y', 'x_s', 'epsilon_pred'):
        assert len(first[key]) == 512 and np.isfinite(first[key]).all()
        np.testing.assert_array_equal(first[key], second[key])
    assert 'không phải kết quả' in first['notice']
    scope['SIGNAL_MODE'] = 'provided'
    try:
        scope['io_signals']()
    except ValueError:
        pass
    else:
        raise AssertionError('Missing provided traces must not silently become illustrations')
    # Fixture checks only: these arrays are never labeled as a model result.
    fixture = qa_dir / 'io_fixture_not_research.npz'
    np.savez(fixture, **{key: first[key] for key in ('y', 'x_s', 'epsilon_pred')})
    scope['SIGNALS_NPZ'] = fixture
    supplied = scope['io_signals']()
    for key in ('y', 'x_s', 'epsilon_pred'):
        np.testing.assert_array_equal(supplied[key], first[key])
    np.savez(fixture, y=first['y'], x_s=first['x_s'])
    try:
        scope['io_signals']()
    except ValueError as exc:
        assert 'epsilon_pred' in str(exc)
    else:
        raise AssertionError('Missing prediction must not be synthesized in provided mode')
    scope['SIGNALS_NPZ'] = None
    scope['SIGNAL_MODE'] = 'symbols'
    with redirect_stdout(StringIO()):
        symbols = scope['make_architecture_figure']()
    assert len(symbols.axes) == 1
    plt.close(symbols)
    scope['SIGNAL_MODE'] = 'illustration'
    assert not plt.get_fignums()

    if options.save_outputs:
        notebook.metadata['validation'] = {
            'execution': 'local CPU; all code cells executed in order',
            'colab_run': False,
            'export': 'mocked; no automatic file export/download',
        }
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
    print('PASS: all cells; inline paper-layout preview; source-checked U4 specification; '
          'no default export/download; optional export wiring; grayscale render; '
          'deterministic labeled illustrations; symbols/provided mode guards; '
          'no required data/checkpoint dependency.')
    print('QA PNGs:', qa_dir)


if __name__ == '__main__':
    main()
