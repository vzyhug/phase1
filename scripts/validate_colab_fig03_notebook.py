"""Offline execution/visual-preflight for Figure 3; no GPU or model data needed."""

import argparse
import base64
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import nbformat
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/colab_draw_fig03_blocks.ipynb'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-outputs', action='store_true')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    scope, images, sizes = {}, [], []
    current = []
    qa = Path(tempfile.mkdtemp(prefix='fig03-notebook-qa-'))
    original_savefig = Figure.savefig

    def memory_save(self, target, *args, **kwargs):
        assert isinstance(target, BytesIO), 'Default run must not export files'
        return original_savefig(self, target, *args, **kwargs)

    def display_image(item):
        with PILImage.open(BytesIO(item.data)) as preview:
            sizes.append(preview.size)
            preview.verify()
        images.append(item.data)
        current.append(nbformat.v4.new_output(
            'display_data', data={'image/png': base64.b64encode(item.data).decode('ascii')}, metadata={}))

    count = 0
    with patch('IPython.display.display', display_image), \
            patch.object(Figure, 'savefig', memory_save), \
            patch.object(Path, 'mkdir', side_effect=AssertionError('Default mkdir forbidden')):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            count += 1
            current = []
            output = StringIO()
            try:
                with redirect_stdout(output):
                    exec(compile(cell.source, f'cell{index}', 'exec'), scope)
            except Exception:
                print(output.getvalue())
                raise
            if output.getvalue():
                current.insert(0, nbformat.v4.new_output('stream', name='stdout', text=output.getvalue()))
            cell.outputs, cell.execution_count = current, count
    assert len(images) == 4
    assert all(h > w for w, h in sizes[:3]) and sizes[3][0] > sizes[3][1]
    assert not scope['EXPORT_FILES'] and not scope['DOWNLOAD_ZIP']
    for name, data in zip(('3a', '3b', '3c', 'combined'), images):
        (qa / f'{name}.png').write_bytes(data)

    # Export calls are intercepted: no PDF/SVG authoring or file downloads.
    saved = []
    class EmptyArchive:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def write(self, *args, **kwargs): pass
    with patch.object(Figure, 'savefig', lambda self, target, **kw: saved.append((Path(target).name, kw))), \
            patch.object(Path, 'mkdir'), patch.object(Path, 'write_text'), \
            patch.object(scope['zipfile'], 'ZipFile', EmptyArchive), redirect_stdout(StringIO()):
        scope['export_blocks']()
    assert [entry[0] for entry in saved] == [f'{name}.{ext}'
             for name in ('3', '3a', '3b', '3c') for ext in ('pdf', 'svg', 'png')]
    assert all(entry[1]['dpi'] == 600 for entry in saved if entry[0].endswith('.png'))

    # The independent panels must not accidentally render another component.
    originals = {name: scope[f'draw_panel_{name}'] for name in 'abc'}
    def forbidden(axis): raise AssertionError('Another component was rendered')
    for selected in 'abc':
        for name in 'abc':
            scope[f'draw_panel_{name}'] = originals[name] if name == selected else forbidden
        with redirect_stdout(StringIO()):
            figure = scope['make_blocks_figure'](selected)
        plt.close(figure)
    scope.update({f'draw_panel_{name}': draw for name, draw in originals.items()})

    gray = {}
    for cell in notebook.cells:
        if cell.cell_type != 'code' or cell.source.lstrip().startswith('preview_blocks('):
            continue
        if 'EXPORT_FILES = False' in cell.source:
            continue
        with redirect_stdout(StringIO()):
            exec(cell.source.replace('GRAYSCALE = False', 'GRAYSCALE = True'), gray)
    with redirect_stdout(StringIO()):
        figure = gray['make_blocks_figure']()
    figure.savefig(qa / 'combined_grayscale.png', dpi=160)
    plt.close(figure)
    assert not plt.get_fignums()
    if args.save_outputs:
        notebook.metadata['validation'] = {
            'execution': 'all code cells executed locally on CPU', 'colab_run': False,
            'default': 'four inline previews, no exports/downloads',
            'exports': 'mocked; independent panels and grayscale checked',
        }
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
    print('PASS: all cells; four previews; independent panels; no default export/download; '
          '12 optional export names/DPI; grayscale render.')
    print('QA PNGs:', qa)


if __name__ == '__main__':
    main()
