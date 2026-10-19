"""Offline regression checks for the U4 PhysioNet retry fix.

No dataset download, GPU, or real waiting is required. This does not replace
a full training run on Kaggle; it tests the code embedded in the notebook.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / 'kaggle_ablation_u4_all_noise_512_100ep.ipynb'


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    trees = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and not cell.source.lstrip().startswith('%'):
            trees.append(ast.parse(cell.source, filename=f'{NOTEBOOK.name}:cell{index}'))

    retry_nodes = [node for tree in trees for node in tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == 'wfdb_read_with_retry']
    assert len(retry_nodes) == 1
    delays, messages = [], []
    scope = {'time': SimpleNamespace(sleep=delays.append),
             'print': lambda *args, **kwargs: messages.append(args)}
    module = ast.Module(body=retry_nodes, type_ignores=[])
    exec(compile(module, str(NOTEBOOK), 'exec'), scope)
    retry = scope['wfdb_read_with_retry']

    # Verify every WFDB I/O call is wrapped, including noise/annotation reads.
    operations = {'rdrecord': 0, 'rdann': 0, 'get_record_list': 0}
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                assert not (node.func.value.id == 'wfdb'
                            and node.func.attr in operations), 'Unprotected WFDB read'
            if isinstance(node.func, ast.Name) and node.func.id == 'wfdb_read_with_retry':
                operation = node.args[0]
                assert isinstance(operation, ast.Attribute)
                assert isinstance(operation.value, ast.Name) and operation.value.id == 'wfdb'
                operations[operation.attr] += 1
    assert operations == {'rdrecord': 2, 'rdann': 1, 'get_record_list': 1}, operations

    marker = object()
    assert retry(lambda: marker) is marker
    assert not delays

    for text in ('502 Error: Bad Gateway for url: sel46.hea', '429 Error',
                 '500 Error', '503 Error', '504 Error', 'read timed out',
                 'connection reset'):
        calls = []
        delays.clear()

        def temporary_failure(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) <= 2:
                raise RuntimeError(text)
            return marker

        assert retry(temporary_failure, 'sel46', pn_dir='qtdb/1.0.0', channels=[0]) is marker
        assert len(calls) == 3 and delays == [2, 4]
        assert all(call == calls[0] for call in calls), 'Retry changed record/read options'

    calls, delays[:] = [], []
    permanent_gateway_error = RuntimeError('502 Error: Bad Gateway')

    def persistent_failure():
        calls.append(1)
        raise permanent_gateway_error

    try:
        retry(persistent_failure)
    except RuntimeError as exc:
        assert exc is permanent_gateway_error
    else:
        raise AssertionError('Persistent failure must not skip the record or return None')
    assert len(calls) == 5 and delays == [2, 4, 8, 16]

    for text in ('404 Error: Not Found', 'Invalid annotation format'):
        calls, delays[:] = [], []
        original_error = ValueError(text)

        def invalid_data():
            calls.append(1)
            raise original_error

        try:
            retry(invalid_data)
        except ValueError as exc:
            assert exc is original_error
        else:
            raise AssertionError('Non-transient errors must propagate')
        assert len(calls) == 1 and not delays

    print('PASS: notebook format/syntax; 4 protected WFDB calls; transient retry; '
          'unchanged arguments; bounded backoff; no record skipping; fail-fast invalid data.')


if __name__ == '__main__':
    main()
