"""Static and lightweight contract checks for the mixed-noise Kaggle notebook."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_train_proposed_tfcdiff_mixed_3600.ipynb"
REQUIRED_SOURCE_FILES = (
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/main_model.py",
    "src/models/unet_1d.py",
    "src/models/blocks/__init__.py",
    "src/models/blocks/hnf_block.py",
    "src/models/blocks/bridge_block.py",
    "src/models/blocks/attention_1d.py",
)

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert notebook.get("nbformat") == 4
cells = notebook["cells"]
assert len(cells) == 25

for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]), filename=f"cell_{index}")

all_text = "\n".join("".join(cell["source"]) for cell in cells)
required_tokens = (
    "tfcdiff_mixed_bwem_ma",
    "common_qtdb_mixed_bwem_ma_3600_ntype1.npz",
    "n_type",
    "SPLIT_SEED = 1234",
    "REFERENCE_BATCH_SIZE = 128",
    "MICRO_BATCH_SIZE = 4",
    "weight_decay=MODEL_CONFIG['train']['weight_decay']",
    "StepLR(optimizer, step_size=150, gamma=0.1)",
    "clip_grad_norm_(model.parameters(), 1.0)",
    "proposed_tfcdiff_mixed_ntype1_seed2024.npz",
    "last_proposed_tfcdiff_mixed_3600.pth",
    "best_proposed_tfcdiff_mixed_3600.pth",
    "test_used_for_checkpoint_selection",
)
for token in required_tokens:
    assert token in all_text, token

assert "TRAIN_BUNDLE_NAME = 'common_qtdb_bw_3600.npz'" not in all_text
assert "gamma=1.0" not in all_text
assert "lr': 1e-4" not in all_text

# Confirm the embedded source snapshot is byte-identical to the current model source.
source_cell = "".join(cells[5]["source"])
match = re.search(
    r'EMBEDDED_SOURCE_B64 = json\.loads\(r"""(\{.*?\})"""\)',
    source_cell,
    re.S,
)
assert match, "Cannot locate embedded source dictionary"
embedded = json.loads(match.group(1))
assert set(embedded) == set(REQUIRED_SOURCE_FILES)
for relative in REQUIRED_SOURCE_FILES:
    current = (ROOT / relative).read_bytes()
    decoded = base64.b64decode(embedded[relative])
    assert decoded == current, relative
    assert hashlib.sha256(decoded).hexdigest() == hashlib.sha256(current).hexdigest()

# Ensure all required fixed QTDB test records are present in the guardrail.
for record in (
    "sel123", "sel233", "sel302", "sel307", "sel820", "sel853", "sel16420",
    "sel16795", "sele0106", "sele0121", "sel32", "sel49", "sel14046", "sel15814",
):
    assert record in all_text, record

# Execute the data-contract cell against a tiny source-train fixture. This checks
# metadata guards, deterministic 70/30 splitting, IDs, q and record leakage.
with tempfile.TemporaryDirectory() as temporary_directory:
    temporary = Path(temporary_directory)
    input_root = temporary / "input"
    report_dir = temporary / "reports"
    input_root.mkdir(); report_dir.mkdir()
    rng = np.random.RandomState(7)
    source_count = 20
    test_records = np.asarray(sorted({
        "sel123", "sel233", "sel302", "sel307", "sel820", "sel853", "sel16420",
        "sel16795", "sele0106", "sele0121", "sel32", "sel49", "sel14046", "sel15814",
    }))
    bundle_path = input_root / "common_qtdb_mixed_bwem_ma_3600_ntype1.npz"
    np.savez_compressed(
        bundle_path,
        clean_source_train=rng.randn(source_count, 3600, 1).astype(np.float32),
        noisy_source_train=rng.randn(source_count, 3600, 1).astype(np.float32),
        sample_id_source_train=np.asarray([f"train:{i}" for i in range(source_count)]),
        record_id_source_train=np.asarray([f"train_record_{i % 3}" for i in range(source_count)]),
        q_source_train=np.full(source_count, 1.0, dtype=np.float32),
        clean_test=rng.randn(len(test_records), 3600, 1).astype(np.float32),
        noisy_test=rng.randn(len(test_records), 3600, 1).astype(np.float32),
        sample_id_test=np.asarray([f"test:{record}" for record in test_records]),
        record_id_test=test_records,
        q_test=np.full(len(test_records), 1.0, dtype=np.float32),
    )

    def fixture_sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    namespace = {
        "np": np, "pd": pd, "hashlib": hashlib,
        "INPUT_ROOT": input_root, "REPORT_DIR": report_dir,
        "TRAIN_BUNDLE_NAME": bundle_path.name, "MANIFEST_NAME": "mixed_manifest.csv",
        "SPLIT_SEED": 1234, "ALLOW_UNVERIFIED_BUNDLE": True,
        "sha256_file": fixture_sha256_file,
        "display": lambda value: None, "print": lambda *args, **kwargs: None,
    }
    exec(compile("".join(cells[7]["source"]), "data_contract_cell", "exec"), namespace)
    assert len(namespace["CLEAN_TRAIN"]) == 14
    assert len(namespace["CLEAN_VAL"]) == 6
    assert len(namespace["CLEAN_TEST_LOCKED"]) == 14
    assert namespace["protocol_verified"] is False
    assert (report_dir / "split_indices.npz").is_file()
    assert (report_dir / "mixed_manifest.csv").is_file()

print(f"OK: {NOTEBOOK}")
print(f"Cells: {len(cells)}; code cells parse; embedded source matches current src/")
print("Synthetic mixed-data contract: OK (source-train -> deterministic 70/30 + 14-record test)")
