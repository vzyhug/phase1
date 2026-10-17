"""Static and synthetic checks for the fresh QTDB BW+EM+MA 3600 notebook."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TARGET = ROOT / "notebooks" / "kaggle_train_qtdb_bw_em_ma.ipynb"
SOURCE = ROOT / "notebooks" / "kaggle_train_proposed_research_mixed_3600.ipynb"


def source(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


target = json.loads(TARGET.read_text(encoding="utf-8"))
prior = json.loads(SOURCE.read_text(encoding="utf-8"))
assert target["nbformat"] == 4
for index, cell in enumerate(target["cells"]):
    if cell["cell_type"] == "code" and not source(target, index).lstrip().startswith("%"):
        ast.parse(source(target, index), filename=f"cell_{index}")

config = source(target, 4)
data = source(target, 8)
resume = source(target, 14)
assert "START_FROM_SCRATCH = True" in config
assert "REQUIRE_RESUME_CHECKPOINT = False" in config
assert "'input_length': 3600" in config
assert "resume_path = None if START_FROM_SCRATCH else resolve_resume_checkpoint()" in resume
assert "make_three_artifact_pairs(CLEAN_TRAIN" in data
assert "make_three_artifact_pairs(CLEAN_VAL" in data
assert "NOISE_TRAIN[artifact] = np.ascontiguousarray(values[:midpoint, 0])" in data
assert "CLEAN_TEST_LOCKED = as_ncl3600(train_bundle['clean_test']" in data
assert "NOISY_TEST_LOCKED = as_ncl3600(train_bundle['noisy_test']" in data
assert "'generated_train_mixture': True" in data
assert source(target, 7) == source(prior, 6), "U-Net source changed"
assert source(target, 9) == source(prior, 8), "DDPM/loss source changed"

tree = ast.parse(data)
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "make_three_artifact_pairs")
compiled = compile(ast.Module(body=[function], type_ignores=[]), "mixed_data", "exec")
with TemporaryDirectory() as folder:
    noise = {
        "BW": np.linspace(-1, 1, 7300, dtype=np.float32),
        "EM": np.sin(np.linspace(0, 100, 7300)).astype(np.float32),
        "MA": np.cos(np.linspace(0, 200, 7300)).astype(np.float32),
    }
    env = {
        "np": np, "pd": pd, "tqdm": lambda values, **kwargs: values,
        "REPORT_DIR": Path(folder), "MIX_LENGTH": 3600, "NOISE_TRAIN": noise,
    }
    exec(compiled, env)
    clean = np.stack([np.linspace(-.5, .7, 3600, dtype=np.float32),
                      np.linspace(.2, 1.1, 3600, dtype=np.float32)])[:, None, :]
    q = np.array([.2, 1.99], dtype=np.float32)
    noisy = env["make_three_artifact_pairs"](clean, q, 1234, "smoke")
    audit = pd.read_csv(Path(folder) / "smoke_mix_manifest.csv")
    assert noisy.shape == (2, 1, 3600) and np.isfinite(noisy).all()
    assert (audit[["bw_start", "em_start", "ma_start"]] >= 0).all().all()
    for index, row in audit.iterrows():
        bw = noise["BW"][int(row.bw_start):int(row.bw_start) + 3600]
        em = noise["EM"][int(row.em_start):int(row.em_start) + 3600]
        ma = noise["MA"][int(row.ma_start):int(row.ma_start) + 3600]
        mixed = (row.a1 * bw + (100 - row.a1) *
                 (row.a2 * em + (100 - row.a2) * ma)) / 100.0
        expected = clean[index, 0] + q[index] * np.ptp(clean[index, 0]) / np.ptp(mixed) * mixed
        np.testing.assert_allclose(noisy[index, 0], expected, rtol=1e-6, atol=1e-5)

from src.models.unet_1d import UNet1D
small_model = UNet1D(in_channels=2, base_channels=8, emb_dim=128, out_channels=1)
with torch.no_grad():
    output = small_model(torch.zeros(1, 1, 3600), torch.zeros(1, 1, 3600), torch.ones(1))
assert output.shape == (1, 1, 3600)
print("OK: fresh notebook syntax, exact 3-artifact mixing formula, fixed test, 3600-point model output")
