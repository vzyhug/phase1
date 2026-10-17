"""Static and lightweight source-level validation for the 3600-point Kaggle notebook."""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import io
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_proposed_conditional_ddpm_3600.ipynb"


def text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def code_after_heading(heading: str) -> str:
    for index, cell in enumerate(notebook["cells"][:-1]):
        if cell.get("cell_type") == "markdown" and heading in text(cell):
            return text(notebook["cells"][index + 1])
    raise AssertionError(heading)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") == "code":
        ast.parse(text(cell), filename=f"{NOTEBOOK.name}:cell{index}")

full_text = NOTEBOOK.read_text(encoding="utf-8")
all_source = "\n".join(text(cell) for cell in notebook["cells"])
required_tokens = (
    "TARGET_EPOCHS = 400",
    "in_channels=2",
    "out_channels=1",
    "per_sample_noisy_maxabs_shared_v1",
    "clean_train",
    "noisy_train",
    "common_manifest.csv",
    "sample_id_test",
    "Auto-detected common bundle",
    "synthesized_from_npz_sample_ids",
    "user_uploaded_common_5model_dataset",
    "ema_model_state_dict",
    "strict=True",
    "proposed_seed2024.npz",
    "latency_ms",
    "metrics_computed_in_model_notebook': False",
)
for token in required_tokens:
    assert token in full_text, token
assert "calculate_metrics" not in full_text
assert "source_roots =" not in full_text
for forbidden in (
    "physionet.org", "NEED_TO_GENERATE_COMMON_DATA", "download_and_extract",
    "clean_windows(", "add_bw(", "qt-database-1.0.0.zip",
):
    assert forbidden not in all_source, forbidden
print(f"Static notebook OK: {len(notebook['cells'])} cells")

# Embedded files must reconstruct byte-for-byte to the current repository.
match = re.search(r'EMBEDDED_SOURCE_B64 = json\.loads\(r"""(\{.*?\})"""\)', all_source)
assert match, "embedded source payload"
embedded = json.loads(match.group(1))
for relative, payload in embedded.items():
    assert base64.b64decode(payload) == (ROOT / relative).read_bytes(), relative
print(f"Embedded source snapshot OK: {len(embedded)} byte-identical files")


# Exercise the single-NPZ common-data path without a separate CSV manifest.
with tempfile.TemporaryDirectory() as temporary:
    temporary_path = Path(temporary)
    input_root = temporary_path / "input"
    report_dir = temporary_path / "reports"
    input_root.mkdir(); report_dir.mkdir()
    rng = np.random.default_rng(7)
    arrays = {
        "clean_train": rng.normal(size=(3, 1, 3600)).astype(np.float32),
        "noisy_train": rng.normal(size=(3, 1, 3600)).astype(np.float32),
        "clean_val": rng.normal(size=(2, 1, 3600)).astype(np.float32),
        "noisy_val": rng.normal(size=(2, 1, 3600)).astype(np.float32),
        "clean_test": rng.normal(size=(2, 1, 3600)).astype(np.float32),
        "noisy_test": rng.normal(size=(2, 1, 3600)).astype(np.float32),
        # Reproduce the Kaggle failure: common producers often store IDs as object.
        "sample_id_test": np.asarray(["test-a", "test-b"], dtype=object),
    }
    np.savez_compressed(input_root / "uploaded-shared-data.npz", **arrays)

    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    namespace = {
        "Path": Path, "np": np, "pd": __import__("pandas"),
        "hashlib": hashlib, "zipfile": zipfile,
        "INPUT_ROOT": input_root, "REPORT_DIR": report_dir,
        "TRAIN_BUNDLE_NAME": "common_qtdb_bw_3600.npz",
        "MANIFEST_NAME": "common_manifest.csv",
        "EXPECTED_PROTOCOL": None, "sha256_file": sha256_file,
        "display": lambda value: None,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code_after_heading("Nạp common tensors"), namespace)
    assert namespace["CLEAN_TRAIN"].shape == (3, 1, 3600)
    assert namespace["TRAIN_MANIFEST"].sample_id.tolist()[0] == "train:00000000"
    assert namespace["manifest_origin"] == "synthesized_from_npz_sample_ids"
    assert (report_dir / "resolved_common_manifest.csv").is_file()
    namespace["train_bundle"].close()
print("Single common-NPZ input contract OK")


# Verify that the unchanged repository architecture supports 3600 exactly.
sys.path.insert(0, str(ROOT))
from src.models.main_model import DDPM, stft_loss  # noqa: E402
from src.models.unet_1d import UNet1D  # noqa: E402

config = {
    "train": {"feats": 8, "batch_size": 1, "epochs": 1, "lr": 1e-4,
              "lambda_time": 1.0, "lambda_freq": 0.1},
    "diffusion": {"beta_start": 1e-4, "beta_end": 0.5,
                  "num_steps": 50, "schedule": "quad"},
}
device = torch.device("cpu")
base = UNet1D(in_channels=2, base_channels=8, emb_dim=128, out_channels=1)
model = DDPM(base, config, device).to(device).eval()
probe = torch.zeros(1, 1, 3600)
alpha = torch.full((1, 1), 0.8)
with torch.no_grad():
    output = model.model(probe, probe, alpha)
    spectral = stft_loss(output, probe, n_fft=128, hop_length=64)
assert output.shape == probe.shape
assert torch.isfinite(output).all() and torch.isfinite(spectral)

clone = DDPM(
    UNet1D(in_channels=2, base_channels=8, emb_dim=128, out_channels=1),
    config,
    device,
)
clone.load_state_dict(model.state_dict(), strict=True)
print("Repository model probe OK: 3600 -> 1800 -> 900 -> 450 -> 3600; STFT and strict load OK")


# Check the noisy-only shared normalization and inverse transform contract.
rng = np.random.default_rng(2024)
clean = rng.normal(size=(3, 1, 3600)).astype(np.float32)
noisy = clean + 0.2 * rng.normal(size=clean.shape).astype(np.float32)
scale = np.max(np.abs(noisy), axis=(1, 2), keepdims=True).astype(np.float32)
clean_norm, noisy_norm = clean / scale, noisy / scale
prediction_norm = clean_norm.copy()
prediction = prediction_norm * scale
assert scale.shape == (3, 1, 1)
assert np.allclose(prediction, clean)
assert np.isfinite(clean_norm).all() and np.isfinite(noisy_norm).all()
print("Noisy-derived normalization/inverse contract OK")
