"""Static validation plus a CPU model smoke test for the generated notebooks."""

from __future__ import annotations

import ast
import json
import platform
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "notebooks" / "kaggle_train_qtdb_bw_em_ma.ipynb",
    ROOT / "notebooks" / "colab_evaluate_qtdb_bw_em_ma.ipynb",
    ROOT / "notebooks" / "kaggle_evaluate_qtdb_bw_em_ma.ipynb",
)


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def code_after_heading(notebook: dict, heading: str) -> str:
    for index, cell in enumerate(notebook["cells"][:-1]):
        if cell.get("cell_type") == "markdown" and heading in source(cell):
            next_cell = notebook["cells"][index + 1]
            if next_cell.get("cell_type") != "code":
                raise AssertionError(f"{heading!r} is not followed by code")
            return source(next_cell)
    raise AssertionError(f"Missing heading: {heading}")


loaded = []
for path in NOTEBOOKS:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["cells"]
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = source(cell)
        if text.lstrip().startswith(("%", "!")):
            continue
        ast.parse(text, filename=f"{path.name}:cell")
    full_text = path.read_text(encoding="utf-8")
    for artifact in ("BW", "EM", "MA"):
        assert artifact in full_text
    loaded.append(notebook)
    print(f"Static OK: {path.name} ({len(notebook['cells'])} cells)")


kaggle_evaluation_text = NOTEBOOKS[2].read_text(encoding="utf-8")
assert "google.colab" not in kaggle_evaluation_text
assert "/content/drive" not in kaggle_evaluation_text
assert "Path('/kaggle/input')" in kaggle_evaluation_text
assert "Path('/kaggle/working')" in kaggle_evaluation_text
assert "best_{artifact.lower()}_protocol{protocol}_{CHECKPOINT_TAG}.pth" in kaggle_evaluation_text
print("Kaggle evaluation paths/checkpoint lookup OK")


# Execute the exact generated model/loss definitions on CPU with a small width.
kaggle = loaded[0]
namespace = {
    "np": np,
    "torch": torch,
    "nn": nn,
    "F": F,
    "DEVICE": torch.device("cpu"),
}
exec(code_after_heading(kaggle, "Kiến trúc HNF U-Net"), namespace)
exec(code_after_heading(kaggle, "Conditional DDPM và loss đa miền"), namespace)

config = {
    "train": {
        "feats": 8,
        "batch_size": 2,
        "epochs": 1,
        "lr": 1e-4,
        "lambda_time": 1.0,
        "lambda_freq": 0.1,
    },
    "diffusion": {
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "num_steps": 4,
        "schedule": "quad",
    },
}
model = namespace["ResearchLossDDPM"](
    namespace["UNet1D"](2, 8, 128, 1), config, torch.device("cpu")
)
clean = torch.randn(2, 1, 512)
noisy = clean + 0.1 * torch.randn_like(clean)
components = model.loss_components(clean, noisy)
assert set(("total", "time", "frequency")).issubset(components)
assert all(torch.isfinite(value).all() for value in components.values())
components["total"].backward()
assert any(parameter.grad is not None for parameter in model.parameters())
with torch.inference_mode():
    restored = model.denoising(noisy, use_ddim=True, ddim_steps=2, ddim_eta=0.0, num_shots=1)
assert restored.shape == clean.shape and torch.isfinite(restored).all()
print("CPU forward/backward/DDIM smoke test OK")


# Exercise the exact generated artifact/protocol dataset logic with synthetic
# arrays, including all three records and both channel/half protocols.
dataset_namespace = {
    "np": np,
    "torch": torch,
    "DataLoader": DataLoader,
    "Dataset": Dataset,
    "SEED": 1234,
    "LENGTH": 512,
    "CONFIG": {"train": {"batch_size": 4}},
    "TRAIN_CLEAN": np.random.default_rng(1).normal(size=(12, 512)).astype(np.float32),
    "TRAIN_INDICES": np.arange(8),
    "ARTIFACTS_TO_TRAIN": ("BW", "EM", "MA"),
    "PROTOCOLS_TO_TRAIN": (1, 2),
}
dataset_namespace["NOISE"] = {
    artifact: np.random.default_rng(index + 10).normal(size=(4096, 2)).astype(np.float32)
    for index, artifact in enumerate(("BW", "EM", "MA"))
}
dataset_namespace["NOISE_MIDPOINT"] = {artifact: 2048 for artifact in dataset_namespace["NOISE"]}
exec(code_after_heading(kaggle, "Dataset cố định theo artifact/protocol"), dataset_namespace)
for artifact in ("BW", "EM", "MA"):
    for protocol in (1, 2):
        dataset = dataset_namespace["FixedArtifactDataset"](
            dataset_namespace["TRAIN_CLEAN"], np.arange(8), artifact, protocol
        )
        clean_item, noisy_item = dataset[0]
        assert clean_item.shape == noisy_item.shape == (1, 512)
        assert torch.isfinite(clean_item).all() and torch.isfinite(noisy_item).all()
print("BW/EM/MA dataset and protocol smoke test OK")


# Execute the exact summary/visualization cell with compact synthetic results.
colab_evaluation = loaded[1]
sample_count = 16
sample_axis = np.arange(sample_count)
time_axis = np.arange(512) / 360.0
synthetic_clean = np.stack([
    np.sin(2 * np.pi * (1.0 + index / 30.0) * time_axis)
    for index in sample_axis
]).astype(np.float32)
synthetic_rows = []
synthetic_noisy = {}
r_values = np.linspace(0.2, 1.99, sample_count)

with tempfile.TemporaryDirectory() as temporary_directory:
    output_directory = Path(temporary_directory) / "evaluation"
    output_directory.mkdir()
    for artifact_index, artifact in enumerate(("BW", "EM", "MA")):
        for protocol in (1, 2):
            rng = np.random.default_rng(100 + artifact_index * 10 + protocol)
            noisy = synthetic_clean + rng.normal(
                scale=0.08 + 0.02 * artifact_index,
                size=synthetic_clean.shape,
            ).astype(np.float32)
            prediction = synthetic_clean + rng.normal(
                scale=0.03 + 0.01 * artifact_index,
                size=synthetic_clean.shape,
            ).astype(np.float32)
            synthetic_noisy[(artifact, protocol)] = noisy
            np.save(
                output_directory / f"xhat_{artifact.lower()}_protocol{protocol}_K1.npy",
                prediction,
            )
            for index in sample_axis:
                error = prediction[index] - synthetic_clean[index]
                signal_power = np.sum(synthetic_clean[index] ** 2)
                input_power = np.sum((noisy[index] - synthetic_clean[index]) ** 2)
                output_power = np.sum(error ** 2)
                synthetic_rows.append({
                    "sample_id": f"sample-{index}",
                    "record_id": f"record-{index % 14:02d}",
                    "artifact": artifact,
                    "protocol": protocol,
                    "K": 1,
                    "r": r_values[index],
                    "output_row": index,
                    "ssd": output_power,
                    "rmse": np.sqrt(np.mean(error ** 2)),
                    "mae": np.mean(np.abs(error)),
                    "mad": np.max(np.abs(error)),
                    "prd_repo": 100 * np.sqrt(output_power / np.sum(
                        (prediction[index] - np.mean(synthetic_clean[index])) ** 2
                    )),
                    "prd_standard": 100 * np.sqrt(output_power / np.sum(
                        (synthetic_clean[index] - np.mean(synthetic_clean[index])) ** 2
                    )),
                    "cosine_similarity": np.dot(synthetic_clean[index], prediction[index]) /
                    (np.linalg.norm(synthetic_clean[index]) * np.linalg.norm(prediction[index])),
                    "snr_in_db": 10 * np.log10(signal_power / input_power),
                    "snr_out_db": 10 * np.log10(signal_power / output_power),
                    "snr_improvement_db": 10 * np.log10(input_power / output_power),
                    "inference_time_ms": 2.0,
                })

    visualization_namespace = {
        "np": np, "pd": pd, "plt": plt, "signal": signal,
        "json": json, "zipfile": zipfile, "platform": platform,
        "torch": torch, "scipy": scipy,
        "wfdb": type("WFDBStub", (), {"__version__": "test"})(),
        "display": lambda value: None,
        "OUT": output_directory,
        "per_sample": pd.DataFrame(synthetic_rows),
        "ARTIFACTS_TO_EVALUATE": ("BW", "EM", "MA"),
        "PROTOCOLS": (1, 2), "K_VALUES": (1,), "SEED": 1234,
        "LENGTH": 512, "TARGET_FS": 360,
        "CLEAN": synthetic_clean, "NOISY": synthetic_noisy,
        "selected_records": tuple(f"record-{index:02d}" for index in range(14)),
        "PREPROCESSING_VERSION": "test", "MODEL_NORMALIZATION": "test",
        "BATCH_SIZE": 4, "DDIM_STEPS": 2, "DDIM_ETA": 0.0,
        "CHECKPOINT_INFO": [], "DEVICE": torch.device("cpu"),
        "MAX_TEST_RECORDS": None, "MAX_TEST_SEGMENTS": None,
    }
    exec(code_after_heading(colab_evaluation, "Tổng hợp, biểu đồ"), visualization_namespace)
    expected_figures = {
        "metric_overview_by_artifact_protocol.png",
        "performance_by_noise_level.png",
        "metric_distributions.png",
        "snr_input_output_scatter.png",
        "record_snr_improvement_heatmap.png",
        "signal_examples_bw_protocol1_K1.png",
        "signal_examples_ma_protocol2_K1.png",
    }
    assert expected_figures.issubset({path.name for path in output_directory.glob("*.png")})
    assert (output_directory / "figure_manifest.csv").is_file()
    assert (output_directory.parent / "qtdb_bw_em_ma_evaluation.zip").is_file()
print("Summary and extended visualization smoke test OK")
