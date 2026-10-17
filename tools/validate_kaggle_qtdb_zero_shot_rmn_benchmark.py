"""Validate the zero-shot RMN notebook with static and synthetic checks."""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_qtdb_zero_shot_rmn_benchmark.ipynb"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def code_after_heading(notebook: dict, heading: str) -> str:
    for index, cell in enumerate(notebook["cells"][:-1]):
        if cell.get("cell_type") == "markdown" and heading in source(cell):
            return source(notebook["cells"][index + 1])
    raise AssertionError(heading)


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    text = source(cell)
    if text.lstrip().startswith("!"):
        continue
    ast.parse(text, filename=f"{NOTEBOOK.name}:cell{index}")
full_text = NOTEBOOK.read_text(encoding="utf-8")
for token in (
    "P1=nv1", "P2=nv2", "ema_model_state_dict",
    "zero_shot_summary_full_rmn.csv", "comparison_matching_subset_summary.csv",
    "zero_shot_snr_by_rmn_composition.png",
):
    assert token in full_text
print(f"Static OK: {NOTEBOOK.name} ({len(notebook['cells'])} cells)")


# Verify the protocol mapping against the repository implementation.
prep_text = (ROOT / "fully_gated_DAE" / "Data_Preparation" /
             "data_preparation_overlapped.py").read_text(encoding="utf-8")
for line in (
    "noise_test_1 = bw_noise_channel2_b",
    "noise_train_1 = bw_noise_channel1_a",
    "noise_test_1 = bw_noise_channel1_b",
    "noise_train_1 = bw_noise_channel2_a",
):
    assert line in prep_text
print("Repository mapping verified: P1=nv1 and P2=nv2")


# Execute the exact metric function on a compact signal probe.
metric_namespace = {"np": np, "pd": pd}
exec(code_after_heading(notebook, "Metric chung"), metric_namespace)
rng = np.random.default_rng(1234)
clean = rng.normal(size=(4, 512)).astype(np.float32)
noisy = clean + 0.2 * rng.normal(size=clean.shape).astype(np.float32)
prediction = clean + 0.05 * rng.normal(size=clean.shape).astype(np.float32)
metrics = metric_namespace["calculate_metrics"](clean, noisy, prediction)
assert len(metrics) == 4 and np.isfinite(metrics.to_numpy()).all()
assert (metrics.SNR_improvement_dB > 0).all()
print("Shared metrics probe OK")


# Exercise aggregation, matching-subset comparison, and plots using synthetic
# per-segment evidence with the same grain as the real benchmark.
with tempfile.TemporaryDirectory() as temporary_directory:
    temporary = Path(temporary_directory)
    input_root = temporary / "input"
    metric_directory = temporary / "current_metrics"
    report_directory = temporary / "reports"
    for directory in (input_root, metric_directory, report_directory):
        directory.mkdir()

    compositions = ["Clean", "BW", "MA", "BW+MA", "EM", "BW+EM", "MA+EM", "BW+MA+EM"]
    zero_frames = []
    for artifact_index, train_artifact in enumerate(("BW", "EM", "MA")):
        rows = []
        for version in (1, 2):
            for segment_index in range(16):
                test_artifact = compositions[segment_index % len(compositions)]
                base = 8.0 - artifact_index * 0.5 + segment_index * 0.01
                rows.append({
                    "model": f"DDPM-{train_artifact} zero-shot RMN",
                    "train_artifact": train_artifact,
                    "noise_version": f"nv{version}", "protocol": version,
                    "noise_scale": 1.0, "test_artifact": test_artifact,
                    "artifact": test_artifact, "subject": f"record-{segment_index % 4}",
                    "segment_index": segment_index, "checkpoint_sha256": "test",
                    "SSD": 10.0 + artifact_index, "MAD_mV": 0.5,
                    "PRD_paper_pct": 40.0 + artifact_index,
                    "PRD_repo_pct": 41.0 + artifact_index, "CosSim": 0.9,
                    "RMSE_mV": 0.1, "MAE_mV": 0.08,
                    "SNR_in_dB": 3.0, "SNR_dB": base,
                    "SNR_improvement_dB": base - 3.0,
                })
        frame = pd.DataFrame(rows)
        frame.to_csv(metric_directory / f"zero_shot_metrics_ddpm_{train_artifact.lower()}.csv", index=False)
        zero_frames.append(frame)

    baseline_rows = []
    for model_index, model in enumerate(("FGDAE q=1", "FGDAE q=2", "FGDAE q=3", "FGDAE q=4")):
        for version in (1, 2):
            for segment_index in range(16):
                artifact = compositions[segment_index % len(compositions)]
                baseline_rows.append({
                    "model": model, "noise_version": f"nv{version}",
                    "segment_index": segment_index, "artifact": artifact,
                    "SSD": 9.0 + model_index, "MAD_mV": 0.45,
                    "PRD_paper_pct": 38.0 + model_index,
                    "PRD_repo_pct": 39.0 + model_index, "CosSim": 0.91,
                    "RMSE_mV": 0.095, "MAE_mV": 0.075,
                    "SNR_dB": 8.5 - model_index * 0.2,
                })
    pd.DataFrame(baseline_rows).to_csv(input_root / "joint_metrics_per_segment.csv", index=False)

    namespace = {
        "np": np, "pd": pd, "plt": plt,
        "hashlib": __import__("hashlib"),
        "INPUT_ROOT": input_root, "METRIC_DIR": metric_directory,
        "REPORT_DIR": report_directory,
        "REQUIRE_ALL_ZERO_SHOT_MODELS": True,
        "VALID_ARTIFACTS": ("BW", "EM", "MA"),
        "RUN_COMPARISON": True,
        "display": lambda value: None,
    }
    exec(code_after_heading(notebook, "Ghép các phiên"), namespace)
    exec(code_after_heading(notebook, "Artifact-specific subset"), namespace)
    exec(code_after_heading(notebook, "Biểu đồ"), namespace)

    for output in (
        "zero_shot_metrics_all.csv",
        "zero_shot_summary_full_rmn.csv",
        "zero_shot_summary_matching_subset.csv",
        "comparison_summary_full_rmn.csv",
        "comparison_matching_subset_summary.csv",
        "zero_shot_full_rmn_comparison.png",
        "zero_shot_snr_by_rmn_composition.png",
        "zero_shot_vs_fgdae_matching_subsets.png",
    ):
        assert (report_directory / output).is_file(), output
print("Synthetic aggregation, subset benchmark, and plots OK")
