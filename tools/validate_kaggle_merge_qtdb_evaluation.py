"""Static and synthetic end-to-end validation for the merge notebook."""

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
NOTEBOOK = ROOT / "notebooks" / "kaggle_merge_qtdb_bw_em_ma_evaluation.ipynb"


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4
code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]
              if cell.get("cell_type") == "code"]
for index, source in enumerate(code_cells):
    ast.parse(source, filename=f"{NOTEBOOK.name}:cell{index}")
full_text = NOTEBOOK.read_text(encoding="utf-8")
for required in (
    "metrics_per_sample_all.csv",
    "final_metric_overview.png",
    "final_performance_by_noise_level.png",
    "final_signal_examples_montage.png",
    "qtdb_bw_em_ma_final_evaluation.zip",
):
    assert required in full_text
print(f"Static OK: {NOTEBOOK.name} ({len(notebook['cells'])} cells)")


with tempfile.TemporaryDirectory() as temporary_directory:
    temporary = Path(temporary_directory)
    input_root = temporary / "input"
    output_root = temporary / "working" / "qtdb_bw_em_ma_final_evaluation"
    input_root.mkdir()
    output_root.parent.mkdir()

    sample_count = 20
    r_values = np.linspace(0.2, 1.99, sample_count)
    for artifact_index, artifact in enumerate(("BW", "EM", "MA")):
        artifact_directory = input_root / artifact.lower()
        artifact_directory.mkdir()
        rows = []
        rng = np.random.default_rng(100 + artifact_index)
        for protocol in (1, 2):
            for sample_index in range(sample_count):
                snr_in = 1.0 + 0.4 * sample_index + rng.normal(0, 0.1)
                improvement = 4.0 - artifact_index * 0.6 + protocol * 0.15 + rng.normal(0, 0.2)
                rmse = 0.05 + artifact_index * 0.01 + rng.uniform(0, 0.01)
                rows.append({
                    "sample_id": f"sample-{sample_index:03d}",
                    "record_id": f"record-{sample_index % 14:02d}",
                    "artifact": artifact,
                    "protocol": protocol,
                    "K": 1,
                    "r": r_values[sample_index],
                    "ssd": rmse ** 2 * 512,
                    "rmse": rmse,
                    "mae": rmse * 0.8,
                    "mad": rmse * 2.5,
                    "prd_repo": 10 + artifact_index + rng.uniform(0, 1),
                    "prd_standard": 9 + artifact_index + rng.uniform(0, 1),
                    "cosine_similarity": 0.95 - artifact_index * 0.02 + rng.uniform(0, 0.01),
                    "snr_in_db": snr_in,
                    "snr_out_db": snr_in + improvement,
                    "snr_improvement_db": improvement,
                    "inference_time_ms": 2.0 + artifact_index * 0.1,
                })
        pd.DataFrame(rows).to_csv(artifact_directory / "metrics_per_sample.csv", index=False)

        for protocol in (1, 2):
            fig, axis = plt.subplots(figsize=(3, 2))
            axis.plot(np.sin(np.linspace(0, 8, 100)), color="#4C78A8")
            axis.set_title(f"{artifact}-P{protocol}")
            fig.tight_layout()
            fig.savefig(artifact_directory / f"signal_examples_{artifact.lower()}_protocol{protocol}_K1.png")
            plt.close(fig)

    namespace = {}
    for source in code_cells:
        source = source.replace("Path('/kaggle/input')", f"Path({str(input_root)!r})")
        source = source.replace(
            "Path('/kaggle/working/qtdb_bw_em_ma_final_evaluation')",
            f"Path({str(output_root)!r})",
        )
        exec(compile(source, NOTEBOOK.name, "exec"), namespace)

    expected_outputs = {
        "metrics_per_sample_all.csv",
        "summary_all.csv",
        "summary_mean_std_all.csv",
        "summary_by_record_all.csv",
        "artifact_ranking.csv",
        "source_manifest.csv",
        "final_metric_overview.png",
        "final_performance_by_noise_level.png",
        "final_protocol_comparison.png",
        "final_metric_distributions.png",
        "final_snr_input_output_scatter.png",
        "final_record_snr_heatmap.png",
        "final_signal_examples_montage.png",
        "figure_manifest.csv",
        "run_info.json",
    }
    actual_outputs = {path.name for path in output_root.iterdir() if path.is_file()}
    assert expected_outputs.issubset(actual_outputs)
    combined = pd.read_csv(output_root / "metrics_per_sample_all.csv")
    assert len(combined) == 3 * 2 * sample_count
    assert set(combined.artifact) == {"BW", "EM", "MA"}
    assert (output_root.parent / "qtdb_bw_em_ma_final_evaluation.zip").is_file()
print("Synthetic BW/EM/MA merge, charts, montage, and archive OK")
