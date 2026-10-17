"""Static validation for the generated cross-noise Kaggle notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "colab_evaluate_qtdb_bw_em_ma.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_cross_noise_generalization.ipynb"


def body(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    target = json.loads(TARGET.read_text(encoding="utf-8"))

    assert target["nbformat"] == 4
    assert len(target["cells"]) == 22
    assert [cell["cell_type"] for cell in target["cells"]] == [
        cell["cell_type"] for cell in source["cells"]
    ]

    # Architecture, diffusion, clean segmentation, and test-data loading must
    # remain byte-for-byte identical to the validated source notebook.
    for index in (6, 8, 12, 14):
        assert body(target, index) == body(source, index), index

    code = "\n".join(
        body(target, index)
        for index, cell in enumerate(target["cells"])
        if cell["cell_type"] == "code"
    )
    required = (
        "MODEL_ARTIFACTS = ('BW', 'EM', 'MA')",
        "TEST_ARTIFACTS = ('BW', 'EM', 'MA')",
        "for model_artifact in MODEL_ARTIFACTS",
        "for test_artifact in TEST_ARTIFACTS",
        "model_key = (model_artifact, protocol)",
        "test_key = (test_artifact, protocol)",
        "common_random_numbers_across_models",
        "cross_noise_matrix.csv",
        "generalization_by_model.csv",
        "snr_generalization_drop_db",
        "prd_generalization_increase_pct_point",
        "cosine_generalization_drop",
    )
    missing = [marker for marker in required if marker not in code]
    assert not missing, missing

    forbidden = (
        "google.colab",
        "/content/drive",
        "MODELS[(artifact, protocol)]",
        "NOISY[(artifact, protocol)]",
    )
    present = [marker for marker in forbidden if marker in code]
    assert not present, present

    # Compile every Python cell except the one containing the IPython %pip magic.
    for index, cell in enumerate(target["cells"]):
        if cell["cell_type"] != "code" or index == 2:
            continue
        ast.parse(body(target, index), filename=f"cell_{index}")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []

    print(
        "Validated:", TARGET,
        "| cells=22 | unchanged core cells=4 | Python cells compile | 3x3 markers OK"
    )


if __name__ == "__main__":
    main()
