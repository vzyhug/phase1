"""Build the Kaggle variant of the QTDB BW/EM/MA evaluation notebook.

The Colab notebook remains the source of truth for the model, preprocessing,
inference, and metric cells.  This builder only replaces platform-specific
setup, paths, and instructions so the two evaluations stay methodologically
identical.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "colab_evaluate_qtdb_bw_em_ma.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_evaluate_qtdb_bw_em_ma.ipynb"


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def build() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    output = copy.deepcopy(notebook)
    cells = output["cells"]

    if len(cells) != 22:
        raise RuntimeError(f"Unexpected source notebook structure: {len(cells)} cells")

    set_source(
        cells[0],
        """# Kaggle evaluation — QTDB với BW, EM và MA

Notebook đánh giá các checkpoint trên cùng 14 QTDB test records, giữ nguyên toàn bộ preprocessing, DDIM và metric của bản Colab. Mỗi artifact dùng đúng nửa sau và kênh đối nghịch với phần training của protocol tương ứng.

**Thiết lập Kaggle:** bật GPU và Internet, thêm dataset chứa các file `best_<artifact>_protocol<1|2>_norm_loss_v2.pth` bằng **Add Input**, rồi chọn artifact trong cell cấu hình. Mặc định chạy full test, DDIM 50 bước, `eta=0`, 1-shot.

Nếu một artifact mất khoảng 3 giờ, nên chạy ba Kaggle version riêng bằng cách lần lượt đặt `ARTIFACTS_TO_EVALUATE = ('BW',)`, `('EM',)` và `('MA',)`. Không cần sửa các cell khác.
""",
    )
    set_source(cells[3], "## 2. Cấu hình Kaggle và thư mục kết quả\n")

    config = "".join(cells[4]["source"])
    colab_prefix = "from google.colab import drive\ndrive.mount('/content/drive')\n\n"
    if not config.startswith(colab_prefix):
        raise RuntimeError("Colab mount block was not found in the expected setup cell")
    config = config.removeprefix(colab_prefix)
    config = config.replace(
        "CHECKPOINT_DIR = Path('/content/drive/MyDrive/qtdb_bw_em_ma_training/checkpoints')\n"
        "OUT = Path('/content/drive/MyDrive/qtdb_bw_em_ma_evaluation')\n"
        "OUT.mkdir(parents=True, exist_ok=True)",
        "CHECKPOINT_ROOT = Path('/kaggle/input')\n"
        "RUN_SLUG = '_'.join(artifact.lower() for artifact in ARTIFACTS_TO_EVALUATE)\n"
        "OUT = Path('/kaggle/working') / f'qtdb_{RUN_SLUG}_evaluation'\n"
        "OUT.mkdir(parents=True, exist_ok=True)",
    )
    config = config.replace(
        "raise RuntimeError('Enable a GPU runtime in Colab before evaluation.')",
        "raise RuntimeError('Enable GPU in Kaggle: Settings → Accelerator → GPU.')",
    )
    config = config.replace(
        "print('GPU:', torch.cuda.get_device_name(0), '| output:', OUT)",
        "print('GPU:', torch.cuda.get_device_name(0), '| checkpoints:', CHECKPOINT_ROOT, '| output:', OUT)",
    )
    if "google.colab" in config or "/content/drive" in config:
        raise RuntimeError("A Colab-specific dependency remains in the Kaggle setup cell")
    set_source(cells[4], config)

    load_models = "".join(cells[10]["source"])
    expected = "matches = sorted(CHECKPOINT_DIR.rglob(filename))"
    if expected not in load_models:
        raise RuntimeError("Expected checkpoint lookup was not found")
    load_models = load_models.replace(
        expected, "matches = sorted(CHECKPOINT_ROOT.rglob(filename))"
    )
    set_source(cells[10], load_models)

    output_code = "".join(cells[20]["source"])
    output_code = output_code.replace(
        "archive_path = OUT.parent / 'qtdb_bw_em_ma_evaluation.zip'",
        "archive_path = OUT.parent / f'{OUT.name}.zip'",
    )
    set_source(cells[20], output_code)

    set_source(
        cells[21],
        """## Output Kaggle

- `metrics_per_sample.csv`: metric từng segment;
- `summary.csv`, `summary_mean_std.csv`: kết quả theo artifact/protocol/K;
- `summary_by_record.csv`: kết quả theo QTDB record;
- `manifest.csv`: provenance của clean/noise pair;
- `xhat_<artifact>_protocol<P>_K<K>.npy`: tín hiệu khôi phục;
- `metric_overview_by_artifact_protocol.png`: bốn metric chính theo nhiễu/protocol;
- `performance_by_noise_level.png`: độ bền theo các nhóm mức nhiễu `r`;
- `metric_distributions.png`, `snr_input_output_scatter.png`: phân phối và quan hệ SNR;
- `record_snr_improvement_heatmap.png`: độ ổn định qua các QTDB records;
- `signal_examples_<artifact>_protocol<P>_K<K>.png`: clean/noisy/denoised và phổ tần;
- `figure_manifest.csv`, `artifact_comparison.png`, `run_info.json` và file ZIP tổng.

Sau khi version hoàn tất, tải file `qtdb_<artifact(s)>_evaluation.zip` ở tab **Output**. Khi chạy tách ba version, mỗi ZIP chứa kết quả độc lập của BW, EM hoặc MA.
""",
    )

    output["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    TARGET.write_text(
        json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    build()
