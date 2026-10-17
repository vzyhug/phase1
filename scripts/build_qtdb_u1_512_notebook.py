"""Build U1 = U0 + exact research HNF blocks, all else unchanged."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle_ablation_u0_qtdb_bw_em_ma_512_100ep.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_ablation_u1_qtdb_bw_em_ma_512_100ep.ipynb"
HNF_SOURCE = ROOT / "src" / "models" / "blocks" / "hnf_block.py"


def source(cell):
    return "".join(cell["source"])


def put(cell, value):
    cell["source"] = value.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

put(cells[0], """# U1 = U0 + HNF — QTDB conditional DDPM, 512 samples

This notebook isolates the contribution of the research **Half Normalized
Filter (HNF)** blocks. Relative to U0, it changes only the feature blocks in
the same 4-level 1D U-Net scaffold. It preserves additive sinusoidal timestep
conditioning, `Identity` bottleneck attention, `x_t` plus noisy ECG input,
QTDB/NSTDB data, preprocessing, diffusion, loss, optimizer, EMA, and validation
checkpoint selection.

Default run: **BW / P1, 100 epochs**. Set `ARTIFACTS_TO_TRAIN` to `('EM',)` or
`('MA',)` for the other independent runs. U1-vs-U0 is a paired architecture
comparison only when the same artifact/protocol and locked test samples are
used. The notebook does not inspect test data while choosing checkpoints.
""")
put(cells[5], "## 3. U1 architecture: original HNF, additive timestep, no attention\n")

configuration = source(cells[4])
assert "'id': 'U0'" in configuration
assert "'feature_block': 'two_conv3_relu'" in configuration
assert "TARGET_EPOCHS = 100" in configuration
configuration = configuration.replace(
    "'id': 'U0', 'feature_block': 'two_conv3_relu'",
    "'id': 'U1', 'feature_block': 'hnf_multiscale_halfnorm_residual'",
)
configuration = configuration.replace(
    "u0_conv_addtime_noattn_100ep_v1", "u1_hnf_addtime_noattn_100ep_v1"
)
configuration = configuration.replace("qtdb_u0_512_100ep", "qtdb_u1_512_100ep")
put(cells[4], configuration)

architecture = source(cells[6])
block_start = architecture.index("class ConvBlock1D(nn.Module):")
block_end = architecture.index("class AdditiveTimeBlock(nn.Module):")
hnf = HNF_SOURCE.read_text(encoding="utf-8")
hnf = hnf[hnf.index("class HNFBlock(nn.Module):"):].strip() + "\n\n\n"
architecture = architecture[:block_start] + hnf + architecture[block_end:]
architecture = architecture.replace("ConvBlock1D(", "HNFBlock(")
architecture = architecture.replace("U0: same four-level", "U1: same four-level")
architecture = architecture.replace("U0 preflight:", "U1 preflight:")
assert "class HNFBlock(nn.Module)" in architecture
assert "class ConvBlock1D" not in architecture
assert "self.attn = nn.Identity()" in architecture
assert "class AdditiveTimeBlock(nn.Module)" in architecture
put(cells[6], architecture)

training = source(cells[20])
assert "'ablation_id': 'U0'" in training
training = training.replace("u0_{artifact.lower()}", "u1_{artifact.lower()}")
training = training.replace("'ablation_id': 'U0'", "'ablation_id': 'U1'")
training = training.replace("payload.get('ablation_id') != 'U0'",
                            "payload.get('ablation_id') != 'U1'")
put(cells[20], training)

package = source(cells[22]).replace("qtdb_u0_512_100ep_outputs.zip",
                                     "qtdb_u1_512_100ep_outputs.zip")
put(cells[22], package)
put(cells[23], """## Checkpoint and comparison contract

`best_u1_<artifact>_protocol1_u1_hnf_addtime_noattn_100ep_v1.pth` stores
the raw and EMA states, U1 identity, configuration, artifact, protocol,
epoch, parameter count and validation loss. Use the EMA state for common test
evaluation. The U0 and U1 notebooks differ in **HNF feature blocks only**;
they must be evaluated with paired test samples, the same DDIM settings and
the same metrics. Do not load U0 weights into U1 or treat the 3600-sample
mixed-noise checkpoint as part of this ablation.
""")

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                  encoding="utf-8")
print(TARGET)
