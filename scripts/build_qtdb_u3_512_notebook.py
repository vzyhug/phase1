"""Build U3 = HNF + original FiLM, without bottleneck attention."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle_ablation_u1_qtdb_bw_em_ma_512_100ep.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_ablation_u3_qtdb_bw_em_ma_512_100ep.ipynb"
FILM_SOURCE = ROOT / "src" / "models" / "blocks" / "bridge_block.py"


def source(cell):
    return "".join(cell["source"])


def put(cell, value):
    cell["source"] = value.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

put(cells[0], """# U3 = HNF + FiLM — QTDB conditional DDPM, 512 samples

U3 combines the original research **HNF** feature blocks with original
**FiLM** timestep scale/shift. Bottleneck attention remains `Identity`, so
U3 versus the full U4 isolates attention. Compared with U1, only FiLM is
added; compared with U2, only HNF is added. The four-level U-Net scaffold,
`x_t` plus noisy ECG input, QTDB/NSTDB data, normalization, DDPM/DDIM,
multi-domain loss, optimizer, EMA and checkpoint selection are unchanged.

Default run: **BW / P1, 100 epochs**. Set `ARTIFACTS_TO_TRAIN` to `('EM',)` or
`('MA',)` for independent runs. Evaluate Ui on paired test samples from the
same artifact/protocol. The test set is not used to select checkpoints.
""")
put(cells[5], "## 3. U3 architecture: original HNF + original FiLM, no attention\n")

configuration = source(cells[4])
assert "'id': 'U1'" in configuration
assert "'feature_block': 'hnf_multiscale_halfnorm_residual'" in configuration
assert "'timestep': 'sinusoidal_additive'" in configuration
assert "TARGET_EPOCHS = 100" in configuration
configuration = configuration.replace("'id': 'U1'", "'id': 'U3'")
configuration = configuration.replace("'timestep': 'sinusoidal_additive'",
                                      "'timestep': 'sinusoidal_film_scale_shift'")
configuration = configuration.replace("u1_hnf_addtime_noattn_100ep_v1",
                                      "u3_hnf_film_noattn_100ep_v1")
configuration = configuration.replace("qtdb_u1_512_100ep", "qtdb_u3_512_100ep")
put(cells[4], configuration)

architecture = source(cells[6])
block_start = architecture.index("class AdditiveTimeBlock(nn.Module):")
block_end = architecture.index("class UNet1D(nn.Module):")
film = FILM_SOURCE.read_text(encoding="utf-8")
film = film[film.index("class BridgeBlock(nn.Module):"):].strip() + "\n\n\n"
architecture = architecture[:block_start] + film + architecture[block_end:]
architecture = architecture.replace("AdditiveTimeBlock(", "BridgeBlock(")
architecture = architecture.replace("U1: same four-level", "U3: same four-level")
architecture = architecture.replace("U1 preflight:", "U3 preflight:")
assert "class HNFBlock(nn.Module)" in architecture
assert "class BridgeBlock(nn.Module)" in architecture
assert "class AdditiveTimeBlock" not in architecture
assert "self.attn = nn.Identity()" in architecture
put(cells[6], architecture)

training = source(cells[20])
assert "'ablation_id': 'U1'" in training
training = training.replace("u1_{artifact.lower()}", "u3_{artifact.lower()}")
training = training.replace("'ablation_id': 'U1'", "'ablation_id': 'U3'")
training = training.replace("payload.get('ablation_id') != 'U1'",
                            "payload.get('ablation_id') != 'U3'")
put(cells[20], training)

package = source(cells[22]).replace("qtdb_u1_512_100ep_outputs.zip",
                                     "qtdb_u3_512_100ep_outputs.zip")
put(cells[22], package)
put(cells[23], """## Checkpoint and comparison contract

`best_u3_<artifact>_protocol1_u3_hnf_film_noattn_100ep_v1.pth` stores the
raw and EMA states, U3 identity, configuration, artifact, protocol, epoch,
parameter count and validation loss. Use EMA for common test evaluation.
U3 versus U1 tests FiLM conditional on HNF; U3 versus U2 tests HNF
conditional on FiLM. U3 versus the existing full U4 tests bottleneck
attention, subject to verifying the same 512-sample training/evaluation
protocol. Use paired test samples and identical DDIM settings/metrics.
""")

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                  encoding="utf-8")
print(TARGET)
