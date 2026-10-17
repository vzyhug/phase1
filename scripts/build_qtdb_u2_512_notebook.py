"""Build U2 = U0 + original research FiLM timestep conditioning."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle_ablation_u0_qtdb_bw_em_ma_512_100ep.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_ablation_u2_qtdb_bw_em_ma_512_100ep.ipynb"
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

put(cells[0], """# U2 = U0 + FiLM — QTDB conditional DDPM, 512 samples

This notebook isolates the contribution of the research **FiLM timestep
conditioning**. Relative to U0, it changes only the timestep conditioning
blocks: additive sinusoidal embedding → original FiLM scale and shift. The
same ordinary two-Conv1D feature blocks and `Identity` bottleneck attention
remain in place. QTDB/NSTDB preprocessing, `x_t` plus noisy ECG input,
diffusion, loss, optimizer, EMA and validation checkpoint selection are
unchanged.

Default run: **BW / P1, 100 epochs**. Set `ARTIFACTS_TO_TRAIN` to `('EM',)` or
`('MA',)` for independent runs. Compare U2 with U0 only on paired test samples
from the same artifact and protocol. Test data do not select the checkpoint.
""")
put(cells[5], "## 3. U2 architecture: plain Conv1D, original FiLM, no attention\n")

configuration = source(cells[4])
assert "'id': 'U0'" in configuration
assert "'timestep': 'sinusoidal_additive'" in configuration
assert "TARGET_EPOCHS = 100" in configuration
configuration = configuration.replace("'id': 'U0'", "'id': 'U2'")
configuration = configuration.replace("'timestep': 'sinusoidal_additive'",
                                      "'timestep': 'sinusoidal_film_scale_shift'")
configuration = configuration.replace("u0_conv_addtime_noattn_100ep_v1",
                                      "u2_conv_film_noattn_100ep_v1")
configuration = configuration.replace("qtdb_u0_512_100ep", "qtdb_u2_512_100ep")
put(cells[4], configuration)

architecture = source(cells[6])
block_start = architecture.index("class AdditiveTimeBlock(nn.Module):")
block_end = architecture.index("class UNet1D(nn.Module):")
film = FILM_SOURCE.read_text(encoding="utf-8")
film = film[film.index("class BridgeBlock(nn.Module):"):].strip() + "\n\n\n"
architecture = architecture[:block_start] + film + architecture[block_end:]
architecture = architecture.replace("AdditiveTimeBlock(", "BridgeBlock(")
architecture = architecture.replace("U0: same four-level", "U2: same four-level")
architecture = architecture.replace("U0 preflight:", "U2 preflight:")
assert "class BridgeBlock(nn.Module)" in architecture
assert "class AdditiveTimeBlock" not in architecture
assert "class ConvBlock1D(nn.Module)" in architecture
assert "self.attn = nn.Identity()" in architecture
put(cells[6], architecture)

training = source(cells[20])
assert "'ablation_id': 'U0'" in training
training = training.replace("u0_{artifact.lower()}", "u2_{artifact.lower()}")
training = training.replace("'ablation_id': 'U0'", "'ablation_id': 'U2'")
training = training.replace("payload.get('ablation_id') != 'U0'",
                            "payload.get('ablation_id') != 'U2'")
put(cells[20], training)

package = source(cells[22]).replace("qtdb_u0_512_100ep_outputs.zip",
                                     "qtdb_u2_512_100ep_outputs.zip")
put(cells[22], package)
put(cells[23], """## Checkpoint and comparison contract

`best_u2_<artifact>_protocol1_u2_conv_film_noattn_100ep_v1.pth` stores the
raw and EMA states, U2 identity, configuration, artifact, protocol, epoch,
parameter count and validation loss. Use the EMA state for common test
evaluation. U0 and U2 differ in **timestep conditioning only**. Evaluate on
paired test samples with identical DDIM settings and metrics; report
parameter counts. Do not load U0 weights into U2 or use 3600-sample mixed
benchmark checkpoints for this ablation.
""")

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                  encoding="utf-8")
print(TARGET)
