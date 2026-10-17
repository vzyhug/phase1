"""Build the self-contained QTDB U0 ablation notebook from the 512-sample trainer.

The data generation, diffusion implementation, loss, optimizer, EMA, and
checkpoint loop are inherited unchanged. Only the U-Net block implementations
and run identity/configuration are specialized for the U0 architecture.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "kaggle-train-qtdb-bw-em-ma-512.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_ablation_u0_qtdb_bw_em_ma_512_100ep.ipynb"


def put(cell, value):
    cell["source"] = value.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

put(cells[0], """# U0 baseline — QTDB conditional DDPM, 512 samples

This is the **plain 1D U-Net control** for the HNF–FiLM–attention ablation. It
keeps the source notebook's QTDB/NSTDB preparation, `x_t` plus noisy ECG input,
diffusion schedule, multi-domain loss, optimizer, EMA, and checkpoint selection.
Only three architecture switches change: HNF → ordinary Conv1D blocks,
FiLM → additive sinusoidal timestep embedding, attention → `Identity`.

Default run: **BW / P1, 100 epochs from scratch**. Set `ARTIFACTS_TO_TRAIN` to
`('EM',)` or `('MA',)` for the other independent runs. Do not claim an apples-to-
apples numerical comparison with U4 until preprocessing, split, loss, EMA,
checkpoint selection and evaluation protocol are also confirmed identical.
The notebook trains and logs validation; the locked test set belongs to the
common evaluation workflow, not checkpoint selection here.
""")
put(cells[3], "## 2. Reproducible configuration — select one artifact per Kaggle run\n")

config = "".join(cells[4]["source"])
config = config.replace("ARTIFACTS_TO_TRAIN = ('EM',) # Hoàng : BW , Nghĩa : MA , Hưng : EM",
                        "ARTIFACTS_TO_TRAIN = ('BW',)  # Change to ('EM',) or ('MA',) for another run")
# The original comment may be serialized with a different Unicode spelling.
config_lines = config.splitlines(keepends=True)
config_lines = [
    "ARTIFACTS_TO_TRAIN = ('BW',)  # Change to ('EM',) or ('MA',) for another run\n"
    if line.startswith("ARTIFACTS_TO_TRAIN =") else line
    for line in config_lines
]
config = "".join(config_lines)
config = config.replace("PROTOCOLS_TO_TRAIN = (1, 2)", "PROTOCOLS_TO_TRAIN = (1,)")
config = config.replace("CHECKPOINT_TAG = 'norm_loss_v2'",
                        "CHECKPOINT_TAG = 'u0_conv_addtime_noattn_100ep_v1'")
config = config.replace("qtdb_bw_em_ma_training')", "qtdb_u0_512_100ep')")
config = config.replace("    'preprocessing_version': PREPROCESSING_VERSION,",
                        "    'preprocessing_version': PREPROCESSING_VERSION,\n"
                        "    'architecture': {'id': 'U0', 'feature_block': 'two_conv3_relu',\n"
                        "                     'timestep': 'sinusoidal_additive', 'attention': 'identity'},")
assert "ARTIFACTS_TO_TRAIN = ('BW',)" in config
assert "TARGET_EPOCHS = 100" in config
assert "EPOCHS_PER_EXPERIMENT_THIS_SESSION = 100" in config
assert "'id': 'U0'" in config
put(cells[4], config)
put(cells[5], "## 3. U0 architecture: plain Conv1D, additive timestep, no attention\n")

put(cells[6], '''import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Two ordinary same-length Conv1D layers; no HNF or half normalization."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AdditiveTimeBlock(nn.Module):
    """Same 128-D sinusoidal diffusion input, but additive rather than FiLM."""

    def __init__(self, features, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(nn.Linear(emb_dim, features), nn.SiLU())

    def forward(self, x, noise_scale):
        noise_scale = noise_scale.reshape(-1)
        half = self.emb_dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=x.device, dtype=x.dtype)
            * (-math.log(10000) / (half - 1))
        )
        angles = noise_scale[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return x + self.proj(embedding).unsqueeze(-1)


class UNet1D(nn.Module):
    """U0: same four-level conditional diffusion U-Net scaffold as U4."""

    def __init__(self, in_channels=2, base_channels=80, emb_dim=128,
                 out_channels=1):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock1D(in_channels, c)
        self.bridge1 = AdditiveTimeBlock(c, emb_dim)
        self.down1 = nn.Conv1d(c, c * 2, 4, stride=2, padding=1)
        self.enc2 = ConvBlock1D(c * 2, c * 2)
        self.bridge2 = AdditiveTimeBlock(c * 2, emb_dim)
        self.down2 = nn.Conv1d(c * 2, c * 4, 4, stride=2, padding=1)
        self.enc3 = ConvBlock1D(c * 4, c * 4)
        self.bridge3 = AdditiveTimeBlock(c * 4, emb_dim)
        self.down3 = nn.Conv1d(c * 4, c * 8, 4, stride=2, padding=1)
        self.enc4 = ConvBlock1D(c * 8, c * 8)
        self.bridge4 = AdditiveTimeBlock(c * 8, emb_dim)
        self.attn = nn.Identity()
        self.up4 = nn.ConvTranspose1d(c * 8, c * 4, 4, stride=2, padding=1)
        self.dec4 = ConvBlock1D(c * 8, c * 4)
        self.up3 = nn.ConvTranspose1d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec3 = ConvBlock1D(c * 4, c * 2)
        self.up2 = nn.ConvTranspose1d(c * 2, c, 4, stride=2, padding=1)
        self.dec2 = ConvBlock1D(c * 2, c)
        self.final = nn.Conv1d(c, out_channels, 1)

    def forward(self, x, cond, noise_scale):
        if x.shape != cond.shape or x.shape[-1] % 8:
            raise ValueError('Expected matching [B,1,L] inputs with L divisible by 8')
        e1 = self.bridge1(self.enc1(torch.cat([x, cond], dim=1)), noise_scale)
        e2 = self.bridge2(self.enc2(self.down1(e1)), noise_scale)
        e3 = self.bridge3(self.enc3(self.down2(e2)), noise_scale)
        e4 = self.attn(self.bridge4(self.enc4(self.down3(e3)), noise_scale))
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.final(d2)


with torch.no_grad():
    probe_model = UNet1D(2, CONFIG['train']['feats'], 128, 1).to(DEVICE)
    probe_input = torch.zeros(2, 1, LENGTH, device=DEVICE)
    probe_output = probe_model(probe_input, probe_input,
                               torch.ones(2, 1, device=DEVICE))
    assert probe_output.shape == probe_input.shape
    print('U0 preflight:', tuple(probe_output.shape), '| parameters:',
          sum(p.numel() for p in probe_model.parameters()))
    del probe_model, probe_input, probe_output
''')

train_cell = "".join(cells[20]["source"])
train_cell = train_cell.replace("return f'{artifact.lower()}_protocol{protocol}_{CHECKPOINT_TAG}'",
                                "return f'u0_{artifact.lower()}_protocol{protocol}_{CHECKPOINT_TAG}'")
train_cell = train_cell.replace("        'artifact': artifact,\n        'protocol': protocol,",
                                "        'ablation_id': 'U0',\n        'artifact': artifact,\n        'protocol': protocol,")
train_cell = train_cell.replace("        'best_validation_loss': best_val,",
                                "        'best_validation_loss': best_val,\n"
                                "        'parameter_count': sum(p.numel() for p in model.parameters()),", 1)
train_cell = train_cell.replace("        if payload.get('artifact') != artifact or int(payload.get('protocol', -1)) != protocol:",
                                "        if payload.get('ablation_id') != 'U0' or payload.get('artifact') != artifact or int(payload.get('protocol', -1)) != protocol:")
put(cells[20], train_cell)

package_cell = "".join(cells[22]["source"])
package_cell = package_cell.replace("qtdb_bw_em_ma_training_outputs.zip",
                                    "qtdb_u0_512_100ep_outputs.zip")
package_cell = package_cell.replace("        'file': path.name, 'artifact': payload.get('artifact'),",
                                    "        'file': path.name, 'ablation_id': payload.get('ablation_id'),\n"
                                    "        'parameters': payload.get('parameter_count'), 'artifact': payload.get('artifact'),")
put(cells[22], package_cell)

put(cells[23], """## Checkpoint and comparison contract

`best_u0_<artifact>_protocol1_u0_conv_addtime_noattn_100ep_v1.pth` stores
the raw and EMA states, U0 identity, configuration, artifact, protocol,
epoch, and validation loss. Use the EMA state for the common test evaluation.
Do not rename it as an original U4 checkpoint or mix it with 3600-sample data.

**Fairness check:** U0 now targets 100 epochs, matching the original U4
training horizon. Before a final architecture claim, verify all other training
and data settings as well. Compare predictions on paired test segments with
the same DDIM sampler/seed and report parameter counts.
""")

for cell in cells:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                  encoding="utf-8")
print(TARGET)
