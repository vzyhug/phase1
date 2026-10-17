"""Validate the research-loss mixed notebook and its local NPZ contract."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.main_model import DDPM, default
from src.models.unet_1d import UNet1D


NOTEBOOK = ROOT / "notebooks" / "kaggle_train_proposed_research_mixed_3600.ipynb"
PRIOR = ROOT / "notebooks" / "kaggle_train_qtdb_bw_em_ma.ipynb"
DATA = ROOT / "common_qtdb_mixed_bwem_ma_3600_ntype1.npz"


def content(cell: dict) -> str:
    return "".join(cell["source"])


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
prior = json.loads(PRIOR.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4
cells = notebook["cells"]
for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        ast.parse(content(cell), filename=f"cell_{index}")

prior_loss = content(prior["cells"][8])
prior_class = next(
    ast.get_source_segment(prior_loss, node)
    for node in ast.parse(prior_loss).body
    if isinstance(node, ast.ClassDef) and node.name == "ResearchLossDDPM"
)
current_model_cell = content(cells[11])
current_class = next(
    ast.get_source_segment(current_model_cell, node)
    for node in ast.parse(current_model_cell).body
    if isinstance(node, ast.ClassDef) and node.name == "ResearchLossDDPM"
)
assert current_class == prior_class, "ResearchLossDDPM differs from prior BW/EM/MA run"
loss_hash = hashlib.sha256(prior_class.encode("utf-8")).hexdigest()
assert loss_hash in content(cells[3]) and loss_hash in current_model_cell

source_cell = content(cells[5])
prefix = 'EMBEDDED_SOURCE_B64 = json.loads(r"""'
embedded_json = source_cell.split(prefix, 1)[1].split('""")', 1)[0]
for relative, encoded in json.loads(embedded_json).items():
    assert base64.b64decode(encoded) == (ROOT / relative).read_bytes(), relative

config = content(cells[3])
training = content(cells[15])
assert "TARGET_EPOCHS = 400" in config
assert "REFERENCE_BATCH_SIZE = 96" in config
assert "MICRO_BATCH_SIZE = 4" in config
assert "REQUIRE_RESUME_CHECKPOINT = True" in config
assert "EXPECTED_MIXED_BUNDLE_SHA256" in config
assert "'lr': 1e-4, 'weight_decay': 0.0" in config
assert "'lambda_time': 1.0, 'lambda_freq': 0.1" in config
assert "ResearchLossDDPM(base, MODEL_CONFIG, DEVICE)" in current_model_cell
assert "model.loss_components(clean, noisy)" in training
assert "components['total'] / GRAD_ACCUM_STEPS" in training
assert "clip_grad_norm_(model.parameters(), 1.0)" in training
assert "ema.update(model)" in training
assert "train_time_loss" in training and "train_frequency_loss" in training
assert "best_proposed_research_mixed_3600.pth" in content(cells[13])
assert "if REQUIRE_RESUME_CHECKPOINT and resume_path is None" in content(cells[13])
assert "last_proposed_tfcdiff_mixed_3600.pth" not in content(cells[13])
assert "proposed_research_mixed_ntype1_seed1234.npz" in content(cells[21])
assert "MAX_NONFINITE_STEPS_PER_EPOCH = 5" in training
assert "nonfinite_gradient_steps" in training
data_loader = content(cells[7])
assert "actual_bundle_sha256 != EXPECTED_MIXED_BUNDLE_SHA256" in data_loader
assert "noise_version != 1" in data_loader

# Execute the exact optimizer guard from the notebook on a tiny CPU model.
guard_node = next(
    node for node in ast.walk(ast.parse(training))
    if isinstance(node, ast.If)
    and "batch_index + 1" in ast.unparse(node.test)
    and "GRAD_ACCUM_STEPS" in ast.unparse(node.test)
)
guard_code = compile(ast.Module(body=[guard_node], type_ignores=[]), "gradient_guard", "exec")


class CountingEma:
    def __init__(self):
        self.updates = 0

    def update(self, _model):
        self.updates += 1


def guard_smoke(gradient_value: float, amp_enabled: bool = False):
    small_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        small_model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(small_model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=amp_enabled)
    if amp_enabled:
        scaler.scale(small_model(torch.ones(1, 1)).sum()).backward()
    small_model.weight.grad = torch.full_like(small_model.weight, gradient_value)
    initial_scale = scaler.get_scale()
    ema = CountingEma()
    variables = {
        "torch": torch, "model": small_model, "optimizer": optimizer,
        "scaler": scaler, "ema": ema, "USE_AMP": amp_enabled,
        "batch_index": 23, "GRAD_ACCUM_STEPS": 24,
        "nonfinite_gradient_steps": 0, "skipped_steps": 0,
        "MAX_NONFINITE_STEPS_PER_EPOCH": 5, "gradient_norms": [],
        "epoch": 40, "print": lambda *args, **kwargs: None,
    }
    exec(guard_code, variables)
    return small_model.weight.item(), ema.updates, variables, initial_scale, scaler.get_scale()


bad_weight, bad_ema, bad_state, _, _ = guard_smoke(float("inf"))
assert bad_weight == 1.0 and bad_ema == 0
assert bad_state["nonfinite_gradient_steps"] == 1
assert bad_state["skipped_steps"] == 1
good_weight, good_ema, good_state, _, _ = guard_smoke(2.0)
assert good_weight < 1.0 and good_ema == 1
assert good_state["nonfinite_gradient_steps"] == 0
amp_bad_weight, amp_bad_ema, _, amp_scale_before, amp_scale_after = guard_smoke(
    float("inf"), amp_enabled=True)
assert amp_bad_weight == 1.0 and amp_bad_ema == 0
assert amp_scale_after == amp_scale_before * 0.5

if DATA.is_file():
    expected_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert expected_hash in config, "Pinned mixed bundle SHA-256 differs from local file"
    shapes = {}
    with zipfile.ZipFile(DATA) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                major, minor = np.lib.format.read_magic(stream)
                if (major, minor) == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(stream)
                shapes[name[:-4]] = shape
    assert shapes["clean_train"] == shapes["noisy_train"] == (22804, 1, 3600)
    assert shapes["clean_val"] == shapes["noisy_val"] == (9774, 1, 3600)
    assert shapes["clean_test"] == shapes["noisy_test"] == (5012, 1, 3600)

# A small CPU pass verifies that all four encoder levels, FiLM bridges,
# bottleneck attention and decoder receive a gradient through the exact loss.
namespace = {"DDPM": DDPM, "torch": torch, "F": F, "default": default}
exec(compile(prior_class, "research_loss", "exec"), namespace)
ResearchLossDDPM = namespace["ResearchLossDDPM"]
torch.manual_seed(1234)
smoke_config = {
    "train": {"lambda_time": 1.0, "lambda_freq": 0.1},
    "diffusion": {"beta_start": 1e-4, "beta_end": 0.5,
                  "num_steps": 50, "schedule": "quad"},
}
smoke_model = ResearchLossDDPM(
    UNet1D(in_channels=2, base_channels=8, emb_dim=128, out_channels=1),
    smoke_config, torch.device("cpu"),
)
clean = torch.randn(1, 1, 128)
noisy = clean + 0.2 * torch.randn_like(clean)
components = smoke_model.loss_components(clean, noisy)
assert all(torch.isfinite(value) for value in components.values())
components["total"].backward()
for module_name in ("enc1", "enc2", "enc3", "enc4", "bridge1", "bridge2",
                    "bridge3", "bridge4", "attn", "dec4", "dec3", "dec2", "final"):
    module = getattr(smoke_model.model, module_name)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in module.parameters()), module_name

print(f"OK: {NOTEBOOK}")
print("Research loss byte-identical to BW/EM/MA notebook; embedded src matches files")
print("CPU forward/backward: all HNF, FiLM, attention and decoder groups receive gradients")
print("Gradient guard: non-finite update skipped, AMP scale halved; finite update and EMA retained")
if DATA.is_file():
    print("Local mixed NPZ headers: 22804 train / 9774 val / 5012 test, [N,1,3600]")
