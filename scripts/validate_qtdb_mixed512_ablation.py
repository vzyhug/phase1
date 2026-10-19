"""Local synthetic preflight for the five standalone MIX7 Kaggle notebooks."""

import ast
import json
import math
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
PATHS = [ROOT / "notebooks" / f"kaggle_ablation_u{i}_all_noise_512_100ep.ipynb"
         for i in range(5)]
baseline_fingerprint = None
baseline_batch = None

for variant_id, path in enumerate(PATHS):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code" and not cell.source.lstrip().startswith("%"):
            ast.parse(cell.source)
    config_cell = notebook.cells[4].source
    assert "ARTIFACTS_TO_TRAIN = ('MIX7',)" in config_cell
    assert "PROTOCOLS_TO_TRAIN = (1,)" in config_cell
    assert "TARGET_EPOCHS = 100" in config_cell
    assert f"qtdb_u{variant_id}_all_noise_512_100ep" in config_cell

    count = 700
    axis = np.linspace(0, 8 * np.pi, 512, dtype=np.float32)
    clean = np.stack([np.sin(axis + i / 20).astype(np.float32)
                      for i in range(count)])
    noise = {}
    for index, artifact in enumerate(("BW", "EM", "MA")):
        rng = np.random.RandomState(100 + index)
        noise[artifact] = rng.standard_normal((4096, 2)).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        scope = {
            "np": np, "pd": pd, "torch": torch, "math": math, "Path": Path,
            "json": json, "hashlib": __import__("hashlib"),
            "DataLoader": DataLoader, "Dataset": Dataset,
            "SEED": 1234, "LENGTH": 512, "DEVICE": torch.device("cpu"),
            "CONFIG": {
                "train": {"feats": 80, "batch_size": 4, "lr": 1e-4,
                          "lambda_time": 1.0, "lambda_freq": 0.1},
                "diffusion": {"beta_start": 1e-4, "beta_end": 0.5,
                              "num_steps": 50, "schedule": "quad"},
            },
            "TRAIN_CLEAN": clean,
            "TRAIN_INDICES": np.arange(490),
            "VAL_INDICES": np.arange(490, count),
            "TRAIN_META": pd.DataFrame({
                "sample_id": [f"sample_{i}" for i in range(count)],
                "record_id": [f"record_{i // 10}" for i in range(count)],
            }),
            "NOISE": noise,
            "NOISE_MIDPOINT": {key: 2048 for key in noise},
            "PROTOCOLS_TO_TRAIN": (1,),
            "OUT": Path(directory),
        }
        exec(notebook.cells[18].source, scope)
        fingerprint = scope["DATA_FINGERPRINT"]
        dataset = scope["FixedMixedDataset"](clean, np.arange(7), 1)
        pairs = [dataset[i][1].numpy() for i in range(7)]
        if baseline_fingerprint is None:
            baseline_fingerprint, baseline_batch = fingerprint, pairs
        else:
            assert fingerprint == baseline_fingerprint
            for left, right in zip(pairs, baseline_batch):
                np.testing.assert_array_equal(left, right)

        exec(notebook.cells[6].source, scope)
        exec(notebook.cells[8].source, scope)
        model = scope["ResearchLossDDPM"](
            scope["UNet1D"](2, 80, 128, 1), scope["CONFIG"],
            torch.device("cpu"),
        )
        clean_batch, noisy_batch = dataset[0]
        components = model.loss_components(clean_batch[None], noisy_batch[None])
        assert torch.isfinite(components["total"])
        components["total"].backward()
        assert torch.isfinite(next(model.parameters()).grad).all()
        model.eval()
        prediction = model.denoising(
            noisy_batch[None], use_ddim=True, ddim_steps=50,
            ddim_eta=0.0, num_shots=1,
        )
        assert prediction.shape == (1, 1, 512)
        assert torch.isfinite(prediction).all()
        print(path.name, "OK", sum(p.numel() for p in model.model.parameters()))

        if variant_id == 0:
            # Exercise the unmodified training/checkpoint path for one tiny
            # synthetic epoch without downloading data or requiring Kaggle.
            # The original trainer unconditionally stores CUDA RNG state; the
            # local PyTorch build is CPU-only, so emulate that state here.
            torch.cuda.get_rng_state = lambda: torch.get_rng_state()
            torch.cuda.set_rng_state = lambda state: torch.set_rng_state(state)
            scope.update({
                "TRAIN_INDICES": np.arange(14),
                "VAL_INDICES": np.arange(14, 28),
                "TRAIN_DIR": Path(directory) / "checkpoints",
                "ARTIFACTS_TO_TRAIN": ("MIX7",),
                "TARGET_EPOCHS": 1,
                "EPOCHS_PER_EXPERIMENT_THIS_SESSION": 1,
                "CHECKPOINT_TAG": "synthetic_preflight",
                "TRAIN_RECORDS": ("synthetic_train",),
                "TEST_RECORDS": ("synthetic_test",),
                "time": __import__("time"),
                "gc": __import__("gc"),
                "shutil": __import__("shutil"),
                "tqdm": tqdm,
                "display": lambda *_: None,
            })
            scope["TRAIN_DIR"].mkdir()
            exec(notebook.cells[20].source, scope)
            checkpoints = list(scope["TRAIN_DIR"].glob("best_*.pth"))
            assert len(checkpoints) == 1
            payload = torch.load(checkpoints[0], map_location="cpu",
                                 weights_only=False)
            assert payload["ablation_id"] == "U0"
            assert payload["artifact"] == "MIX7"
            assert payload["data_fingerprint"] == fingerprint
            print("Synthetic one-epoch checkpoint path OK")

print("All five MIX7 notebooks: same synthetic pairs/fingerprint, finite loss/grad/DDIM")
