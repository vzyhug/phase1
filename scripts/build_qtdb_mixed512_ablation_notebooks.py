"""Build five standalone Kaggle notebooks for one-model, seven-mask mixed-noise ablation.

All variants inherit the existing 512-sample QTDB preparation and conditional
DDPM training code. They share an identical deterministic BW/EM/MA mixture
dataset and differ only in the U-Net feature, timestep, and attention blocks.
"""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
ATTENTION_SOURCE = ROOT / "src" / "models" / "blocks" / "attention_1d.py"
VARIANTS = (
    ("u0", "none", "additive", "identity"),
    ("u1", "hnf", "additive", "identity"),
    ("u2", "none", "film", "identity"),
    ("u3", "hnf", "film", "identity"),
    ("u4", "hnf", "film", "self_attention"),
)


def cell_source(cell):
    return "".join(cell["source"])


def set_cell(cell, content):
    cell["source"] = content.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


MIXED_DATA_CELL = '''# One immutable seven-subset mixture assignment for every Ui.
# The seven masks are BW, EM, MA, BW+EM, BW+MA, EM+MA, BW+EM+MA.
MIX_COMPONENTS = ('BW', 'EM', 'MA')
MIX_MASKS = np.asarray([
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
], dtype=np.uint8)
MIX_NAMES = ('BW', 'EM', 'MA', 'BW+EM', 'BW+MA', 'EM+MA', 'BW+EM+MA')
mask_pool = np.tile(np.arange(7, dtype=np.uint8),
                    (len(TRAIN_CLEAN) + 6) // 7)[:len(TRAIN_CLEAN)]
MIX_MASK_IDS = np.random.RandomState(SEED + 404).permutation(mask_pool)
MIX_LEVELS = (np.random.RandomState(SEED).randint(
    20, 200, size=len(TRAIN_CLEAN)).astype(np.float32) / 100.0)
assert np.all(MIX_MASKS.sum(axis=1) >= 1)
assert len(MIX_MASK_IDS) == len(MIX_LEVELS) == len(TRAIN_CLEAN)
for split_name, split_indices in (('train', TRAIN_INDICES),
                                  ('validation', VAL_INDICES)):
    counts = np.bincount(MIX_MASK_IDS[split_indices], minlength=7)
    assert np.all(counts > 0), (split_name, counts)
    print(split_name, dict(zip(MIX_NAMES, counts.tolist())))


class FixedMixedDataset(Dataset):
    """Deterministic clean/noisy pairs; all Ui see the same seven mask classes."""

    def __init__(self, clean_array, indices, protocol):
        self.clean = clean_array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.protocol = int(protocol)
        train_channel = 0 if protocol == 1 else 1
        self.noise = {}
        self.patch_count = {}
        for artifact in MIX_COMPONENTS:
            values = NOISE[artifact][:NOISE_MIDPOINT[artifact], train_channel]
            self.noise[artifact] = values
            self.patch_count[artifact] = len(values) // LENGTH
            if self.patch_count[artifact] == 0:
                raise RuntimeError(f'{artifact}: noise training half too short')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        source_index = int(self.indices[item])
        clean_raw = self.clean[source_index]
        mask = MIX_MASKS[int(MIX_MASK_IDS[source_index])]
        patches = []
        for enabled, artifact in zip(mask, MIX_COMPONENTS):
            if enabled:
                start = (source_index % self.patch_count[artifact]) * LENGTH
                patches.append(self.noise[artifact][start:start + LENGTH])
        mixed_patch = np.sum(np.stack(patches, axis=0), axis=0)
        clean_range = float(np.ptp(clean_raw))
        noise_range = float(np.ptp(mixed_patch))
        if clean_range <= 0 or noise_range <= 0:
            raise ValueError(f'Zero range at QTDB sample {source_index}')
        noisy_raw = (clean_raw + float(MIX_LEVELS[source_index])
                     * clean_range / noise_range * mixed_patch)
        scale = max(float(np.max(np.abs(clean_raw))), 1e-8)
        clean = torch.from_numpy((clean_raw / scale).astype(np.float32))[None, :]
        noisy = torch.from_numpy((noisy_raw / scale).astype(np.float32))[None, :]
        return clean, noisy


def make_loader(artifact, protocol, indices, shuffle, drop_last, epoch=0):
    if artifact != 'MIX7':
        raise ValueError('This notebook trains one MIX7 model, not three specialists')
    dataset = FixedMixedDataset(TRAIN_CLEAN, indices, protocol)
    generator = torch.Generator().manual_seed(
        SEED + 44 * 100_000 + protocol * 1_000 + epoch)
    return DataLoader(dataset, batch_size=CONFIG['train']['batch_size'],
                      shuffle=shuffle, drop_last=drop_last,
                      generator=generator, num_workers=0, pin_memory=True)


def sha_array(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode('ascii'))
    digest.update(str(values.dtype).encode('ascii'))
    digest.update(values.tobytes())
    return digest.hexdigest()


DATA_FINGERPRINT = {
    'clean_512': sha_array(TRAIN_CLEAN),
    'train_indices': sha_array(TRAIN_INDICES),
    'validation_indices': sha_array(VAL_INDICES),
    'mask_ids': sha_array(MIX_MASK_IDS),
    'noise_levels': sha_array(MIX_LEVELS),
    'bw_noise': sha_array(NOISE['BW']),
    'em_noise': sha_array(NOISE['EM']),
    'ma_noise': sha_array(NOISE['MA']),
}
(OUT / 'data_fingerprint.json').write_text(
    json.dumps(DATA_FINGERPRINT, indent=2), encoding='utf-8')
manifest = TRAIN_META[['sample_id', 'record_id']].copy()
manifest['split'] = np.where(
    np.isin(np.arange(len(TRAIN_CLEAN)), TRAIN_INDICES), 'train', 'validation')
manifest['mask'] = np.asarray(MIX_NAMES, dtype=object)[MIX_MASK_IDS]
manifest['noise_level_r'] = MIX_LEVELS
manifest.to_csv(OUT / 'mixed_pair_manifest.csv', index=False)

probe_clean, probe_noisy = next(iter(make_loader(
    'MIX7', PROTOCOLS_TO_TRAIN[0], TRAIN_INDICES[:128], False, False)))
assert probe_clean.shape[1:] == (1, LENGTH) and probe_noisy.shape == probe_clean.shape
assert torch.isfinite(probe_clean).all() and torch.isfinite(probe_noisy).all()
print('MIX7 loader probe:', tuple(probe_clean.shape), tuple(probe_noisy.shape))
'''


def build(variant_id, hnf, timestep, attention):
    source_variant = "u3" if variant_id == "u4" else variant_id
    source_path = NOTEBOOKS / f"kaggle_ablation_{source_variant}_qtdb_bw_em_ma_512_100ep.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    destination = NOTEBOOKS / f"kaggle_ablation_{variant_id}_all_noise_512_100ep.ipynb"

    set_cell(cells[0], f"""# {variant_id.upper()} — one QTDB model for all BW/EM/MA noise subsets (512 samples)

This standalone Kaggle notebook trains **one** conditional DDPM model on a
balanced deterministic mix of seven nonempty artifact subsets: BW, EM, MA,
BW+EM, BW+MA, EM+MA, BW+EM+MA. It does **not** train three separate models.
Each noisy pair uses the same 512-sample QTDB clean segment, NSTDB artifact
patches and relative peak-to-peak level `r` across U0–U4. The three-source
patches are summed first and the resulting mixture is scaled once.

Architecture: HNF={hnf}; timestep={timestep}; attention={attention}. The
conditional DDPM, multi-domain loss, 70/30 clean-segment train/validation
split, optimizer, EMA and 100-epoch budget match the 512-sample research
training setup. **Protocol 1** uses the first NSTDB half/channel 1 for
training; held-out QTDB records and the other noise half/channel belong to
the future common test evaluation. Test data do not select checkpoints.

Enable a Kaggle GPU and Internet. Optional input caches are discovered under
`/kaggle/input`; no source-code upload is required. If a session is stopped,
attach its prior output and rerun to resume from `last`.
""")
    set_cell(cells[17], "## 9. Fixed seven-subset mixed-noise dataset and fingerprint\n")

    configuration = cell_source(cells[4])
    artifact_line = next(line for line in configuration.splitlines()
                         if line.startswith("ARTIFACTS_TO_TRAIN ="))
    configuration = configuration.replace(
        artifact_line, "ARTIFACTS_TO_TRAIN = ('MIX7',)  # One mixed model",
    )
    assert "ARTIFACTS_TO_TRAIN = ('MIX7',)" in configuration
    configuration = configuration.replace(
        "CHECKPOINT_TAG = '" + {
            "u0": "u0_conv_addtime_noattn_100ep_v1",
            "u1": "u1_hnf_addtime_noattn_100ep_v1",
            "u2": "u2_conv_film_noattn_100ep_v1",
            "u3": "u3_hnf_film_noattn_100ep_v1",
            "u4": "u3_hnf_film_noattn_100ep_v1",
        }[variant_id] + "'",
        "CHECKPOINT_TAG = 'mixed512_100ep_v1'",
    )
    configuration = configuration.replace(
        f"qtdb_{source_variant}_512_100ep",
        f"qtdb_{variant_id}_all_noise_512_100ep",
    )
    old_check = "if not ARTIFACTS_TO_TRAIN or not set(ARTIFACTS_TO_TRAIN).issubset(VALID_ARTIFACTS):"
    assert old_check in configuration
    configuration = configuration.replace(old_check,
                                          "if ARTIFACTS_TO_TRAIN != ('MIX7',):")
    configuration = configuration.replace(
        "raise ValueError(f'ARTIFACTS_TO_TRAIN must be selected from {tuple(VALID_ARTIFACTS)}')",
        "raise ValueError(\"This notebook requires ARTIFACTS_TO_TRAIN=('MIX7',)\")",
    )
    configuration = configuration.replace(
        "    'preprocessing_version': PREPROCESSING_VERSION,",
        "    'preprocessing_version': PREPROCESSING_VERSION,\n"
        "    'data_protocol': 'qtdb_nstdb_balanced_7_nonempty_masks_512_v1',",
    )
    assert "TARGET_EPOCHS = 100" in configuration
    assert "PROTOCOLS_TO_TRAIN = (1,)" in configuration
    assert "'data_protocol': 'qtdb_nstdb_balanced_7_nonempty_masks_512_v1'" in configuration
    retry_helper = '''\n\ndef wfdb_read_with_retry(operation, *args, **kwargs):
    """Retry transient PhysioNet errors without skipping or changing records."""
    for attempt in range(5):
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            transient = any(token in message for token in (
                '429', '500 error', '502', '503', '504', 'bad gateway',
                'timeout', 'timed out', 'connection error',
                'connection reset', 'temporarily unavailable',
            ))
            if not transient or attempt == 4:
                raise
            delay = min(2 ** (attempt + 1), 16)
            print(f'PhysioNet transient error ({exc}); retry {attempt + 1}/4 '
                  f'in {delay}s')
            time.sleep(delay)
'''
    assert "import time\n" in configuration
    configuration = configuration.replace("import wfdb\n", "import wfdb\n" + retry_helper, 1)
    set_cell(cells[4], configuration)

    for index in (10, 14, 16):
        source = cell_source(cells[index])
        source = source.replace("wfdb.rdrecord(", "wfdb_read_with_retry(wfdb.rdrecord, ")
        source = source.replace("wfdb.rdann(", "wfdb_read_with_retry(wfdb.rdann, ")
        source = source.replace("wfdb.get_record_list(",
                                "wfdb_read_with_retry(wfdb.get_record_list, ")
        set_cell(cells[index], source)

    if variant_id == "u4":
        architecture = cell_source(cells[6])
        attention_code = ATTENTION_SOURCE.read_text(encoding="utf-8")
        attention_code = attention_code[attention_code.index(
            "class SelfAttention1D(nn.Module):"):].strip() + "\n\n\n"
        insert_at = architecture.index("class UNet1D(nn.Module):")
        architecture = (architecture[:insert_at] + attention_code
                        + architecture[insert_at:])
        architecture = architecture.replace("self.attn = nn.Identity()",
                                            "self.attn = SelfAttention1D(c * 8)")
        architecture = architecture.replace("U3: same four-level", "U4: same four-level")
        architecture = architecture.replace("U3 preflight:", "U4 preflight:")
        assert "class SelfAttention1D(nn.Module)" in architecture
        set_cell(cells[5], "## 3. U4: original HNF + FiLM + bottleneck attention\n")
        set_cell(cells[6], architecture)
        configuration = cell_source(cells[4]).replace("'id': 'U3'", "'id': 'U4'")
        configuration = configuration.replace("'attention': 'identity'",
                                              "'attention': 'self_attention_4head'")
        set_cell(cells[4], configuration)

    set_cell(cells[18], MIXED_DATA_CELL)

    training = cell_source(cells[20])
    if variant_id == "u4":
        training = training.replace("u3_{artifact.lower()}", "u4_{artifact.lower()}")
        training = training.replace("'ablation_id': 'U3'", "'ablation_id': 'U4'")
        training = training.replace("payload.get('ablation_id') != 'U3'",
                                    "payload.get('ablation_id') != 'U4'")
    assert f"'ablation_id': '{variant_id.upper()}'" in training
    training = training.replace("{'BW': 11, 'EM': 22, 'MA': 33}[artifact]",
                                "{'MIX7': 44}[artifact]")
    training = training.replace("        'ablation_id':", 
                                "        'data_fingerprint': DATA_FINGERPRINT,\n"
                                "        'ablation_id':", 1)
    old_resume_check = "        if payload.get('config') != CONFIG or tuple(payload.get('train_records', ())) != TRAIN_RECORDS:"
    assert old_resume_check in training
    training = training.replace(old_resume_check,
        "        if (payload.get('config') != CONFIG or\n"
        "                tuple(payload.get('train_records', ())) != TRAIN_RECORDS or\n"
        "                payload.get('data_fingerprint') != DATA_FINGERPRINT):")
    set_cell(cells[20], training)

    package = cell_source(cells[22]).replace(
        f"qtdb_{'u3' if variant_id == 'u4' else variant_id}_512_100ep_outputs.zip",
        f"qtdb_{variant_id}_all_noise_512_100ep_outputs.zip",
    )
    set_cell(cells[22], package)
    set_cell(cells[23], f"""## Checkpoint and comparison contract

Best checkpoint: `best_{variant_id}_mix7_protocol1_mixed512_100ep_v1.pth`.
It stores model and EMA states, {variant_id.upper()} identity, training config,
epoch, parameter count and the data fingerprint. Every U0–U4 notebook must
produce the **same** `data_fingerprint.json`; compare those files before
claiming a paired architecture ablation. Use the EMA state and one common
DDIM/test/metric pipeline. Existing specialist BW/EM/MA checkpoints and the
3600-sample benchmark checkpoint are not members of this MIX7 ablation.
""")

    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)
                           + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    selected = set(sys.argv[1:]) or {variant[0] for variant in VARIANTS}
    unknown = selected - {variant[0] for variant in VARIANTS}
    if unknown:
        raise SystemExit(f"Unknown variant(s): {sorted(unknown)}")
    for variant in VARIANTS:
        if variant[0] in selected:
            build(*variant)
