"""Build the mixed QTDB notebook with the prior BW/EM/MA research objective."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "kaggle_train_proposed_tfcdiff_mixed_3600.ipynb"
PRIOR_NOTEBOOK = ROOT / "notebooks" / "kaggle_train_qtdb_bw_em_ma.ipynb"
TARGET = ROOT / "notebooks" / "kaggle_train_proposed_research_mixed_3600.ipynb"


def get_source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def replace_required(value: str, old: str, new: str, label: str) -> str:
    if old not in value:
        raise RuntimeError(f"Cannot locate {label}: {old!r}")
    return value.replace(old, new)


notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
prior = json.loads(PRIOR_NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]
prior_loss_cell = get_source(prior["cells"][8])
prior_ast = ast.parse(prior_loss_cell)
research_class = next(
    ast.get_source_segment(prior_loss_cell, node)
    for node in prior_ast.body
    if isinstance(node, ast.ClassDef) and node.name == "ResearchLossDDPM"
)
assert research_class is not None
loss_hash = hashlib.sha256(research_class.encode("utf-8")).hexdigest()

set_source(cells[0], """# Conditional DDPM nghiên cứu — QTDB mixed BW+EM+MA, 3600 mẫu

Notebook huấn luyện trên `common_qtdb_mixed_bwem_ma_3600_ntype1.npz` bằng kiến trúc U-Net HNF–FiLM–attention và **ResearchLossDDPM** đã dùng cho các checkpoint BW/EM/MA trước. Hàm loss được chép nguyên từ notebook nghiên cứu đó và ghi SHA-256 trong cấu hình.

Run này bắt đầu từ epoch 0 với checkpoint/tên output riêng. Checkpoint của run mixed dùng source loss L1-sum trước đây không tương thích để resume.
""")
set_source(cells[1], """## 1. Input Kaggle

Gắn đúng một Dataset chứa `common_qtdb_mixed_bwem_ma_3600_ntype1.npz`. File hiện có gồm `clean_train/noisy_train` (22.804), `clean_val/noisy_val` (9.774), `clean_test/noisy_test` (5.012), `record_id_*`, `q_*`, `sample_id_test`, `train_idx`, `val_idx`, `noise_version` và `sampling_rate`.

Để tiếp tục run đã lưu epoch 40, gắn thêm Kaggle Dataset chứa file **giải nén** `last_proposed_research_mixed_3600.pth`; nên gắn cả `best_proposed_research_mixed_3600.pth` và `training_proposed_research_mixed_3600.csv` nếu có. Notebook dò file bằng tên trong `/kaggle/input`, không mở checkpoint nằm riêng bên trong ZIP.

Notebook giữ nguyên các cặp/split đã lưu, kiểm tra 14 QTDB test record, shape và tính hữu hạn. Dữ liệu không có `protocol_name`/`n_type` nên `protocol_verified=False` được ghi rõ trong báo cáo. Đây là file đã được dùng làm input mixed; không ghép với NPZ BW-only.
""")
set_source(cells[2], """## 2. Cấu hình nghiên cứu

Giữ U-Net HNF–FiLM–attention, DDPM 50 bước, `beta_start=1e-4`, `beta_end=0.5`, quadratic schedule, `lambda_time=1`, `lambda_freq=0.1`, Adam `1e-4` không weight decay, StepLR `(150, 0.1)`, clip gradient 1.0 và EMA 0.9. Loss giống checkpoint BW/EM/MA: L1 *mean*; STFT Hann tương đối có trọng số SNR và khôi phục bằng alpha liên tục.

`configs/base.yaml` đăng ký 400 epoch, batch 96. Trên T4, mặc định chạy micro-batch 4 × 24 lần tích lũy để cập nhật sau 96 mẫu; AMP được bật để tiết kiệm bộ nhớ. Việc chuẩn hóa theo noisy max-abs tuân theo proposal 3600 và cho phép inference không dùng clean test.
""")

set_source(cells[1], get_source(cells[1]) + "\nNotebook khóa SHA-256 của đúng bundle mixed hiện tại để bản BW-only hoặc cặp dữ liệu sinh lại không thể âm thầm thay thế khi resume. Vì NPZ không lưu hệ số và vị trí cắt riêng của BW/EM/MA, hash chỉ xác nhận đúng file, chưa chứng minh độc lập công thức trộn.\n")

config = get_source(cells[3])
config = replace_required(
    config,
    "MODE = 'train'                 # 'train' hoặc 'export'",
    "MODE = 'train'                 # 'train' hoặc 'export'\n"
    "REQUIRE_RESUME_CHECKPOINT = True  # phiên tiếp tục từ checkpoint epoch 40; False nếu cố ý train mới",
    "resume requirement",
)
for old, new in (
    ("MAX_EPOCHS_THIS_SESSION = 5", "MAX_EPOCHS_THIS_SESSION = 5"),
    ("REFERENCE_BATCH_SIZE = 128", "REFERENCE_BATCH_SIZE = 96"),
    ("SEED = 2024", "SEED = 1234"),
    ("'lr': 1e-3, 'weight_decay': 5e-4", "'lr': 1e-4, 'weight_decay': 0.0"),
    ("'name': 'proposed_hnf_ddpm_tfcdiff_mixed_bwem_ma_ntype1'",
     "'name': 'proposed_research_ddpm_tfcdiff_mixed_bwem_ma_ntype1'"),
    ("'data_protocol': 'tfcdiff_mixed_bwem_ma', 'n_type': 1,",
     "'data_protocol': 'tfcdiff_mixed_bwem_ma', 'n_type': 1,\n"
     "        'loss_version': 'l1_mean_relative_stft_snr_weighted_continuous_alpha_v1',\n"
     f"        'research_loss_sha256': '{loss_hash}',"),
    ("/kaggle/working/proposed_tfcdiff_mixed_3600",
     "/kaggle/working/proposed_research_mixed_3600"),
    ("TRAIN_BUNDLE_NAME = 'common_qtdb_mixed_bwem_ma_3600_ntype1.npz'",
     "TRAIN_BUNDLE_NAME = 'common_qtdb_mixed_bwem_ma_3600_ntype1.npz'\n"
     "EXPECTED_MIXED_BUNDLE_SHA256 = '517bfe0b825fadb05f272e73c205508de034f8d3d403e384b1e1a4afb6c26a4f'"),
):
    if old != new:
        config = replace_required(config, old, new, "research config")
set_source(cells[3], config)

data_cell = get_source(cells[7])
data_cell = replace_required(
    data_cell,
    "train_bundle_path = unique_input(TRAIN_BUNDLE_NAME)\ntrain_bundle = np.load(train_bundle_path, allow_pickle=True)",
    "train_bundle_path = unique_input(TRAIN_BUNDLE_NAME)\n"
    "actual_bundle_sha256 = sha256_file(train_bundle_path)\n"
    "if actual_bundle_sha256 != EXPECTED_MIXED_BUNDLE_SHA256:\n"
    "    raise RuntimeError(\n"
    "        f'Wrong mixed bundle SHA-256: {actual_bundle_sha256}; expected '\n"
    "        f'{EXPECTED_MIXED_BUNDLE_SHA256}. Do not attach BW-only or regenerated pairs.')\n"
    "train_bundle = np.load(train_bundle_path, allow_pickle=True)",
    "frozen mixed bundle hash",
)
data_cell = replace_required(
    data_cell,
    "train_bundle.close()",
    """# Audit the frozen 70/30 source indices shipped in this mixed bundle.
if {'train_idx', 'val_idx'}.issubset(train_bundle.files):
    source_train_idx = np.asarray(train_bundle['train_idx'], dtype=np.int64).reshape(-1)
    source_val_idx = np.asarray(train_bundle['val_idx'], dtype=np.int64).reshape(-1)
    if len(source_train_idx) != len(CLEAN_TRAIN) or len(source_val_idx) != len(CLEAN_VAL):
        raise ValueError('train_idx/val_idx length mismatch')
    if len(np.unique(source_train_idx)) != len(source_train_idx) or len(np.unique(source_val_idx)) != len(source_val_idx):
        raise ValueError('Duplicate source indices in mixed split')
    if np.intersect1d(source_train_idx, source_val_idx).size:
        raise ValueError('Source train/validation indices overlap')
    np.savez(REPORT_DIR / 'source_split_indices.npz',
             train_idx=source_train_idx, val_idx=source_val_idx)
noise_version = int(scalar(train_bundle, 'noise_version', -1))
if noise_version != 1:
    raise ValueError(f'Expected noise_version=1, got {noise_version}')
(REPORT_DIR / 'mixed_input_audit.json').write_text(
    json.dumps({'bundle_sha256': DATA_FINGERPRINTS['bundle_file'],
                'protocol_verified': protocol_verified, 'noise_version': noise_version,
                'sampling_rate': sampling_rate, 'n_type': n_type,
                'train_count': len(CLEAN_TRAIN), 'val_count': len(CLEAN_VAL),
                'test_count': len(CLEAN_TEST_LOCKED)}, indent=2), encoding='utf-8')
train_bundle.close()""",
    "frozen source split audit",
)
set_source(cells[7], data_cell)

model = get_source(cells[11])
model = replace_required(model, "return DDPM(base, MODEL_CONFIG, DEVICE).to(DEVICE)",
                         "return ResearchLossDDPM(base, MODEL_CONFIG, DEVICE).to(DEVICE)",
                         "research model construction")
model = replace_required(
    model,
    "probe = torch.zeros(1, 1, 3600, device=DEVICE)",
    "probe = torch.sin(torch.linspace(0, 8 * torch.pi, 3600, device=DEVICE)).view(1, 1, 3600)",
    "realistic preflight signal",
)
model = replace_required(
    model,
    "spectral_probe = stft_loss(probe, probe, n_fft=128, hop_length=64)\n"
    "    if not torch.isfinite(spectral_probe):\n"
    "        raise FloatingPointError('STFT preflight non-finite')",
    "research_probe = model.loss_components(probe, probe)\n"
    "    if not all(torch.isfinite(value) for value in research_probe.values()):\n"
    "        raise FloatingPointError('Research loss preflight non-finite')",
    "research loss preflight",
)
model = replace_required(
    model,
    "del strict_probe, probe, direct_output",
    "del strict_probe, probe, direct_output, research_probe",
    "preflight cleanup",
)
model = model.replace("STFT(128,64) OK; strict=True OK",
                      "research STFT(128,64) OK; strict=True OK")
model = (
    "import torch.nn.functional as F\n"
    "from src.models.main_model import default\n\n"
    "# Extracted verbatim from kaggle_train_qtdb_bw_em_ma.ipynb.\n"
    f"RESEARCH_LOSS_SHA256 = '{loss_hash}'\n"
    "assert RESEARCH_LOSS_SHA256 == MODEL_CONFIG['experiment']['research_loss_sha256']\n\n"
    + research_class + "\n\n" + model
)
set_source(cells[11], model)

old_names = {
    "last_proposed_tfcdiff_mixed_3600.pth": "last_proposed_research_mixed_3600.pth",
    "best_proposed_tfcdiff_mixed_3600.pth": "best_proposed_research_mixed_3600.pth",
    "training_proposed_tfcdiff_mixed_3600.csv": "training_proposed_research_mixed_3600.csv",
    "proposed_tfcdiff_mixed_ntype1_seed2024.npz": "proposed_research_mixed_ntype1_seed1234.npz",
    "proposed_hnf_ddpm_tfcdiff_mixed_bwem_ma_ntype1":
        "proposed_research_ddpm_tfcdiff_mixed_bwem_ma_ntype1",
    "/kaggle/working/proposed_tfcdiff_mixed_3600_outputs.zip":
        "/kaggle/working/proposed_research_mixed_3600_outputs.zip",
}
for index in range(12, len(cells)):
    value = get_source(cells[index])
    for old, new in old_names.items():
        value = value.replace(old, new)
    set_source(cells[index], value)

resume_cell = get_source(cells[13])
resume_cell = replace_required(
    resume_cell,
    "resume_path = resolve_resume_checkpoint()",
    "resume_path = resolve_resume_checkpoint()\n"
    "if REQUIRE_RESUME_CHECKPOINT and resume_path is None:\n"
    "    raise FileNotFoundError(\n"
    "        f'Không tìm thấy {LAST_NAME} trong Kaggle Input. Gắn checkpoint đã giải nén '\n"
    "        'trước khi Run All; notebook sẽ không khởi tạo run mới.')",
    "fail-fast resume guard",
)
resume_cell = replace_required(
    resume_cell,
    "    if previous_best.is_file():\n"
    "        shutil.copy2(previous_best, CHECKPOINT_DIR / BEST_NAME)",
    "    if previous_best.is_file():\n"
    "        shutil.copy2(previous_best, CHECKPOINT_DIR / BEST_NAME)\n"
    "    else:\n"
    "        print('WARNING: Chỉ tìm thấy last checkpoint; thiếu best checkpoint trước đây. '\n"
    "              'Training vẫn tiếp tục, nhưng cần best gốc để giữ epoch tốt nhất trước resume.')",
    "resume best checkpoint warning",
)
set_source(cells[13], resume_cell)

run_info = get_source(cells[23])
run_info = replace_required(
    run_info,
    "'protocol_verified': protocol_verified, 'n_type': n_type,",
    "'protocol_verified': protocol_verified, 'noise_version': noise_version, 'n_type': n_type,",
    "run info noise version",
)
set_source(cells[23], run_info)

training = r'''# Stop if repeated non-finite gradients indicate a persistent problem.
MAX_NONFINITE_STEPS_PER_EPOCH = 5

def validation_loss_fixed(model, epoch):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    torch.manual_seed(SEED + 1_000_000)
    torch.cuda.manual_seed_all(SEED + 1_000_000)
    model.eval()
    sums = {'total': 0.0, 'time': 0.0, 'frequency': 0.0,
            'frequency_unweighted': 0.0, 'frequency_weight': 0.0}
    examples = 0
    loader = make_loader(val_dataset, shuffle=False, epoch=0)
    try:
        with torch.no_grad():
            for clean, noisy in tqdm(loader, desc=f'Validation epoch {epoch + 1}', leave=False):
                clean = clean.to(DEVICE, non_blocking=True)
                noisy = noisy.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    components = model.loss_components(clean, noisy)
                if not all(torch.isfinite(value) for value in components.values()):
                    raise FloatingPointError('Validation loss NaN/Inf')
                for name in sums:
                    sums[name] += float(components[name]) * len(clean)
                examples += len(clean)
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    if examples == 0:
        raise RuntimeError('Validation set nhỏ hơn micro-batch')
    return {name: value / examples for name, value in sums.items()}

def append_log(row):
    path = REPORT_DIR / LOG_NAME
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)

session_end = TARGET_EPOCHS if MAX_EPOCHS_THIS_SESSION is None else min(
    TARGET_EPOCHS, start_epoch + MAX_EPOCHS_THIS_SESSION)

if MODE == 'train':
    if start_epoch >= TARGET_EPOCHS:
        print('Training đã đủ target; không chạy thêm epoch.')
    for epoch in range(start_epoch, session_end):
        model.train()
        sums = {'total': 0.0, 'time': 0.0, 'frequency': 0.0}
        examples = 0
        gradient_norms = []
        skipped_steps = 0
        nonfinite_gradient_steps = 0
        loader = make_loader(train_dataset, shuffle=True, epoch=epoch)
        usable_batches = (len(loader) // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
        if usable_batches == 0:
            raise RuntimeError('Train set nhỏ hơn effective batch 96')
        progress = tqdm(loader, desc=f'Epoch {epoch + 1}/{TARGET_EPOCHS}')
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (clean, noisy) in enumerate(progress):
            if batch_index >= usable_batches:
                break
            clean = clean.to(DEVICE, non_blocking=True)
            noisy = noisy.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                components = model.loss_components(clean, noisy)
                backward_loss = components['total'] / GRAD_ACCUM_STEPS
            if not all(torch.isfinite(value) for value in components.values()):
                raise FloatingPointError(f'Loss NaN/Inf tại epoch {epoch + 1}')
            scaler.scale(backward_loss).backward()
            if (batch_index + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    # Do not step the optimizer or EMA with invalid gradients.
                    # clip_grad_norm_ may have touched them; zero them immediately.
                    nonfinite_gradient_steps += 1
                    skipped_steps += 1
                    old_scale = scaler.get_scale()
                    if USE_AMP:
                        scaler.update(new_scale=max(old_scale * 0.5, 1.0))
                    else:
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    print(f'WARNING epoch {epoch + 1}, batch {batch_index + 1}: '
                          f'gradient norm={float(gradient_norm)}; skipped update; '
                          f'AMP scale {old_scale:g} -> {scaler.get_scale():g}')
                    if nonfinite_gradient_steps > MAX_NONFINITE_STEPS_PER_EPOCH:
                        raise FloatingPointError(
                            f'Quá {MAX_NONFINITE_STEPS_PER_EPOCH} bước gradient NaN/Inf '
                            f'tại epoch {epoch + 1}; cần kiểm tra loss/AMP thay vì tiếp tục im lặng.')
                else:
                    gradient_norms.append(float(gradient_norm))
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        ema.update(model)
                    else:
                        skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
            for name in sums:
                sums[name] += float(components[name].detach()) * len(clean)
            examples += len(clean)
            progress.set_postfix(loss=f'{sums["total"] / examples:.5f}')
        scheduler.step()
        train_values = {name: value / examples for name, value in sums.items()}
        val_values = validation_loss_fixed(model, epoch)
        val_loss = val_values['total']
        improved = val_loss < best_validation_loss
        if improved:
            best_validation_loss = val_loss
        append_log({
            'epoch': epoch + 1, 'train_loss': train_values['total'],
            'train_time_loss': train_values['time'],
            'train_frequency_loss': train_values['frequency'],
            'validation_loss': val_loss,
            'validation_time_loss': val_values['time'],
            'validation_frequency_loss': val_values['frequency'],
            'validation_frequency_unweighted': val_values['frequency_unweighted'],
            'validation_frequency_weight': val_values['frequency_weight'],
            'best_validation_loss': best_validation_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_norm_mean': float(np.mean(gradient_norms)),
            'gradient_norm_max': float(np.max(gradient_norms)),
            'nonfinite_gradient_steps': nonfinite_gradient_steps,
            'amp_skipped_steps': skipped_steps,
            'micro_batch_size': MICRO_BATCH_SIZE,
            'effective_batch_size': REFERENCE_BATCH_SIZE,
            'gradient_accumulation_steps': GRAD_ACCUM_STEPS,
            'amp': USE_AMP,
        })
        save_checkpoint(epoch, is_best=False)
        if improved:
            save_checkpoint(epoch, is_best=True)
        print(f'Epoch {epoch + 1}: train={train_values["total"]:.6f}; '
              f'val={val_loss:.6f}; best={best_validation_loss:.6f}'
              + (' [BEST]' if improved else ''))
        gc.collect(); torch.cuda.empty_cache()

last_candidates = checkpoint_candidates(LAST_NAME)
completed_epochs = start_epoch if not last_candidates else int(
    torch.load(last_candidates[0], map_location='cpu', weights_only=False)['epoch']) + 1
TRAINING_COMPLETE = completed_epochs >= TARGET_EPOCHS
print(f'Đã hoàn thành {completed_epochs}/{TARGET_EPOCHS} epoch. '
      + ('Có thể export.' if TRAINING_COMPLETE else
         'Save Version, gắn output làm Input và chạy lại để resume.'))
'''
set_source(cells[15], training)

set_source(cells[16], """## 9. Đường học của loss nghiên cứu

Log ghi riêng time loss, frequency loss, gradient norm và số bước AMP bỏ qua; dùng các cột này để kiểm tra ổn định qua các phiên.
""")
set_source(cells[17], r'''log_path = REPORT_DIR / LOG_NAME
if log_path.is_file():
    training_log = pd.read_csv(log_path).drop_duplicates('epoch', keep='last').sort_values('epoch')
    training_log.to_csv(log_path, index=False)
    display(training_log.tail(10))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(training_log.epoch, training_log.train_loss, label='Train', color='#3568A8')
    axes[0].plot(training_log.epoch, training_log.validation_loss, label='Validation', color='#D39C2C')
    axes[0].plot(training_log.epoch, training_log.best_validation_loss,
                 label='Best validation', color='#333333', linestyle='--')
    axes[0].set(title='Total research loss', xlabel='Epoch', ylabel='Normalized loss')
    axes[0].legend()
    axes[1].plot(training_log.epoch, training_log.train_time_loss,
                 label='Time (train)', color='#3568A8')
    axes[1].plot(training_log.epoch, training_log.validation_time_loss,
                 label='Time (validation)', color='#D39C2C')
    axes[1].plot(training_log.epoch, training_log.train_frequency_loss,
                 label='Frequency (train)', color='#3568A8', linestyle='--')
    axes[1].plot(training_log.epoch, training_log.validation_frequency_loss,
                 label='Frequency (validation)', color='#D39C2C', linestyle='--')
    axes[1].set(title='Loss components before lambda', xlabel='Epoch', ylabel='Component loss')
    axes[1].legend(fontsize=8)
    axes[2].plot(training_log.epoch, training_log.gradient_norm_mean,
                 label='Mean', color='#3568A8')
    axes[2].plot(training_log.epoch, training_log.gradient_norm_max,
                 label='Maximum', color='#D39C2C', alpha=.7)
    axes[2].set(title='Gradient norm before clipping', xlabel='Epoch', ylabel='L2 norm')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('QTDB mixed BW+EM+MA — Research DDPM', fontsize=13)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'training_history.png', dpi=180, bbox_inches='tight')
    plt.show()
''')

set_source(cells[24], """## Tiếp tục và xuất benchmark

- Chạy với GPU T4 và Input `common_qtdb_mixed_bwem_ma_3600_ntype1.npz`. `MAX_EPOCHS_THIS_SESSION=5` là mặc định an toàn; có thể tăng sau khi đo thời gian mỗi epoch.
- Sau mỗi phiên, **Save Version** rồi gắn output đó làm Input ở phiên tiếp theo. Notebook chỉ resume `last_proposed_research_mixed_3600.pth`; nếu có thì gắn cả `best_proposed_research_mixed_3600.pth` và training CSV. `REQUIRE_RESUME_CHECKPOINT=True` khiến notebook dừng sớm nếu thiếu `last`. Từ checkpoint epoch 40, `MAX_EPOCHS_THIS_SESSION=5` sẽ chạy epoch 41–45.
- Khi gặp một gradient NaN/Inf, notebook bỏ qua optimizer/EMA update của effective batch đó, giảm AMP scale và ghi cảnh báo. Nếu quá 5 lần trong một epoch, notebook vẫn dừng để tránh che giấu lỗi kéo dài.
- Khi đủ 400 epoch, output `proposed_research_mixed_ntype1_seed1234.npz` chứa prediction, ID và latency cho notebook benchmark chung.
""")

for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        ast.parse(get_source(cell), filename=f"cell_{index}")

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {TARGET}")
print(f"Research loss SHA-256: {loss_hash}")
