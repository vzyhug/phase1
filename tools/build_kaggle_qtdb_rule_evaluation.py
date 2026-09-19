import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = json.loads((root / 'notebooks/colab_qtdb_rule_evaluation.ipynb').read_text(encoding='utf-8'))
out = root / 'notebooks/kaggle_qtdb_rule_evaluation.ipynb'

def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)}
def body(i): return ''.join(src['cells'][i]['source'])

imports = body(6).replace("OUT = Path('/content/drive/MyDrive/qtdb_rule_evaluation_100epoch')",
                          "OUT = Path('/kaggle/working/qtdb_rule_evaluation_100epoch')")

checkpoint = r'''# Gắn Kaggle Dataset/Notebook Output chứa đúng hai checkpoint bằng Add Input.
CHECKPOINT_FILENAMES = {
    1: 'best_protocol1.pth',
    2: 'best_protocol2.pth',
}
DECLARED_TRAIN_EPOCHS = 100
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'train': {'feats': 80, 'batch_size': 96, 'epochs': DECLARED_TRAIN_EPOCHS,
              'lr': 1e-4, 'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
}

def locate_unique_checkpoint(filename):
    root = Path('/kaggle/input')
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Cần đúng một file {filename} trong Kaggle Add Input; tìm thấy {len(matches)}: {matches}'
        )
    return matches[0]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

MODELS, CHECKPOINT_INFO = {}, []
for protocol, filename in CHECKPOINT_FILENAMES.items():
    path = locate_unique_checkpoint(filename)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise RuntimeError(f'{filename} không phải checkpoint best do notebook train tạo ra')
    if int(payload.get('protocol', -1)) != protocol:
        raise RuntimeError(f'{filename}: protocol trong checkpoint không khớp')
    saved_diffusion = payload.get('config', {}).get('diffusion', {})
    if saved_diffusion and saved_diffusion != CONFIG['diffusion']:
        raise RuntimeError(f'{filename}: cấu hình diffusion không khớp: {saved_diffusion}')
    model = DDPM(UNet1D(2, 80, 128, 1), CONFIG, DEVICE).to(DEVICE)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    probe = torch.zeros(1, 1, LENGTH, device=DEVICE)
    with torch.inference_mode():
        probe_output = model.model(probe, probe, torch.ones(1, 1, device=DEVICE))
    assert probe_output.shape == probe.shape and torch.isfinite(probe_output).all()
    MODELS[protocol] = model
    info = {
        'protocol': protocol, 'path': str(path), 'sha256': sha256(path),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'selected_epoch': int(payload.get('epoch', -1)) + 1,
        'best_validation_loss': float(payload.get('best_validation_loss', np.nan)),
        'declared_training_completed_epochs': DECLARED_TRAIN_EPOCHS,
        'saved_training_config': payload.get('config', {}),
    }
    CHECKPOINT_INFO.append(info)
    print(f"Protocol {protocol}: strict load OK; best epoch={info['selected_epoch']}; "
          f"validation={info['best_validation_loss']:.6f}; {path}")
display(pd.DataFrame(CHECKPOINT_INFO))
'''

cells = [
md('''# Kaggle Save & Run All — đánh giá QTDB/BW từ checkpoint 100 epoch\n\nNotebook này **chỉ đánh giá**, không train lại. Trước khi chạy, dùng **Add Input** để gắn Dataset hoặc Notebook Output chứa đúng `best_protocol1.pth` và `best_protocol2.pth`. Bật **GPU** và **Internet**, sau đó chọn **Save Version → Save & Run All**.\n\nNotebook tự tìm checkpoint trong `/kaggle/input`, tạo lại 14 QTDB test records cùng hai protocol NSTDB BW theo `rule.md`, chạy DDIM K = 1/3/5/10 và lưu kết quả vào `/kaggle/working/qtdb_rule_evaluation_100epoch/`.'''),
md('## 1. Cài thư viện'), code(body(2)),
md('## 2. Cấu hình Save & Run All'), code(imports),
md('## 3. Kiến trúc HNF U-Net + FiLM + attention'), code(body(8)),
md('## 4. DDPM/DDIM và loss đa miền của nghiên cứu'), code(body(10)),
md('## 5. Tự tìm và nạp hai checkpoint best'), code(checkpoint),
md('## 6. Hàm tạo QTDB ground truth'), code(body(14)),
md('## 7. Tải NSTDB BW'), code(body(16)),
md('## 8. Mở 14 QTDB test records'), code(body(18)),
md('## 9. Tạo hai protocol BW và manifest'), code(body(20)),
md('## 10. Chạy DDIM K = 1/3/5/10'), code(body(22)),
md('## 11. Kiểm toán và tổng hợp metric'), code(body(24)),
md('## 12. Biểu đồ và metadata'), code(body(26)),
md('''## Output của Kaggle version\n\nSau khi Save & Run All hoàn tất, mở tab **Output** và tải thư mục `qtdb_rule_evaluation_100epoch`. Các tệp chính:\n\n- `metrics_per_sample.csv`\n- `summary.csv` và `summary_mean_std.csv`\n- `summary_by_record.csv`\n- `manifest.csv`\n- `xhat_protocol*_K*.npy`\n- `run_info.json`\n- `summary_by_K.png` và `waveform_example.png`''')]

nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(out)
