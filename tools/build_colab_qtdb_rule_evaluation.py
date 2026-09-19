import json, textwrap
from pathlib import Path

root=Path.cwd()
src=json.loads((root/'notebooks/kaggle_qtdb_rule_benchmark.ipynb').read_text(encoding='utf-8'))
out=root/'notebooks/colab_qtdb_rule_evaluation.ipynb'

def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)}
def body(i): return ''.join(src['cells'][i]['source'])
def unguard(i):
    s=body(i)
    assert s.startswith('if READY_FOR_TEST:\n')
    return textwrap.dedent(s[len('if READY_FOR_TEST:\n'):])

imports=body(3)
imports=imports.replace("OUT = Path('/kaggle/working/qtdb_rule_eval') if Path('/kaggle/working').exists() else Path('qtdb_rule_eval')", "OUT = Path('/content/drive/MyDrive/qtdb_rule_evaluation_100epoch')")

checkpoint_cell=r'''# Sửa đúng thư mục chứa hai checkpoint best trên Google Drive.
CHECKPOINT_DIR = Path('/content/drive/MyDrive/qtdb_rule_eval/train')
CHECKPOINT_PATHS = {
    1: CHECKPOINT_DIR / 'best_protocol1.pth',
    2: CHECKPOINT_DIR / 'best_protocol2.pth',
}
DECLARED_TRAIN_EPOCHS = 100
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'train': {'feats': 80, 'batch_size': 96, 'epochs': DECLARED_TRAIN_EPOCHS,
              'lr': 1e-4, 'lambda_time': 1.0, 'lambda_freq': 0.1},
    'diffusion': {'beta_start': 1e-4, 'beta_end': 0.5,
                  'num_steps': 50, 'schedule': 'quad'},
}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

MODELS, CHECKPOINT_INFO = {}, []
for protocol, path in CHECKPOINT_PATHS.items():
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy checkpoint protocol {protocol}: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise RuntimeError(f'{path.name} không phải checkpoint best do notebook train tạo ra')
    if int(payload.get('protocol', -1)) != protocol:
        raise RuntimeError(f'{path.name}: protocol trong checkpoint không khớp')
    saved_diffusion = payload.get('config', {}).get('diffusion', {})
    if saved_diffusion and saved_diffusion != CONFIG['diffusion']:
        raise RuntimeError(f'{path.name}: cấu hình diffusion không khớp: {saved_diffusion}')
    model = DDPM(UNet1D(2, 80, 128, 1), CONFIG, DEVICE).to(DEVICE)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    probe = torch.zeros(1, 1, LENGTH, device=DEVICE)
    with torch.inference_mode():
        assert model.model(probe, probe, torch.ones(1, 1, device=DEVICE)).shape == probe.shape
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
          f"validation={info['best_validation_loss']:.6f}")
display(pd.DataFrame(CHECKPOINT_INFO))
'''

noise_cell='''noise_record = wfdb.rdrecord('bw', pn_dir=PHYSIONET_NST)
BW = noise_record.p_signal.astype(np.float32)
if int(noise_record.fs) != TARGET_FS:
    BW = signal.resample(BW, int(round(len(BW) * TARGET_FS / float(noise_record.fs))), axis=0).astype(np.float32)
assert BW.ndim == 2 and BW.shape[1] >= 2 and np.isfinite(BW).all()
MIDPOINT = len(BW) // 2
print('NSTDB BW:', BW.shape, '| midpoint:', MIDPOINT)
'''

plot=unguard(29)
plot=plot.replace("'full_benchmark': TRAIN_RECORD_LIMIT is None and MAX_TEST_RECORDS is None and MAX_TEST_SEGMENTS is None,", "'full_benchmark': MAX_TEST_RECORDS is None and MAX_TEST_SEGMENTS is None,")
plot=plot.replace("'training_history_verified_from_checkpoint': True,\n    'train_records': list(TRAIN_RECORDS),\n    'train_N': len(TRAIN_INDICES), 'validation_N': len(VAL_INDICES),", "'declared_training_completed_epochs': DECLARED_TRAIN_EPOCHS,\n    'checkpoint_selection': 'minimum validation loss during the completed 100-epoch training run',")

cells=[
md('''# Đánh giá QTDB/BW trên Colab — checkpoint 100 epoch\n\nNotebook này **chỉ đánh giá**, không huấn luyện lại. Nó nạp hai checkpoint validation-best của protocol 1/2, tạo lại 14 QTDB test records và hai nửa BW test theo `rule.md`, chạy DDIM K = 1/3/5/10, rồi xuất metric per-sample và bảng tổng hợp vào Google Drive.\n\nTrước khi chạy, đặt `best_protocol1.pth` và `best_protocol2.pth` trong thư mục cấu hình ở cell checkpoint. Bật GPU và Internet trong Colab.'''),
md('## 1. Cài thư viện'), code(body(2).replace('%pip','!pip')),
md('## 2. Mount Google Drive'), code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
md('## 3. Cấu hình đánh giá'), code(imports),
md('## 4. Kiến trúc HNF U-Net + FiLM + attention'), code(body(5)),
md('## 5. DDPM/DDIM và loss đa miền của nghiên cứu'), code(body(7)),
md('## 6. Nạp hai checkpoint best'), code(checkpoint_cell),
md('## 7. Hàm tạo QTDB ground truth'), code(body(9)),
md('## 8. Tải NSTDB BW'), code(noise_cell),
md('## 9. Mở 14 record test'), code(unguard(21)),
md('## 10. Tạo hai protocol BW và manifest'), code(unguard(23)),
md('## 11. Chạy DDIM K = 1/3/5/10'), code(unguard(25)),
md('## 12. Kiểm toán và tổng hợp metric'), code(unguard(27)),
md('## 13. Biểu đồ và metadata'), code(plot),
md('''## Tệp kết quả\n\n- `metrics_per_sample.csv`: SSD, MAD, PRD_repo, PRD_standard, cosine, SNR-in/out và ΔSNR từng đoạn.\n- `summary.csv` và `summary_mean_std.csv`: protocol 1, protocol 2, pooled và nhóm r.\n- `summary_by_record.csv`: tổng hợp theo QT record.\n- `manifest.csv`: cặp clean/noisy, kênh BW, vị trí patch và r.\n- `xhat_protocol*_K*.npy`: tín hiệu đầu ra để kiểm tra lại metric.\n- `run_info.json`: hash checkpoint và cấu hình chạy.''')]

nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(out)
