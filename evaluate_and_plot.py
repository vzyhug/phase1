import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pywt
from scipy.signal import butter, filtfilt, medfilt

from src.evaluation import metrics
from src.inference.ddim_sampler import DDIMDenoiser

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = np.zeros_like(data)
    for i in range(data.shape[0]):
        y[i, :] = filtfilt(b, a, data[i, :])
    return y

def apply_median_filter(data, kernel_size=5):
    y = np.zeros_like(data)
    for i in range(data.shape[0]):
        y[i, :] = medfilt(data[i, :], kernel_size)
    return y

def wavelet_denoise(data, wavelet='db4', level=4):
    y = np.zeros_like(data)
    for i in range(data.shape[0]):
        coeffs = pywt.wavedec(data[i, :], wavelet, level=level)
        # Thresholding (soft)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(data[i, :])))
        coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        y[i, :] = pywt.waverec(coeffs_thresh, wavelet)
    return y

def evaluate_and_plot():
    clean_file = 'data/synthesis/clean.npy'
    noisy_file = 'data/synthesis/noisy.npy'
    
    if os.path.exists(clean_file) and os.path.exists(noisy_file):
        print("Đang tải dữ liệu test...")
        clean_sample = np.load(clean_file)[0] # Shape: (512, 12) hoặc (N, 12)
        noisy_sample = np.load(noisy_file)[0] 
        # Chuyển về dạng (12, 512)
        clean_12 = clean_sample.T
        noisy_12 = noisy_sample.T
    else:
        print("⚠️ Không tìm thấy file dữ liệu (data/synthesis/clean.npy). Đang tạo dữ liệu mẫu để minh họa mã...")
        t = np.linspace(0, 1, 512)
        clean_12 = np.array([np.sin(2 * np.pi * 5 * t + i) for i in range(12)])
        noisy_12 = clean_12 + np.random.normal(0, 0.5, clean_12.shape)

    results = {'Noisy': noisy_12}

    print("Đang áp dụng Butterworth Filter...")
    results['Butterworth'] = butter_lowpass_filter(noisy_12, cutoff=40.0, fs=360.0)
    
    print("Đang áp dụng Median Filter...")
    results['Median'] = apply_median_filter(noisy_12, kernel_size=7)
    
    print("Đang áp dụng Wavelet Denoising...")
    results['Wavelet'] = wavelet_denoise(noisy_12, wavelet='db4', level=4)

    print("Đang áp dụng Mô hình Deep Learning (DDIM)...")
    try:
        denoiser = DDIMDenoiser(config_path='configs/base.yaml',
                                checkpoint='checkpoints/model.pth',
                                device='cuda:0' if torch.cuda.is_available() else 'cpu')
        # noisy_12.T về lại shape (512, 12) cho DDIMDenoiser
        recon = denoiser.denoise(noisy_12.T, ddim_steps=50, eta=0.0, num_shots=1)
        results['DL_Model'] = recon
        print("Áp dụng DDIM thành công.")
    except Exception as e:
        print(f"⚠️ Không thể chạy mô hình DL (Có thể chưa có checkpoints/model.pth). Lỗi: {e}")

    print("\n" + "="*70)
    print(f"{'Method':<18} | {'SSD':<10} | {'MAD':<10} | {'PRD (%)':<10} | {'Cosine Sim':<12}")
    print("-" * 70)
    
    for name, pred in results.items():
        if pred is None:
            continue
            
        ssd = np.mean(metrics.SSD(clean_12, pred))
        mad = np.mean(metrics.MAD(clean_12, pred))
        prd = np.mean(metrics.PRD(clean_12, pred))
        
        # Mở rộng chiều (12, 512) -> (12, 512, 1) cho hàm COS_SIM hiện tại
        cos_sim = np.mean(metrics.COS_SIM(clean_12[..., np.newaxis], pred[..., np.newaxis]))
        
        print(f"{name:<18} | {ssd:<10.4f} | {mad:<10.4f} | {prd:<10.2f} | {cos_sim:<12.4f}")
    print("="*70)

    # Hiển thị biểu đồ 12 đạo trình
    print("\nĐang tạo biểu đồ 12 đạo trình...")
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    axes = axes.flatten()
    
    channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time_axis = np.arange(noisy_12.shape[1])
    
    colors = {
        'Butterworth': 'orange',
        'Median': 'green',
        'Wavelet': 'purple',
        'DL_Model': 'red'
    }
    
    for i in range(12):
        ax = axes[i]
        ax.plot(time_axis, clean_12[i], label='Clean (Ground Truth)', color='black', linewidth=2)
        ax.plot(time_axis, noisy_12[i], label='Noisy', color='gray', alpha=0.3)
        
        for name in ['Butterworth', 'Median', 'Wavelet', 'DL_Model']:
            if name in results and results[name] is not None:
                ax.plot(time_axis, results[name][i], label=name, color=colors[name], linewidth=1.5, alpha=0.8)
            
        ax.set_title(f"Đạo trình: {channels[i]}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300)
    print("Đã lưu hình ảnh so sánh kết quả vào file 'evaluation_results.png'")

if __name__ == '__main__':
    evaluate_and_plot()
