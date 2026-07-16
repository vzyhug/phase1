import os
import wfdb
import numpy as np

def synthesize_noisy(clean_segments, noise_path, output_dir, delta_range=(0.2, 2.0), seed=42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    # Đọc noise từ MIT-BIH NST (kênh đầu)
    noise_sig, _ = wfdb.rdsamp(noise_path)
    noise_sig = noise_sig[:, 0]
    noisy_list, clean_list, deltas = [], [], []
    for seg in clean_segments:
        start = np.random.randint(0, len(noise_sig) - 512)
        noise_patch = noise_sig[start:start+512]
        # scale noise
        power_clean = np.mean(seg**2)
        power_noise = np.mean(noise_patch**2)
        scale = np.sqrt(power_clean / power_noise) if power_noise > 0 else 1.0
        delta = np.random.uniform(*delta_range)
        noise_scaled = noise_patch * scale * delta
        noisy = seg + noise_scaled
        noisy_list.append(noisy.astype(np.float32))
        clean_list.append(seg.astype(np.float32))
        deltas.append(delta)
    np.save(os.path.join(output_dir, 'noisy.npy'), np.array(noisy_list))
    np.save(os.path.join(output_dir, 'clean.npy'), np.array(clean_list))
    np.save(os.path.join(output_dir, 'deltas.npy'), np.array(deltas))
    print(f"Synthesized {len(noisy_list)} samples.")