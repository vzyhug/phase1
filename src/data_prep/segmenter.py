import os
import numpy as np

def segment_signals(input_dir, output_dir, seg_len=512, stride_train=256, stride_test=512, test_records=None):
    os.makedirs(output_dir, exist_ok=True)
    all_segments, all_labels = [], []
    for f in os.listdir(input_dir):
        if not f.endswith('.npy'): continue
        record = f[:-4]
        data = np.load(os.path.join(input_dir, f))
        if data.ndim == 2:
            data = data[:, 0]   # lấy kênh đầu
        stride = stride_test if record in test_records else stride_train
        for start in range(0, len(data) - seg_len + 1, stride):
            seg = data[start:start+seg_len]
            all_segments.append(seg.astype(np.float32))
            all_labels.append(record)
    all_segments = np.array(all_segments)
    np.save(os.path.join(output_dir, 'segments.npy'), all_segments)
    np.save(os.path.join(output_dir, 'labels.npy'), np.array(all_labels))
    return all_segments, all_labels