import os
import numpy as np
from src.data_prep.resampler import process_ludb
from src.data_prep.segmenter import segment_signals
from src.data_prep.normalizer import normalize_segments
from src.data_prep.synthesizer import synthesize_noisy

def Data_Preparation(n_type=1, force_rebuild=False, test_records=None):
    """
    Bắt chước interface của Data_Preparation gốc.
    Trả về (X_train, y_train, X_test, y_test) dạng numpy arrays.
    Nếu các file đã tồn tại và không force_rebuild, thì tải từ disk.
    """
    data_dir = 'data/synthesis'
    clean_file = os.path.join(data_dir, 'clean.npy')
    noisy_file = os.path.join(data_dir, 'noisy.npy')
    labels_file = 'data/processed/segments_512/labels.npy'

    if not force_rebuild and os.path.exists(clean_file) and os.path.exists(noisy_file):
        print("Loading existing synthesized data...")
        clean = np.load(clean_file)
        noisy = np.load(noisy_file)
        labels = np.load(labels_file) if os.path.exists(labels_file) else None
    else:
        print("Preprocessing data from scratch...")
        # 1. Resample
        process_ludb('data/raw/ludb_database', 'data/processed/clean_500hz')
        # 2. Segment
        seg, labels = segment_signals('data/processed/clean_500hz', 'data/processed/segments_512',
                                      test_records=test_records or [])
        # 3. Normalize
        seg_norm = normalize_segments(seg)
        np.save('data/processed/segments_512/segments_norm.npy', seg_norm)
        # 4. Synthesize
        synthesize_noisy(seg_norm, 'data/raw/mit_bih_nst', data_dir)
        clean = np.load(clean_file)
        noisy = np.load(noisy_file)

    # Phân chia train/test dựa trên labels
    if labels is not None:
        # labels là mảng string, ta phân chia theo record
        unique_records = np.unique(labels)
        # Giả sử test_records đã được truyền vào, nếu không thì dùng danh sách mặc định
        if test_records is None:
            test_records = ['sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853',
                            'sel16420', 'sel16795', 'sel0106', 'sel0121', 'sel32',
                            'sel49', 'sel14046', 'sel15815']
        train_mask = ~np.isin(labels, test_records)
        test_mask = np.isin(labels, test_records)
        X_train = clean[train_mask]
        y_train = noisy[train_mask]
        X_test = clean[test_mask]
        y_test = noisy[test_mask]
    else:
        # fallback: dùng split 70-30
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(clean, noisy, test_size=0.3, random_state=42)

    # Expand dims fallback for 1 channel (if somehow we have 1 channel data without channel dim)
    if X_train.ndim == 2:
        X_train = X_train[..., np.newaxis]
        y_train = y_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        y_test = y_test[..., np.newaxis]

    return X_train, y_train, X_test, y_test