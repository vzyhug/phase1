import os
import argparse
import yaml
import numpy as np
import torch
from src.data_prep.data_preparation import Data_Preparation
from src.training.trainer import train_model
from src.inference.ddim_sampler import DDIMDenoiser
from src.evuluation import metrics

TEST_RECORDS = ['sel123', 'sel233', 'sel302', 'sel307', 'sel820', 'sel853',
                'sel16420', 'sel16795', 'sel0106', 'sel0121', 'sel32',
                'sel49', 'sel14046', 'sel15815']

def main(mode):
    if mode == 'preprocess':
        print("=== Preprocessing: Generating synthesized data ===")
        # Gọi Data_Preparation sẽ tự động thực hiện resample, segment, normalize, synthesize
        # và trả về numpy arrays (nhưng ta không cần dùng, chỉ cần tạo file)
        Data_Preparation(n_type=1, force_rebuild=True, test_records=TEST_RECORDS)
        print("Preprocessing done. Data saved in data/synthesis/")

    elif mode == 'train':
        print("=== Training model ===")
        train_model('configs/base.yaml', device='cuda:0' if torch.cuda.is_available() else 'cpu')

    elif mode == 'infer':
        print("=== Running inference on a sample ===")
        # Lấy một mẫu test (đầu tiên)
        noisy_sample = np.load('data/synthesis/noisy.npy')[0]   # (512,)
        clean_sample = np.load('data/synthesis/clean.npy')[0]
        denoiser = DDIMDenoiser(config_path='configs/base.yaml',
                                checkpoint='checkpoints/model.pth')
        recon = denoiser.denoise(noisy_sample, ddim_steps=50, eta=0.0, num_shots=1)
        # Tính metrics
        ssd = metrics.SSD(clean_sample.reshape(1, -1, 1), recon.reshape(1, -1, 1))[0]
        mad = metrics.MAD(clean_sample.reshape(1, -1, 1), recon.reshape(1, -1, 1))[0]
        prd = metrics.PRD(clean_sample.reshape(1, -1, 1), recon.reshape(1, -1, 1))[0]
        cos = metrics.COS_SIM(clean_sample.reshape(1, -1, 1), recon.reshape(1, -1, 1))[0][0][0]
        print(f"SSD: {ssd:.4f}, MAD: {mad:.4f}, PRD: {prd:.2f}%, Cosine: {cos:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preprocess', 'train', 'infer'], default='preprocess')
    args = parser.parse_args()
    main(args.mode)