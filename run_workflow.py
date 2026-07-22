import os
import argparse
import yaml
import numpy as np
import torch
from src.data_prep.data_preparation import Data_Preparation
from src.training.trainer import train_model
from src.inference.ddim_sampler import DDIMDenoiser
from src.evaluation import metrics

TEST_RECORDS = [str(i) for i in range(180, 201)]

def main(mode, model_choice=None):
    if mode == 'preprocess':
        print("=== Preprocessing: Generating synthesized data ===")
        # Gọi Data_Preparation sẽ tự động thực hiện resample, segment, normalize, synthesize
        # và trả về numpy arrays (nhưng ta không cần dùng, chỉ cần tạo file)
        Data_Preparation(n_type=1, force_rebuild=True, test_records=TEST_RECORDS)
        print("Preprocessing done. Data saved in data/synthesis/")

    elif mode == 'train':
        print("=== Training model ===")
        train_model('configs/base.yaml', model_choice=model_choice, device='cuda:0' if torch.cuda.is_available() else 'cpu')

    elif mode == 'infer':
        print("=== Running inference on a sample ===")
        # Lấy một mẫu test (đầu tiên)
        noisy_sample = np.load('data/synthesis/noisy.npy')[0]   # (512,)
        clean_sample = np.load('data/synthesis/clean.npy')[0]
        denoiser = DDIMDenoiser(config_path='configs/base.yaml',
                                checkpoint='checkpoints/model.pth')
        recon = denoiser.denoise(noisy_sample, ddim_steps=50, eta=0.0, num_shots=1)
        # clean_sample is (512, 12), recon is (12, 512) after squeeze
        # Transpose clean_sample to match (12, 512)
        clean_12 = clean_sample.T
        
        # Calculate metrics for each channel then average
        ssd = np.mean(metrics.SSD(clean_12, recon))
        mad = np.mean(metrics.MAD(clean_12, recon))
        prd = np.mean(metrics.PRD(clean_12, recon))
        
        # COS_SIM requires 3D or specific handling? Let's just expand dims to (12, 512, 1) for the existing function
        cos = np.mean(metrics.COS_SIM(clean_12[..., np.newaxis], recon[..., np.newaxis]))
        
        print(f"Mean across 12 channels - SSD: {ssd:.4f}, MAD: {mad:.4f}, PRD: {prd:.2f}%, Cosine: {cos:.4f}")

    elif mode == 'eval':
        print("=== Running evaluation and comparison ===")
        from evaluate_and_plot import evaluate_and_plot
        evaluate_and_plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preprocess', 'train', 'infer', 'eval'], default='preprocess')
    parser.add_argument('--model', choices=['1', '2'], default='1', help='[1] 1D U-Net, [2] ConditionalModel (Bài báo gốc)')
    args = parser.parse_args()
    main(args.mode, args.model)