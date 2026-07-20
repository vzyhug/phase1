import os
import csv
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .dataset import ECGDataset
from .utils import train
from src.models.unet_1d import UNet1D
from src.models.denoising_model_small import ConditionalModel
from src.models.main_model import DDPM

def train_model(config_path, model_choice=None, device='cuda:0'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    from src.data_prep.data_preparation import Data_Preparation
    X_train, y_train, X_test, y_test = Data_Preparation(n_type=1, force_rebuild=False)

    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)

    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.3, random_state=42)
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], shuffle=False, drop_last=True)

    # Mặc định sử dụng 1D U-Net nếu không được truyền vào
    if model_choice is None:
        model_choice = '1'

    if str(model_choice) == '1':
        print("-> Đã chọn: 1D U-Net")
        base_model = UNet1D(
            in_channels=24,                      # concat x_t (12) + cond (12)
            base_channels=config['train']['feats'],
            emb_dim=128,
            out_channels=12
        ).to(device)
    elif str(model_choice) == '2':
        print("-> Đã chọn: ConditionalModel (Bài báo gốc)")
        base_model = ConditionalModel(feats=config['train']['feats'], in_channels=12, out_channels=12).to(device)
    else:
        print("-> Lựa chọn không hợp lệ. Mặc định sử dụng: 1D U-Net")
        base_model = UNet1D(
            in_channels=24,
            base_channels=config['train']['feats'],
            emb_dim=128,
            out_channels=12
        ).to(device)

    model = DDPM(base_model, config, device)

    foldername = './checkpoints/'
    os.makedirs(foldername, exist_ok=True)

    train(model, config['train'], train_loader, device,
          valid_loader=val_loader,
          valid_epoch_interval=config['train'].get('valid_epoch_interval', 1),
          foldername=foldername)
    print("Training completed. Model saved in", foldername)