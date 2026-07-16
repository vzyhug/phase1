import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from src.training.dataset import ECGDataset
from src.training.utils import train
from src.models.main_model import DDPM
from src.models.denoising_model_small import ConditionalModel

def train_model(config_path, device='cuda:0'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Dùng Data_Preparation để lấy dữ liệu
    from src.data_prep.data_preparation import Data_Preparation
    X_train, y_train, X_test, y_test = Data_Preparation(n_type=1, force_rebuild=False)

    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)

    # Split train/val
    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=0.3, random_state=42)
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], shuffle=False, drop_last=True)

    base_model = ConditionalModel(config['train']['feats']).to(device)
    model = DDPM(base_model, config, device)

    foldername = './checkpoints/'
    os.makedirs(foldername, exist_ok=True)

    train(model, config['train'], train_loader, device,
          valid_loader=val_loader,
          valid_epoch_interval=config['train'].get('valid_epoch_interval', 1),
          foldername=foldername)
    print("Training completed. Model saved in", foldername)