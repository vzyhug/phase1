import torch
import yaml
import numpy as np
from src.models.unet_1d import UNet1D          # <-- thay vì ConditionalModel
from src.models.main_model import DDPM

class DDIMDenoiser:
    def __init__(self, config_path='configs/base.yaml', checkpoint='checkpoints/model.pth', device='cuda:0'):
        self.device = device
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        base_model = UNet1D(
            in_channels=24,
            base_channels=self.config['train']['feats'],
            emb_dim=128,
            out_channels=12
        ).to(device)
        self.model = DDPM(base_model, self.config, device)
        state_dict = torch.load(checkpoint, map_location=device)
        # Lọc bỏ các buffer không khớp (nếu có)
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('model.')}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def denoise(self, noisy_signal, ddim_steps=50, eta=0.0, num_shots=1):
        if isinstance(noisy_signal, np.ndarray):
            # noisy_signal có shape (512, 12) -> (1, 12, 512)
            noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).permute(0, 2, 1).to(self.device)
        else:
            noisy_tensor = noisy_signal.to(self.device)
        with torch.no_grad():
            output = self.model.denoising(noisy_tensor, use_ddim=True,
                                          ddim_steps=ddim_steps, ddim_eta=eta, num_shots=num_shots)
        return output.squeeze().cpu().numpy()