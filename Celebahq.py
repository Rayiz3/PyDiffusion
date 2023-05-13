import functools
import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets import celeba

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("current device : {}".format(device))

class GaussianDiffusion():
    def __init__(self, betas: np.ndarray, loss_type, dtype=np.float32):
        # check beta
        assert (betas > 0).all() and (betas <= 1).all()  # 0 < beta <= 1
        
        # alpha and beta
        self.alphas = 1 - betas
        self.alphas_bar = np.cumprod(betas, 0)
        self.alphas_bar_prev = np.append(1., alphas_bar[:-1])
        self.betas = betas
        
        # operation of alpha
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.aqrt(1 - self.alphas_bar)
        self.log_one_minus_alphas_bar = np.log(1 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = np.sqrt(1/self.alphas_bar)
        self.sqrt_recip_minus_one_alphas = np.sqrt(1/self.alphas_bar - 1)
        
        # coefficient of posterior
        self.post_variance = (1 - self.alphas_bar_prev) / (1 - self.alphas_bar) * self.betas
        self.post_log_clip_variance = np.log(np.max(self.post_variance, 1e-20))  # because post_variance = 0 at first
        self.post_mean_x0 = np.sqrt(self.alphas_bar_prev) / (1 - self.alphas_bar) * self.betas
        self.post_mean_x1 = (1 - self.alphas_bar_prev) / (1 - self.alphas_bar) * np.sqrt(1 - self.betas)
        
        # others
        self.loss_type = loss_type
        self.time_steps = betas.shape
        self.num_time_steps = int(self.time_steps)
    
    @staticmethod

class Model(nn.Module):
    def __init__(self, model_name, betas: np.ndarray, loss_type: str, num_classes: int, randflip, block_size: int):
        self.model_name = model_name