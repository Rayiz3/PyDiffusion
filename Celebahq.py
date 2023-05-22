import functools
import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets import celeba

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("current device : {}".format(device))

def meanflat(x):
    return torch.mean(x, dim=list(range(1, len(x.shape))))


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
        self.one_minus_alphas_bar = 1 - self.alphas_bar
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
    ## index ## ==================================================
    # - extract corresponding elements
    # - inputs are all batched form with batch_size = B
    # [arguments]
    # - a : target tensor
    # - t : list of target index (B*1)
    # - x_shape : shape of data x (B*...)
    # ============================================================
    def index(a, t, x_shape):
        assert t.shape == x_shape[0]
        out = torch.gather(a, 0, t)
        return torch.reshape(out, [t.shape] + ((len(x_shape) - 1) * [1]))
    
    ## q_statistics ## ===========================================
    # - return mean / variance / log variance of q(x_t | x_0)
    # [arguments]
    # - x_0 : first data
    # - t : time step
    # ============================================================
    def q_statistics(self, x_0, t):
        mean = self.index(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        variance = self.index(self.one_minus_alphas_bar, t, x_0.shape)
        log_variance = self.index(self.log_one_minus_alphas_bar, t, x_0.shape)

        return mean, variance, log_variance
    
    ## q_sample ## ===============================================
    # - sample diffusion output of step t
    # [arguments]
    # - x_0 : first data
    # - t : time step
    # - noise : noise type
    # ============================================================
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn(x_0.shape)
            
        term_x_0 = self.index(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        term_noise = self.index(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        
        return term_x_0 + term_noise

    ## predict ## ================================================
    # - predict x_0 from noise at step t
    # [arguments]
    # - x_t : data at step t
    # - t : time step
    # - noise : noise type
    # ============================================================
    def predict(self, x_t, t, noise):
        assert x_t.shape == noise.shape
        term_x_t = self.index(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
        term_noise = self.index(self.sqrt_recip_minus_one_alphas, t, x_t.shape) * noise
        
        return term_x_t + term_noise

    ## q_post_statistics ## =======================================
    # - return mean / variance / log variance of q(x_t-1 | x_t, x_0)
    # [arguments]
    # - x_0 : first data
    # - x_t : data at step t
    # - t : time step
    # ============================================================
    def q_post_statistics(self, x_0, x_t, t):
        assert x_0.shape == x_1.shape
        term_x_0 = self.index(self.post_mean_x0, t, x_t.shape) * x_0
        term_x_t = self.index(self.post_mean_x1, t, x_t.shape) * x_t
        mean = term_x_0 + term_x_t
        variance = self.index(self.post_variance, t, x_t.shape)
        log_variance = self.index(self.post_log_clip_variance, t, x_t.shape)

        return mean, variance, log_variance

    ## p_loss ## =================================================
    # - return loss of predicted x_0
    # [arguments]
    # - denoise_fn : denoising function?
    # - x_0 : first data
    # - t : time step
    # - noise : noise type
    # ============================================================
    def p_loss(self, denoise_fn, x_0, t, noise):
        assert x_0.shape == noise.shape
        if noise is None:
            noise = torch.randn(x_0.shape)
        x_t = self.q_sample(x_0, t, noise=noise)
        noise_theta = denoise_fn(x_t, t)
        
        if self.loss_type == 'noisepred':
            return meanflat((noise - noise_theta) ** 2)
        else:
            raise NotImplementedError(self.loss_type)
    
    ## p_statistics ## ===========================================
    # - return mean / variance / log variance of p(x_t-1 | x_t)
    # [arguments]
    # - denoise_fn : denoising function?
    # - x_t : data at step t
    # - t : time step
    # - clip_denoised : if it is cliped between -1 ~ 1
    # ============================================================
    def p_statistics(self, denoise_fn, x_t, t, clip_denoised):
        if self.loss_type == 'noisepred':
            x_0 = self.predict(x_t, t, denoise_fn(x_t, t))
        else:
            raise NotImplementedError(self.loss_type)

        if clip_denoised:
            torch.clamp(x_0, -1, 1)
        
        return self.q_post_statistics(x_0, x_t, t)

class Model(nn.Module):
    def __init__(self, model_name, betas: np.ndarray, loss_type: str, num_classes: int, randflip, block_size: int):
        self.model_name = model_name