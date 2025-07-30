import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Convert to numpy and move to CPU
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    # Calculate SSIM for each image in batch
    ssim_values = []
    for i in range(img1_np.shape[0]):
        # Transpose from CHW to HWC
        im1 = np.transpose(img1_np[i], (1, 2, 0))
        im2 = np.transpose(img2_np[i], (1, 2, 0))
        
        # Calculate SSIM
        ssim_val = ssim(im1, im2, multichannel=True, data_range=1.0)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def calculate_metrics(pred, target):
    """Calculate reconstruction metrics"""
    with torch.no_grad():
        # Denormalize if needed
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # PSNR
        psnr_val = calculate_psnr(pred, target)
        
        # SSIM
        ssim_val = calculate_ssim(pred, target)
        
        # MSE
        mse_val = torch.mean((pred - target) ** 2)
        
        return {
            'psnr': psnr_val.item() if torch.is_tensor(psnr_val) else psnr_val,
            'ssim': ssim_val,
            'mse': mse_val.item()
        }