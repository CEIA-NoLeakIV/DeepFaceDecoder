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
    """Calculate SSIM between two images - FIXED VERSION"""
    # Convert to numpy and move to CPU
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    # Calculate SSIM for each image in batch
    ssim_values = []
    for i in range(img1_np.shape[0]):
        # Transpose from CHW to HWC
        im1 = np.transpose(img1_np[i], (1, 2, 0))
        im2 = np.transpose(img2_np[i], (1, 2, 0))
        
        # Ensure images are in valid range [0, 1]
        im1 = np.clip(im1, 0, 1)
        im2 = np.clip(im2, 0, 1)
        
        # Check image size
        h, w = im1.shape[:2]
        
        try:
            if h < 7 or w < 7:
                # Use smaller window size for small images
                win_size = min(h, w)
                if win_size % 2 == 0:
                    win_size -= 1  # Ensure odd window size
                if win_size < 3:
                    win_size = 3
                    
                ssim_val = ssim(im1, im2, 
                               channel_axis=2,  # Updated parameter name
                               data_range=1.0,
                               win_size=win_size)
            else:
                # Use default window size for normal images
                ssim_val = ssim(im1, im2, 
                               channel_axis=2,  # Updated parameter name  
                               data_range=1.0)
                               
        except Exception as e:
            print(f"SSIM calculation failed: {e}")
            # Fallback to simple correlation if SSIM fails
            ssim_val = np.corrcoef(im1.flatten(), im2.flatten())[0, 1]
            if np.isnan(ssim_val):
                ssim_val = 0.0
        
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def calculate_metrics(pred, target):
    """Calculate reconstruction metrics - ENHANCED VERSION"""
    with torch.no_grad():
        # Ensure tensors are in valid range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Check for valid input
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # PSNR using PyTorch
        try:
            psnr_val = calculate_psnr(pred, target)
        except Exception as e:
            print(f"PSNR calculation failed: {e}")
            psnr_val = 0.0
        
        # SSIM using scikit-image
        try:
            ssim_val = calculate_ssim(pred, target)
        except Exception as e:
            print(f"SSIM calculation failed: {e}")
            ssim_val = 0.0
        
        # MSE
        try:
            mse_val = torch.mean((pred - target) ** 2)
        except Exception as e:
            print(f"MSE calculation failed: {e}")
            mse_val = float('inf')
        
        return {
            'psnr': psnr_val.item() if torch.is_tensor(psnr_val) else psnr_val,
            'ssim': ssim_val,
            'mse': mse_val.item() if torch.is_tensor(mse_val) else mse_val
        }

def calculate_batch_metrics(pred_batch, target_batch):
    """Calculate metrics for entire batch efficiently"""
    with torch.no_grad():
        batch_size = pred_batch.size(0)
        
        psnr_scores = []
        ssim_scores = []
        mse_scores = []
        
        for i in range(batch_size):
            metrics = calculate_metrics(pred_batch[i:i+1], target_batch[i:i+1])
            psnr_scores.append(metrics['psnr'])
            ssim_scores.append(metrics['ssim'])
            mse_scores.append(metrics['mse'])
        
        return {
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'psnr_scores': psnr_scores,
            'ssim_scores': ssim_scores,
            'mse_scores': mse_scores
        }