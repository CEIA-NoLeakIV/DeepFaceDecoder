import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from models import VGGFace, DFDDecoder
from utils.data_loader import VGGFace2Dataset, get_data_transforms
from utils.metrics import calculate_metrics, calculate_batch_metrics
from utils.losses import DFDLoss

def denormalize_image(tensor, is_reconstructed=False):
    """FIXED: Proper denormalization matching training script"""
    if is_reconstructed:
        # Reconstructed images are in [0, 1] from Sigmoid
        return torch.clamp(tensor, 0, 1)
    else:
        # Original images need ImageNet denormalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if tensor.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        
        denorm = tensor * std + mean
        return torch.clamp(denorm, 0, 1)

def test_model(face_model, decoder, test_loader, device, config):
    """Test the DFD model"""
    face_model.eval()
    decoder.eval()
    
    criterion = DFDLoss(config).to(device)
    
    total_loss = 0.0
    all_psnr_scores = []
    all_ssim_scores = []
    all_mse_scores = []
    
    num_samples = 0
    
    print("Testing model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images = batch['image'].to(device)
            batch_size = images.size(0)
            
            # Extract embeddings
            embeddings = face_model.get_embedding(images)
            
            # Reconstruct images
            reconstructed = decoder(embeddings)
            
            # Proper denormalization for loss calculation
            target_denorm = denormalize_image(images, is_reconstructed=False)
            
            # Calculate loss using denormalized targets
            loss_dict = criterion(reconstructed, target_denorm)
            total_loss += loss_dict['total_loss'].item()
            
            # Calculate metrics using properly denormalized images
            batch_metrics = calculate_batch_metrics(reconstructed, target_denorm)
            
            all_psnr_scores.extend(batch_metrics['psnr_scores'])
            all_ssim_scores.extend(batch_metrics['ssim_scores'])
            all_mse_scores.extend(batch_metrics['mse_scores'])
            
            num_samples += batch_size
            
            # Save some sample reconstructions
            if batch_idx < 5:  # Save first 5 batches
                save_sample_batch(
                    target_denorm, reconstructed, 
                    f"test_samples_batch_{batch_idx}.png"
                )
            
            # Print progress for first few batches
            if batch_idx < 3:
                print(f"Batch {batch_idx}: PSNR={batch_metrics['psnr_mean']:.2f}, "
                      f"SSIM={batch_metrics['ssim_mean']:.3f}, "
                      f"MSE={batch_metrics['mse_mean']:.6f}")
    
    # Calculate average metrics
    avg_loss = total_loss / len(test_loader)
    avg_psnr = np.mean(all_psnr_scores)
    avg_ssim = np.mean(all_ssim_scores)
    avg_mse = np.mean(all_mse_scores)
    
    print(f"\nTest Results ({num_samples} samples):")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"PSNR Std: {np.std(all_psnr_scores):.4f}")
    print(f"SSIM Std: {np.std(all_ssim_scores):.4f}")
    
    # Plot metrics distribution
    plot_metrics_distribution(all_psnr_scores, all_ssim_scores, all_mse_scores)
    
    return {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mse': avg_mse,
        'psnr_scores': all_psnr_scores,
        'ssim_scores': all_ssim_scores,
        'mse_scores': all_mse_scores
    }

def save_sample_batch(original, reconstructed, filename, num_display=8):
    """Save sample reconstructions from a batch"""
    num_display = min(num_display, original.size(0))
    
    # Select samples to display
    orig_samples = original[:num_display]
    recon_samples = reconstructed[:num_display]
    
    # Create comparison grid (original on top, reconstructed on bottom)
    comparison = torch.cat([orig_samples, recon_samples], dim=0)
    
    from torchvision.utils import save_image
    save_image(comparison, filename, nrow=num_display, normalize=False)
    print(f"Saved comparison to {filename}")

def plot_metrics_distribution(psnr_scores, ssim_scores, mse_scores):
    """Plot distribution of metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR distribution
    axes[0].hist(psnr_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title(f'PSNR Distribution\nMean: {np.mean(psnr_scores):.2f} +/- {np.std(psnr_scores):.2f} dB')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(np.mean(psnr_scores), color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    # SSIM distribution
    axes[1].hist(ssim_scores, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title(f'SSIM Distribution\nMean: {np.mean(ssim_scores):.3f} +/- {np.std(ssim_scores):.3f}')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(np.mean(ssim_scores), color='red', linestyle='--', label='Mean')
    axes[1].legend()
    
    # MSE distribution
    axes[2].hist(mse_scores, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[2].set_title(f'MSE Distribution\nMean: {np.mean(mse_scores):.6f} +/- {np.std(mse_scores):.6f}')
    axes[2].set_xlabel('MSE')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(np.mean(mse_scores), color='red', linestyle='--', label='Mean')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Metrics distribution saved to metrics_distribution.png")

def test_embedding_space_analysis(face_model, decoder, test_loader, device):
    """Analyze the embedding space characteristics"""
    face_model.eval()
    decoder.eval()
    
    embeddings_list = []
    identities_list = []
    
    print("Analyzing embedding space...")
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            identities = batch['identity']
            
            # Extract embeddings
            embeddings = face_model.get_embedding(images)
            
            embeddings_list.append(embeddings.cpu())
            identities_list.extend(identities)
            
            sample_count += images.size(0)
            
            # Limit analysis to reasonable number for memory
            if sample_count >= 1000:
                break
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
    
    print(f"Collected {all_embeddings.shape[0]} embeddings")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    # Analyze embedding statistics
    print(f"Embedding statistics:")
    print(f"  Mean: {np.mean(all_embeddings):.6f}")
    print(f"  Std: {np.std(all_embeddings):.6f}")
    print(f"  Min: {np.min(all_embeddings):.6f}")
    print(f"  Max: {np.max(all_embeddings):.6f}")
    print(f"  L2 norm mean: {np.mean(np.linalg.norm(all_embeddings, axis=1)):.6f}")
    
    # Check for dead neurons (always zero)
    dead_neurons = np.sum(np.all(all_embeddings == 0, axis=0))
    print(f"  Dead neurons: {dead_neurons}/{all_embeddings.shape[1]}")
    
    # Plot embedding distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(all_embeddings.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Embedding Values Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    norms = np.linalg.norm(all_embeddings, axis=1)
    plt.hist(norms, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Embedding L2 Norms Distribution')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    means_per_dim = np.mean(all_embeddings, axis=0)
    plt.plot(means_per_dim)
    plt.title('Mean Value per Embedding Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Mean Value')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    stds_per_dim = np.std(all_embeddings, axis=0)
    plt.plot(stds_per_dim)
    plt.title('Std Dev per Embedding Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Std Dev')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Embedding analysis saved to embedding_analysis.png")
    
    return all_embeddings, identities_list

def main():
    config = Config()
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    _, val_transform = get_data_transforms(config.IMAGE_SIZE, augment=False)
    test_dataset = VGGFace2Dataset(
        config.DATA_ROOT, 
        split='test', 
        transform=val_transform,
        image_size=config.IMAGE_SIZE
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load models
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'dfd_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Available files in {config.CHECKPOINT_DIR}:")
        if os.path.exists(config.CHECKPOINT_DIR):
            for f in os.listdir(config.CHECKPOINT_DIR):
                print(f"  {f}")
        return
    
    print("Loading models...")
    face_model = VGGFace(embedding_dim=config.EMBEDDING_DIM).to(device)
    decoder = DFDDecoder(embedding_dim=config.EMBEDDING_DIM, 
                        image_size=config.IMAGE_SIZE).to(device)
    
    # Load face model
    if os.path.exists(config.VGGFACE_MODEL_PATH):
        print("Loading pre-trained face model...")
        face_model.load_state_dict(torch.load(config.VGGFACE_MODEL_PATH, map_location=device))
    else:
        print("WARNING: Pre-trained face model not found!")
    
    # Load decoder
    print("Loading decoder checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Run tests
    print("\n" + "="*50)
    print("STARTING COMPREHENSIVE TESTING")
    print("="*50)
    
    test_results = test_model(face_model, decoder, test_loader, device, config)
    
    # Embedding space analysis
    print("\n" + "="*50)
    print("EMBEDDING SPACE ANALYSIS")
    print("="*50)
    embeddings, identities = test_embedding_space_analysis(face_model, decoder, test_loader, device)
    
    # Save results
    results_file = 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write("DFD Test Results\n")
        f.write("================\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n\n")
        f.write(f"Average Loss: {test_results['loss']:.4f}\n")
        f.write(f"Average PSNR: {test_results['psnr']:.4f} +/- {np.std(test_results['psnr_scores']):.4f} dB\n")
        f.write(f"Average SSIM: {test_results['ssim']:.4f} +/- {np.std(test_results['ssim_scores']):.4f}\n")
        f.write(f"Average MSE: {test_results['mse']:.6f} +/- {np.std(test_results['mse_scores']):.6f}\n")
        f.write(f"\nNumber of test samples: {len(test_results['psnr_scores'])}\n")
        f.write(f"\nPercentiles:\n")
        f.write(f"PSNR - 25th: {np.percentile(test_results['psnr_scores'], 25):.2f}, "
                f"50th: {np.percentile(test_results['psnr_scores'], 50):.2f}, "
                f"75th: {np.percentile(test_results['psnr_scores'], 75):.2f}\n")
        f.write(f"SSIM - 25th: {np.percentile(test_results['ssim_scores'], 25):.3f}, "
                f"50th: {np.percentile(test_results['ssim_scores'], 50):.3f}, "
                f"75th: {np.percentile(test_results['ssim_scores'], 75):.3f}\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Quality assessment
    avg_psnr = test_results['psnr']
    avg_ssim = test_results['ssim']
    
    print("\n" + "="*50)
    print("QUALITY ASSESSMENT")
    print("="*50)
    
    if avg_psnr > 20:
        print("✓ PSNR > 20 dB: Good reconstruction quality")
    elif avg_psnr > 15:
        print("⚠ PSNR 15-20 dB: Moderate reconstruction quality")
    else:
        print("✗ PSNR < 15 dB: Poor reconstruction quality")
    
    if avg_ssim > 0.8:
        print("✓ SSIM > 0.8: Excellent structural similarity")
    elif avg_ssim > 0.6:
        print("⚠ SSIM 0.6-0.8: Good structural similarity")
    else:
        print("✗ SSIM < 0.6: Poor structural similarity")

if __name__ == '__main__':
    main()