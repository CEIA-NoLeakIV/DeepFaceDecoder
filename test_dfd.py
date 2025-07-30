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
from utils.metrics import calculate_metrics
from utils.losses import DFDLoss

def test_model(face_model, decoder, test_loader, device, config):
    """Test the DFD model"""
    face_model.eval()
    decoder.eval()
    
    criterion = DFDLoss(config).to(device)
    
    total_loss = 0.0
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    
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
            
            # Calculate loss
            loss_dict = criterion(reconstructed, images)
            total_loss += loss_dict['total_loss'].item()
            
            # Denormalize for metrics calculation
            images_denorm = denormalize_batch(images)
            reconstructed_denorm = denormalize_batch(reconstructed)
            
            # Calculate metrics for each image in batch
            for i in range(batch_size):
                metrics = calculate_metrics(
                    reconstructed_denorm[i:i+1], 
                    images_denorm[i:i+1]
                )
                psnr_scores.append(metrics['psnr'])
                ssim_scores.append(metrics['ssim'])
                mse_scores.append(metrics['mse'])
            
            num_samples += batch_size
            
            # Save some sample reconstructions
            if batch_idx < 5:  # Save first 5 batches
                save_sample_batch(
                    images_denorm, reconstructed_denorm, 
                    f"test_samples_batch_{batch_idx}.png"
                )
    
    # Calculate average metrics
    avg_loss = total_loss / len(test_loader)
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    
    print(f"\nTest Results ({num_samples} samples):")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")
    
    # Plot metrics distribution
    plot_metrics_distribution(psnr_scores, ssim_scores, mse_scores)
    
    return {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mse': avg_mse,
        'psnr_scores': psnr_scores,
        'ssim_scores': ssim_scores,
        'mse_scores': mse_scores
    }

def denormalize_batch(tensor):
    """Denormalize batch of images"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return torch.clamp(tensor * std + mean, 0, 1)

def save_sample_batch(original, reconstructed, filename, num_display=8):
    """Save sample reconstructions from a batch"""
    num_display = min(num_display, original.size(0))
    
    # Select samples to display
    orig_samples = original[:num_display]
    recon_samples = reconstructed[:num_display]
    
    # Create comparison grid
    comparison = torch.cat([orig_samples, recon_samples], dim=0)
    
    from torchvision.utils import save_image
    save_image(comparison, filename, nrow=num_display, normalize=False)

def plot_metrics_distribution(psnr_scores, ssim_scores, mse_scores):
    """Plot distribution of metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR distribution
    axes[0].hist(psnr_scores, bins=50, alpha=0.7, color='blue')
    axes[0].set_title(f'PSNR Distribution\nMean: {np.mean(psnr_scores):.2f} dB')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # SSIM distribution
    axes[1].hist(ssim_scores, bins=50, alpha=0.7, color='green')
    axes[1].set_title(f'SSIM Distribution\nMean: {np.mean(ssim_scores):.3f}')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # MSE distribution
    axes[2].hist(mse_scores, bins=50, alpha=0.7, color='red')
    axes[2].set_title(f'MSE Distribution\nMean: {np.mean(mse_scores):.6f}')
    axes[2].set_xlabel('MSE')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_embedding_space_analysis(face_model, decoder, test_loader, device):
    """Analyze the embedding space characteristics"""
    face_model.eval()
    decoder.eval()
    
    embeddings_list = []
    identities_list = []
    
    print("Analyzing embedding space...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            identities = batch['identity']
            
            # Extract embeddings
            embeddings = face_model.get_embedding(images)
            
            embeddings_list.append(embeddings.cpu())
            identities_list.extend(identities)
    
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
    
    # Plot embedding distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_embeddings.flatten(), bins=100, alpha=0.7)
    plt.title('Embedding Values Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('embedding_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return all_embeddings, identities_list

def main():
    config = Config()
    device = torch.device(config.DEVICE)
    
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
        return
    
    print("Loading models...")
    face_model = VGGFace(embedding_dim=config.EMBEDDING_DIM).to(device)
    decoder = DFDDecoder(embedding_dim=config.EMBEDDING_DIM, 
                        image_size=config.IMAGE_SIZE).to(device)
    
    # Load face model
    if os.path.exists(config.VGGFACE_MODEL_PATH):
        face_model.load_state_dict(torch.load(config.VGGFACE_MODEL_PATH, map_location=device))
    
    # Load decoder
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Run tests
    test_results = test_model(face_model, decoder, test_loader, device, config)
    
    # Embedding space analysis
    embeddings, identities = test_embedding_space_analysis(face_model, decoder, test_loader, device)
    
    # Save results
    results_file = 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write("DFD Test Results\n")
        f.write("================\n\n")
        f.write(f"Average Loss: {test_results['loss']:.4f}\n")
        f.write(f"Average PSNR: {test_results['psnr']:.4f} dB\n")
        f.write(f"Average SSIM: {test_results['ssim']:.4f}\n")
        f.write(f"Average MSE: {test_results['mse']:.6f}\n")
        f.write(f"\nNumber of test samples: {len(test_results['psnr_scores'])}\n")
    
    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()