import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import numpy as np

from config import Config
from models import VGGFace, DFDDecoder
from utils.data_loader import get_data_loaders
from utils.losses import DFDLoss
from utils.metrics import calculate_metrics

def denormalize_image(tensor, is_reconstructed=False):
    """FIXED: Proper denormalization for visualization"""
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

def train_epoch(face_model, decoder, train_loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch - ENHANCED"""
    decoder.train()
    face_model.eval()  # Keep face model frozen
    
    running_loss = 0.0
    running_losses = {
        'pixel': 0.0, 'grad': 0.0, 'perc': 0.0,
        'l_pixel': 0.0, 'l_grad': 0.0, 'l_perc': 0.0
    }
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        
        # Extract embeddings (frozen face model)
        with torch.no_grad():
            embeddings = face_model.get_embedding(images)
        
        # Add small noise to embeddings to prevent overfitting during training
        if decoder.training:
            noise = torch.randn_like(embeddings) * 0.01
            embeddings = embeddings + noise
        
        # Reconstruct images
        reconstructed = decoder(embeddings)
        
        # CRITICAL: Denormalize target images to match reconstructed range [0,1]
        target_denorm = denormalize_image(images, is_reconstructed=False)
        
        # Calculate loss using denormalized targets
        loss_dict = criterion(reconstructed, target_denorm)
        total_loss = loss_dict['total_loss']
        
        # Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: NaN/Inf loss detected at batch {batch_idx}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.GRAD_CLIP_NORM)
        
        optimizer.step()
        
        # Update metrics
        running_loss += total_loss.item()
        for key in running_losses:
            if key == 'pixel':
                running_losses[key] += loss_dict['pixel_loss'].item()
            elif key == 'grad':
                running_losses[key] += loss_dict['gradient_loss'].item()
            elif key == 'perc':
                running_losses[key] += loss_dict['perceptual_loss'].item()
            elif key == 'l_pixel':
                running_losses[key] += loss_dict['local_pixel_loss'].item()
            elif key == 'l_grad':
                running_losses[key] += loss_dict['local_gradient_loss'].item()
            elif key == 'l_perc':
                running_losses[key] += loss_dict['local_perceptual_loss'].item()
        
        # Update progress bar with detailed losses
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Avg': f'{avg_loss:.4f}',
            'Pixel': f'{loss_dict["pixel_loss"].item():.4f}',
            'Grad': f'{loss_dict["gradient_loss"].item():.4f}'
        })
    
    # Calculate average losses
    avg_losses = {key: val / num_batches for key, val in running_losses.items()}
    return running_loss / num_batches, avg_losses

def validate_epoch(face_model, decoder, val_loader, criterion, device):
    """Validate for one epoch - ENHANCED"""
    decoder.eval()
    face_model.eval()
    
    val_loss = 0.0
    val_losses = {
        'pixel': 0.0, 'grad': 0.0, 'perc': 0.0,
        'l_pixel': 0.0, 'l_grad': 0.0, 'l_perc': 0.0
    }
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            
            # Extract embeddings
            embeddings = face_model.get_embedding(images)
            
            # Reconstruct images
            reconstructed = decoder(embeddings)
            
            # Denormalize target images
            target_denorm = denormalize_image(images, is_reconstructed=False)
            
            # Calculate loss
            loss_dict = criterion(reconstructed, target_denorm)
            val_loss += loss_dict['total_loss'].item()
            
            # Track individual losses
            val_losses['pixel'] += loss_dict['pixel_loss'].item()
            val_losses['grad'] += loss_dict['gradient_loss'].item()
            val_losses['perc'] += loss_dict['perceptual_loss'].item()
            val_losses['l_pixel'] += loss_dict['local_pixel_loss'].item()
            val_losses['l_grad'] += loss_dict['local_gradient_loss'].item()
            val_losses['l_perc'] += loss_dict['local_perceptual_loss'].item()
    
    avg_val_losses = {key: val / num_batches for key, val in val_losses.items()}
    return val_loss / num_batches, avg_val_losses

def save_sample_reconstructions(face_model, decoder, val_loader, device, save_path, num_samples=8):
    """FIXED: Save sample reconstructions for visualization"""
    decoder.eval()
    face_model.eval()
    
    # Get a batch of validation images
    batch = next(iter(val_loader))
    images = batch['image'][:num_samples].to(device)
    
    with torch.no_grad():
        embeddings = face_model.get_embedding(images)
        reconstructed = decoder(embeddings)
    
    # Proper denormalization
    images_denorm = denormalize_image(images, is_reconstructed=False)
    reconstructed_denorm = denormalize_image(reconstructed, is_reconstructed=True)
    
    # Create comparison grid
    comparison = torch.cat([images_denorm, reconstructed_denorm], dim=0)
    
    # Save images
    from torchvision.utils import save_image
    save_image(comparison, save_path, nrow=num_samples, normalize=False)

def main():
    # Configuration
    config = Config()
    config.create_dirs()
    
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(config)
    
    # Models
    print("Initializing models...")
    face_model = VGGFace(embedding_dim=config.EMBEDDING_DIM).to(device)
    decoder = DFDDecoder(embedding_dim=config.EMBEDDING_DIM, 
                        image_size=config.IMAGE_SIZE).to(device)
    
    # Load pre-trained face model if available
    if os.path.exists(config.VGGFACE_MODEL_PATH):
        print("Loading pre-trained face model...")
        face_model.load_state_dict(torch.load(config.VGGFACE_MODEL_PATH))
    
    # Freeze face model
    for param in face_model.parameters():
        param.requires_grad = False
    
    # Loss and optimizer - ENHANCED
    criterion = DFDLoss(config).to(device)
    optimizer = optim.Adam(decoder.parameters(), 
                          lr=config.LEARNING_RATE,
                          weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=config.SCHEDULER_STEP_SIZE, 
                                         gamma=config.SCHEDULER_GAMMA)
    
    # Tensorboard
    writer = SummaryWriter(config.LOG_DIR)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_losses = train_epoch(face_model, decoder, train_loader, 
                                              criterion, optimizer, device, epoch, config)
        
        # Validate
        val_loss, val_losses = validate_epoch(face_model, decoder, val_loader, 
                                            criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics - ENHANCED
        writer.add_scalar('Loss/Train_Total', train_loss, epoch)
        writer.add_scalar('Loss/Val_Total', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log individual losses
        for key, val in train_losses.items():
            writer.add_scalar(f'Train_Loss/{key}', val, epoch)
        for key, val in val_losses.items():
            writer.add_scalar(f'Val_Loss/{key}', val, epoch)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  Time: {epoch_time:.2f}s')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        print('-' * 50)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_path = os.path.join(config.CHECKPOINT_DIR, 'dfd_best.pth')
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Save checkpoint
        if epoch % config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'dfd_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            
            # Save sample reconstructions
            sample_path = os.path.join(config.LOG_DIR, f'samples_epoch_{epoch}.png')
            save_sample_reconstructions(face_model, decoder, val_loader, 
                                      device, sample_path)
    
    print("Training completed!")
    writer.close()

if __name__ == '__main__':
    main()