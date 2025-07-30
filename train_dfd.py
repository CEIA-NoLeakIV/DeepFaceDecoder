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

def denormalize_image(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean

def train_epoch(face_model, decoder, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    decoder.train()
    face_model.eval()  # Keep face model frozen
    
    running_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        
        # Extract embeddings (frozen face model)
        with torch.no_grad():
            embeddings = face_model.get_embedding(images)
        
        # Reconstruct images
        reconstructed = decoder(embeddings)
        
        # Calculate loss
        loss_dict = criterion(reconstructed, images)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += total_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
        })
    
    return running_loss / num_batches

def validate_epoch(face_model, decoder, val_loader, criterion, device):
    """Validate for one epoch"""
    decoder.eval()
    face_model.eval()
    
    val_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            
            # Extract embeddings
            embeddings = face_model.get_embedding(images)
            
            # Reconstruct images
            reconstructed = decoder(embeddings)
            
            # Calculate loss
            loss_dict = criterion(reconstructed, images)
            val_loss += loss_dict['total_loss'].item()
    
    return val_loss / num_batches

def save_sample_reconstructions(face_model, decoder, val_loader, device, save_path, num_samples=8):
    """Save sample reconstructions for visualization"""
    decoder.eval()
    face_model.eval()
    
    # Get a batch of validation images
    batch = next(iter(val_loader))
    images = batch['image'][:num_samples].to(device)
    
    with torch.no_grad():
        embeddings = face_model.get_embedding(images)
        reconstructed = decoder(embeddings)
    
    # Denormalize for visualization
    images = denormalize_image(images).clamp(0, 1)
    reconstructed = denormalize_image(reconstructed).clamp(0, 1)
    
    # Create comparison grid
    comparison = torch.cat([images, reconstructed], dim=0)
    
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
    
    # Loss and optimizer
    criterion = DFDLoss(config).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Tensorboard
    writer = SummaryWriter(config.LOG_DIR)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(face_model, decoder, train_loader, 
                               criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate_epoch(face_model, decoder, val_loader, 
                                criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Time: {epoch_time:.2f}s')
        print('-' * 50)
        
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, 'dfd_best.pth')
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_path)
    
    print("Training completed!")
    writer.close()

if __name__ == '__main__':
    main()