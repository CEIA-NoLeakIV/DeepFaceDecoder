import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from config import Config
from models import VGGFace, DFDDecoder

def load_models(checkpoint_path, config):
    """Load trained models"""
    device = torch.device(config.DEVICE)
    
    # Initialize models
    face_model = VGGFace(embedding_dim=config.EMBEDDING_DIM).to(device)
    decoder = DFDDecoder(embedding_dim=config.EMBEDDING_DIM, 
                        image_size=config.IMAGE_SIZE).to(device)
    
    # Load face model
    if os.path.exists(config.VGGFACE_MODEL_PATH):
        face_model.load_state_dict(torch.load(config.VGGFACE_MODEL_PATH, map_location=device))
    
    # Load decoder
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    face_model.eval()
    decoder.eval()
    
    return face_model, decoder

def preprocess_image(image_path, image_size=224):
    """Preprocess input image"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def denormalize_image(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean

def reconstruct_face(face_model, decoder, image_path, device):
    """Reconstruct face from image"""
    # Preprocess image
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        # Extract embedding
        embedding = face_model.get_embedding(image)
        
        # Reconstruct image
        reconstructed = decoder(embedding)
        
        # Denormalize for visualization
        original = denormalize_image(image).clamp(0, 1)
        reconstructed = denormalize_image(reconstructed).clamp(0, 1)
    
    return original.squeeze(0), reconstructed.squeeze(0), embedding.squeeze(0)

def visualize_reconstruction(original, reconstructed, save_path=None):
    """Visualize original and reconstructed images"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convert to numpy and transpose for matplotlib
    orig_np = original.cpu().numpy().transpose(1, 2, 0)
    recon_np = reconstructed.cpu().numpy().transpose(1, 2, 0)
    
    axes[0].imshow(orig_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def analyze_embedding(embedding, save_path=None):
    """Analyze embedding characteristics"""
    embedding_np = embedding.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Embedding histogram
    axes[0, 0].hist(embedding_np, bins=50, alpha=0.7)
    axes[0, 0].set_title('Embedding Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Embedding values
    axes[0, 1].plot(embedding_np)
    axes[0, 1].set_title('Embedding Values')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Value')
    
    # Embedding heatmap
    # Reshape to square for visualization
    size = int(np.sqrt(len(embedding_np)))
    if size * size == len(embedding_np):
        heatmap = embedding_np.reshape(size, size)
    else:
        # Pad or truncate to make square
        target_size = size ** 2
        if len(embedding_np) > target_size:
            heatmap = embedding_np[:target_size].reshape(size, size)
        else:
            padded = np.pad(embedding_np, (0, target_size - len(embedding_np)))
            heatmap = padded.reshape(size, size)
    
    im = axes[1, 0].imshow(heatmap, cmap='viridis')
    axes[1, 0].set_title('Embedding Heatmap')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Statistics
    stats_text = f"""
    Statistics:
    Min: {embedding_np.min():.4f}
    Max: {embedding_np.max():.4f}
    Mean: {embedding_np.mean():.4f}
    Std: {embedding_np.std():.4f}
    L2 Norm: {np.linalg.norm(embedding_np):.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def batch_inference(face_model, decoder, image_folder, output_folder, device):
    """Run inference on a folder of images"""
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Processing {len(image_files)} images...")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        
        try:
            # Reconstruct
            original, reconstructed, embedding = reconstruct_face(
                face_model, decoder, image_path, device
            )
            
            # Save reconstruction
            output_path = os.path.join(output_folder, f'reconstruction_{i:04d}.png')
            visualize_reconstruction(original, reconstructed, output_path)
            
            # Save embedding analysis
            embedding_path = os.path.join(output_folder, f'embedding_{i:04d}.png')
            analyze_embedding(embedding, embedding_path)
            
            print(f"Processed {i+1}/{len(image_files)}: {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def main():
    config = Config()
    device = torch.device(config.DEVICE)
    
    # Load models
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'dfd_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print("Loading models...")
    face_model, decoder = load_models(checkpoint_path, config)
    
    # Example: Single image inference
    image_path = input("Enter path to image (or press Enter to skip): ")
    if image_path and os.path.exists(image_path):
        print("Reconstructing face...")
        original, reconstructed, embedding = reconstruct_face(
            face_model, decoder, image_path, device
        )
        
        visualize_reconstruction(original, reconstructed)
        analyze_embedding(embedding)
    
    # Example: Batch inference
    folder_path = input("Enter path to image folder (or press Enter to skip): ")
    if folder_path and os.path.exists(folder_path):
        output_folder = input("Enter output folder path: ")
        batch_inference(face_model, decoder, folder_path, output_folder, device)

if __name__ == '__main__':
    main()