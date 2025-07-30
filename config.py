import torch
import os

class Config:
    # Paths
    DATA_ROOT = './data/vggface2'
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    
    # Model settings
    EMBEDDING_DIM = 512  # VGGFace embedding dimension
    IMAGE_SIZE = 224
    CROP_SIZE = 224
    
    # Training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loss weights (Eq. 2 from paper)
    LAMBDA_GRAD = 1.0
    LAMBDA_PERC = 1.0
    LAMBDA_L_PIX = 1.0
    LAMBDA_L_GRAD = 1.0
    LAMBDA_L_PERC = 1.0
    
    # Data augmentation
    USE_AUGMENTATION = True
    
    # Validation
    VAL_SPLIT = 0.1
    SAVE_FREQ = 5
    
    # Inference
    NUM_SAMPLES = 10
    
    # Pre-trained models
    VGGFACE_MODEL_PATH = './models/vggface_model.pth'
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)