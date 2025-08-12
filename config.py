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
    
    # Training settings - FIXED
    BATCH_SIZE = 16  # Reduced for stability
    LEARNING_RATE = 5e-5  # CRITICAL FIX: Much lower learning rate
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loss weights (Eq. 2 from paper) - REBALANCED
    LAMBDA_GRAD = 0.1      # Reduced from 1.0
    LAMBDA_PERC = 0.1      # Reduced from 1.0  
    LAMBDA_L_PIX = 0.5     # Reduced from 1.0
    LAMBDA_L_GRAD = 0.1    # Reduced from 1.0
    LAMBDA_L_PERC = 0.1    # Reduced from 1.0
    
    # Data augmentation
    USE_AUGMENTATION = True
    
    # Validation
    VAL_SPLIT = 0.1
    SAVE_FREQ = 5
    
    # Inference
    NUM_SAMPLES = 10
    
    # Pre-trained models
    VGGFACE_MODEL_PATH = './models/vggface_model.pth'
    
    # Optimizer settings - NEW
    WEIGHT_DECAY = 1e-4
    SCHEDULER_STEP_SIZE = 20  # Reduced from 30
    SCHEDULER_GAMMA = 0.5     # More conservative decay
    
    # Gradient clipping - NEW
    GRAD_CLIP_NORM = 1.0
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)