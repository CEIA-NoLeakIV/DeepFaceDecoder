import torch
import torch.nn as nn

class DFDDecoder(nn.Module):
    """Deep Face Decoder - Inverted VGG architecture (FIXED VERSION)"""
    
    def __init__(self, embedding_dim=512, image_size=224):
        super(DFDDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        
        # Calculate initial feature map size
        init_size = image_size // 32  # After 5 downsampling layers
        
        # Initial projection with better initialization
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512 * init_size * init_size),
            nn.BatchNorm1d(512 * init_size * init_size),
            nn.ReLU(True)
        )
        
        # Decoder layers (inverted VGG) with skip connections consideration
        self.decoder = nn.Sequential(
            # Block 1: 512 -> 512
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 7x7 -> 14x14
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Block 2: 512 -> 256  
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 14x14 -> 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Block 3: 256 -> 128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28x28 -> 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Block 4: 128 -> 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 56x56 -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Block 5: 64 -> 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 112x112 -> 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Final layer: 32 -> 3
            nn.Conv2d(32, 3, 3, 1, 1),
            # CRITICAL FIX: Use Sigmoid instead of Tanh for ImageNet normalization
            nn.Sigmoid()  # Output in [0, 1] - matches denormalized ImageNet range
        )
        
        # Reduced dropout for better learning
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.5
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, embedding):
        batch_size = embedding.size(0)
        
        # Project embedding to feature map
        x = self.fc(embedding)
        
        # Reshape to feature map
        init_size = self.image_size // 32
        x = x.view(batch_size, 512, init_size, init_size)
        
        # Apply dropout only during training
        if self.training:
            x = self.dropout(x)
        
        # Decode to image
        reconstructed = self.decoder(x)
        
        return reconstructed