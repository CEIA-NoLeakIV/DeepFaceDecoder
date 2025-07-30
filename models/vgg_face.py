import torch
import torch.nn as nn
import torchvision.models as models

class VGGFace(nn.Module):
    """VGG-16 based face recognition model"""
    
    def __init__(self, num_classes=8631, embedding_dim=512):
        super(VGGFace, self).__init__()
        
        # Load pre-trained VGG-16
        self.backbone = models.vgg16(pretrained=True)
        
        # Modify classifier for face recognition
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, embedding_dim),
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_embedding=False):
        # Feature extraction
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Get embedding
        embedding = self.backbone.classifier(features)
        
        if return_embedding:
            return embedding
        
        # Classification
        logits = self.classifier(embedding)
        return logits, embedding
    
    def get_embedding(self, x):
        """Extract normalized embedding"""
        with torch.no_grad():
            embedding = self.forward(x, return_embedding=True)
            # L2 normalization
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding