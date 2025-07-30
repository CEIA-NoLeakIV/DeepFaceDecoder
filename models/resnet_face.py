import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFace(nn.Module):
    """ResNet-50 based face recognition model"""
    
    def __init__(self, num_classes=8631, embedding_dim=2048):
        super(ResNetFace, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_embedding=False):
        # Feature extraction
        embedding = self.backbone(x)
        
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