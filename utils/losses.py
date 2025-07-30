import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DFDLoss(nn.Module):
    """Multi-term loss function for DFD (Equation 2 from paper)"""
    
    def __init__(self, config, perceptual_model='vgg16'):
        super(DFDLoss, self).__init__()
        
        self.config = config
        
        # Perceptual network
        if perceptual_model == 'vgg16':
            vgg = models.vgg16(pretrained=True).features
            self.perceptual_net = nn.Sequential(*list(vgg.children())[:23])  # conv4_3
        else:
            resnet = models.resnet50(pretrained=True)
            self.perceptual_net = nn.Sequential(*list(resnet.children())[:-2])
            
        # Freeze perceptual network
        for param in self.perceptual_net.parameters():
            param.requires_grad = False
        
        self.perceptual_net.eval()
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def pixel_loss(self, pred, target):
        """L2 pixel loss (Equation 3)"""
        return self.mse_loss(pred, target)
    
    def gradient_loss(self, pred, target):
        """Gradient loss (Equation 4)"""
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # Expand for RGB channels
        sobel_x = sobel_x.repeat(3, 1, 1, 1)
        sobel_y = sobel_y.repeat(3, 1, 1, 1)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, sobel_x, groups=3, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, groups=3, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, groups=3, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, groups=3, padding=1)
        
        # Gradient magnitude
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return self.mse_loss(pred_grad, target_grad)
    
    def perceptual_loss(self, pred, target):
        """Perceptual loss (Equation 5)"""
        pred_features = self.perceptual_net(pred)
        target_features = self.perceptual_net(target)
        return self.mse_loss(pred_features, target_features)
    
    def get_local_region(self, image, crop_ratio=0.6):
        """Extract local facial region"""
        b, c, h, w = image.shape
        
        # Center crop
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    def forward(self, pred, target):
        """Complete loss function (Equation 2)"""
        # Global losses
        loss_pix = self.pixel_loss(pred, target)
        loss_grad = self.gradient_loss(pred, target)
        loss_perc = self.perceptual_loss(pred, target)
        
        # Local losses
        pred_local = self.get_local_region(pred)
        target_local = self.get_local_region(target)
        
        loss_l_pix = self.pixel_loss(pred_local, target_local)
        loss_l_grad = self.gradient_loss(pred_local, target_local)
        loss_l_perc = self.perceptual_loss(pred_local, target_local)
        
        # Combined loss
        total_loss = (loss_pix + 
                     self.config.LAMBDA_GRAD * loss_grad + 
                     self.config.LAMBDA_PERC * loss_perc +
                     self.config.LAMBDA_L_PIX * loss_l_pix +
                     self.config.LAMBDA_L_GRAD * loss_l_grad +
                     self.config.LAMBDA_L_PERC * loss_l_perc)
        
        return {
            'total_loss': total_loss,
            'pixel_loss': loss_pix,
            'gradient_loss': loss_grad,
            'perceptual_loss': loss_perc,
            'local_pixel_loss': loss_l_pix,
            'local_gradient_loss': loss_l_grad,
            'local_perceptual_loss': loss_l_perc
        }