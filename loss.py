import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

# Discriminator Loss
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, real_pred, fake_pred):
        real_loss = self.bce(real_pred, torch.ones_like(real_pred))
        fake_loss = self.bce(fake_pred, torch.zeros_like(fake_pred))
        return real_loss + fake_loss
    

# Generator Adversarial Loss
class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, fake_pred):
        return self.bce(fake_pred, torch.ones_like(fake_pred))
    

# VGG Feature Extractor
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=35):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:feature_layer].eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)
    

# VGG Loss
class VGGLoss(nn.Module):
    def __init__(self, feature_layer=35, use_rescaling=True):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.criterion = nn.MSELoss()
        self.use_rescaling = use_rescaling

        # Rescaling factor from paper
        self.rescaling_factor = 0.006 if use_rescaling else 1.0

    def vgg_preprocess(self, x):
        # Convert from [-1 1] to [0 1]
        x = (x + 1.0) / 2.0
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
        
        return (x - mean) / std

    def forward(self, sr, hr):
        sr_features = self.vgg(self.vgg_preprocess(sr))
        hr_features = self.vgg(self.vgg_preprocess(hr))

        loss = self.criterion(sr_features, hr_features)

        return loss * self.rescaling_factor


# Equation 3
# Combined Perceptual Loss (for SRGAN)
class PerceptualLoss(nn.Module):
    def __init__(self, content_weight=1.0, adversarial_weight=1e-3, 
                 use_vgg=False, vgg_layer=35):
        super().__init__()
        
        # Content loss (MSE or VGG)
        if use_vgg:
            self.content_loss = VGGLoss(feature_layer=vgg_layer)
        else:
            self.content_loss = nn.MSELoss()
        
        # Adversarial loss
        self.adversarial_loss = GeneratorAdversarialLoss()
        
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        self.use_vgg = use_vgg
    
    def forward(self, sr, hr, discriminator_pred):
        # Content loss (MSE or VGG)
        content = self.content_loss(sr, hr)
        
        # Adversarial loss
        adversarial = self.adversarial_loss(discriminator_pred)
        
        # Combined loss
        total = self.content_weight * content + self.adversarial_weight * adversarial
        
        return total, content, adversarial

