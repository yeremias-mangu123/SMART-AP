"""
UNet Model for Floor Plan Wall Segmentation
Uses segmentation-models-pytorch with ResNet18 encoder pre-trained on ImageNet.
"""

import segmentation_models_pytorch as smp
import torch


def create_model(encoder_name="resnet18", encoder_weights="imagenet", num_classes=1):
    """Create a UNet segmentation model for wall detection.
    
    Args:
        encoder_name: Backbone encoder (resnet18 is fast and accurate enough)
        encoder_weights: Pre-trained weights to use
        num_classes: Number of output classes (1 for binary wall/non-wall)
    
    Returns:
        UNet model ready for training or inference
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,  # We'll use BCEWithLogitsLoss which includes sigmoid
    )
    return model


class DiceBCELoss(torch.nn.Module):
    """Combined Dice + BCE loss for better segmentation training."""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # BCE component
        bce_loss = self.bce(pred, target)
        
        # Dice component
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
