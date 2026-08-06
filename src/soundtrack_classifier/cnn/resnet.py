import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    """
    CNN Classifier: Adapts a ResNet-18 to accept 1-channel CQT spectrograms 
    and output a single probability distribution over the target classes.
    """
    def __init__(self, num_classes: int = 12, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes
        
        # Load ResNet-18
        # Note: We use pretrained=False here by default because standard ImageNet weights 
        # expect 3-channel RGB. If using pretrained=True, you'd sum the weights of the 
        # first conv layer across the channel dimension to make it 1-channel compatible.
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 1-channel input (our spectrograms)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        # Modify the final fully connected layer to output `num_classes`
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, 1, freq_bins, time_steps]
        Returns:
            Logits of shape [batch_size, num_classes]
        """
        return self.backbone(x)
