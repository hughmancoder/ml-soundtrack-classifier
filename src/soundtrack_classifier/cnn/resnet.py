import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int = 12, in_channels: int = 1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18Classifier(nn.Module):
    """
    Legacy CNN Classifier: Adapts a lightweight ResNet-18 to accept 1-channel CQT spectrograms.
    Preserved for backward compatibility with existing checkpoints.
    """
    def __init__(self, num_classes: int = 12, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNetBackbone(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet50Classifier(nn.Module):
    """
    Transfer Learning CNN Classifier: Adapts a pre-trained ResNet-50 backbone to accept
    CQT spectrograms and classify multi-class acoustic instrument events.
    """
    def __init__(self, num_classes: int = 12, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
            
        self.backbone = resnet50(weights=weights)
        
        # Replace the final classification head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, 1, freq_bins, time_steps] or [batch_size, 3, freq_bins, time_steps]
        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # ResNet-50 expects 3-channel input (RGB). 
        # We duplicate the 1-channel CQT spectrogram across 3 channels
        # to effectively utilize pre-trained ImageNet feature weights.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        return self.backbone(x)

