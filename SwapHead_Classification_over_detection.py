import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16
from torchvision.models import mobilenet_v3_large

# Load SSD and MobileNetV3 models
ssd_model = ssd300_vgg16(pretrained=True)
mobilenet = mobilenet_v3_large(pretrained=True)

# Step 1: Extract SSD's backbone (feature extractor)
# The SSD backbone is based on VGG; we grab its 'backbone' directly
ssd_backbone = ssd_model.backbone

# Step 2: Extract MobileNetV3's classifier head
mobilenet_classifier = mobilenet.classifier

# Step 3: Define custom model combining SSD backbone + MobileNet head
class SSDMobileNetClassifier(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Needed to reduce SSD features
        self.classifier = classifier  # From MobileNetV3

    def forward(self, x):
        # Extract feature maps from SSD backbone
        features = self.backbone(x)["0"]  # take first feature map
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Build the combined model
custom_model = SSDMobileNetClassifier(ssd_backbone, mobilenet_classifier)
custom_model.eval()

torch.save(custom_model, 'ssd_mobilenet_combined_full.pth')
torch.save(custom_model.state_dict(), 'ssd_mobilenet_combined_state_dict.pth')