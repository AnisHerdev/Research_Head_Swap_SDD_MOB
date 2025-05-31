import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# === 1. Define the combined model ===
class SSDMobileNetClassifier(nn.Module):
    def __init__(self, backbone, conv_512_to_960, classifier):
        super().__init__()
        self.backbone = backbone
        self.channel_mapper = conv_512_to_960
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)["0"]          # [B, 512, H, W]
        x = self.channel_mapper(features)         # [B, 960, H, W]
        x = self.avgpool(x)                       # [B, 960, 1, 1]
        x = torch.flatten(x, 1)                   # [B, 960]
        x = self.classifier(x)                    # [B, 1000]
        return x

# === 2. Load pretrained models ===
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# torch.save(ssd_model.state_dict(), 'ssd_mobilenet_combined_pretrained_state_dict_0000000.pth')
# torch.save(mobilenet.state_dict(), 'mobilenet_v3_large_pretrained_state_dict_00000000.pth')
ssd_backbone = ssd_model.backbone
mobilenet_classifier = mobilenet.classifier
conv_512_to_960 = nn.Conv2d(in_channels=512, out_channels=960, kernel_size=1)

# === 3. Build and save combined model ===
combined_model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
combined_model.eval()

torch.save(combined_model.state_dict(), 'ssd_mobilenet_combined_pretrained_state_dict.pth')
print(" Model saved successfully.")
