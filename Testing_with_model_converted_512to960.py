import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms
from PIL import Image
import urllib.request

# === 1. Rebuild model structure ===
class SSDMobileNetClassifier(nn.Module):
    def __init__(self, backbone, conv_512_to_960, classifier):
        super().__init__()
        self.backbone = backbone
        self.channel_mapper = conv_512_to_960
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)["0"]
        x = self.channel_mapper(features)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# === 2. Re-load pretrained SSD + MobileNet heads ===
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

ssd_backbone = ssd_model.backbone
mobilenet_classifier = mobilenet.classifier
print(mobilenet_classifier)
# exit()
conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)

model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
model.load_state_dict(torch.load('ssd_mobilenet_combined_pretrained_state_dict.pth', map_location='cpu'))
model.eval()

# === 3. Image Preprocessing ===
image_path = 'elephant.jpg'  # <-- Replace with your image path
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((300, 300)),  # SSD input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0)
# === 4. Inference ===
with torch.no_grad():
    # output = model(input_tensor)
    output = model(torch.randn(4, 3, 512, 512))  # Dummy input for testing
    print("Output shape: ", output.shape)  # Should be [4, 1000] for batch size of 4
    print(output.cpu().numpy())
    probs = torch.nn.functional.softmax(output[0], dim=0)
# === 5. Print Top Prediction ===
label_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
label_path = 'imagenet_classes.txt'
urllib.request.urlretrieve(label_url, label_path)

with open(label_path) as f:
    class_names = [line.strip() for line in f.readlines()]

top_prob, top_idx = torch.topk(probs, 1)
print(top_idx.item())
print(top_prob)
print(f"Predicted: {class_names[top_idx.item()]} ({top_prob.item():.4f})")
