import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import urllib.request

# === Define the same custom class ===
class SSDMobileNetClassifier(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)["0"]
        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# === Create safe_globals dictionary ===
safe_globals = {
    'SSDMobileNetClassifier': SSDMobileNetClassifier
}

# === Load the model safely ===
from torch.serialization import safe_load

try:
    with safe_load(open("ssd_mobilenet_combined_full.pth", "rb"), safe_globals=safe_globals) as loaded:
        model = loaded
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# === Preprocess image ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

image_path = 'elephant.jpg'  # Replace with actual image path
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(image).unsqueeze(0).to(device)

# === Run inference ===
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)

# === Load ImageNet labels ===
label_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
label_path = 'imagenet_classes.txt'
urllib.request.urlretrieve(label_url, label_path)

with open(label_path) as f:
    class_names = [line.strip() for line in f.readlines()]

top_prob, top_idx = torch.topk(probs, 1)
print(f"Predicted: {class_names[top_idx.item()]} ({top_prob.item():.4f})")
