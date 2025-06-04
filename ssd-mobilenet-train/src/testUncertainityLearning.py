import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import sys

# ============ Device Setup ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============ Custom Model ============
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
        return self.classifier(x)

# ============ Load Model ============
def load_model(checkpoint_path):
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    mobilenet.classifier[-1] = nn.Linear(mobilenet.classifier[-1].in_features, 10)
    conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)

    model = SSDMobileNetClassifier(ssd_model.backbone, conv_512_to_960, mobilenet.classifier)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ============ Data Preparation ============
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# ============ Evaluation ============
def evaluate(model, dataloader):
    correct = total = 0
    with torch.no_grad():
        for idx,(inputs, labels) in enumerate(dataloader):
            print(f"\r[{idx+1}/{len(dataloader)}] Processing batch...", end="")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ============ Main ============
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_model.pth>")
        sys.exit(1)

    model_path = sys.argv[1]
    model = load_model(model_path)
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy of {model_path}: {accuracy:.4f}")
