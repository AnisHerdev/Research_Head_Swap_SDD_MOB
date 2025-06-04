import torch
from torchvision import transforms
from PIL import Image
from model import SSDMobileNetClassifier
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Rebuild model structure
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
ssd_backbone = ssd_model.backbone
mobilenet_classifier = mobilenet.classifier
conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
model = model.to(device)

# 2. Load saved weights
# model.load_state_dict(torch.load("active_learning_model_iter_10.pth", map_location=device))
# model.load_state_dict(torch.load("active_learning_model_iter_10.pth", map_location=device))
# model.eval()
checkpoint = torch.load("active_learning_model_iter_10.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Preprocess input image
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = Image.open("car.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 4. Run inference
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    print("Predicted class index:", predicted_class)
    print("Predicted class name:", cifar10_classes[predicted_class])

# 5. Calculate accuracy on the **last** 3000 samples of CIFAR-10 test set
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
total_test = len(test_dataset)
test_subset = Subset(test_dataset, list(range(total_test - 3000, total_test)))  # Use only last 3000 samples
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
print("Size of CIFAR-10 test subset:", len(test_subset))
with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        print("\r[Batch {}]".format(idx + 1), end='', flush=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy on CIFAR-10 (last 3000 samples): {accuracy:.2f}%")