import torch
from torchvision import transforms
from PIL import Image
from model import SSDMobileNetClassifier
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn

# 1. Rebuild model structure
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
ssd_backbone = ssd_model.backbone
mobilenet_classifier = mobilenet.classifier
conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)

# 2. Load saved weights
model.load_state_dict(torch.load("final_ssd_mobilenet.pth", map_location="cpu"))
model.eval()

# 3. Preprocess input image
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = Image.open("elephant.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

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