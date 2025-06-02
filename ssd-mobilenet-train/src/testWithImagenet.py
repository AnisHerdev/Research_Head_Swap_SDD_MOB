import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn
import os
from model import SSDMobileNetClassifier

def load_test_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dir = os.path.join(data_dir, "train")  # Adjust if your test set is elsewhere
    full_dataset = ImageFolder(root=test_dir, transform=transform)
    # Use the same split logic as training
    indices = list(range(len(full_dataset)))
    test_size = int(0.3 * 10000)  # 30% of 10000
    test_indices = indices[:test_size]
    print(f"Number of test images: {len(test_indices)}")
    test_subset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader, full_dataset.classes

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    data_dir = "./data/tiny-imagenet-200"  # Adjust if needed

    # Load test data
    test_loader, class_names = load_test_data(data_dir, batch_size)

    # Rebuild model
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    ssd_backbone = ssd_model.backbone
    mobilenet_classifier = mobilenet.classifier
    conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
    model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
    model = model.to(device)

    # Load trained weights
    model.load_state_dict(torch.load("final_ssd_mobilenet_Imagenet_Resumed.pth", map_location=device))

    # Evaluate
    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")