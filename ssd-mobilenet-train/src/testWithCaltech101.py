import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
from model import SSDMobileNetClassifier
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations (must match training)
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset and split (must match training split)
    data_dir = "./data/101_ObjectCategories"
    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Rebuild model structure
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    ssd_backbone = ssd_model.backbone
    mobilenet_classifier = mobilenet.classifier
    conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
    model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
    model = model.to(device)

    # Load trained weights
    model.load_state_dict(torch.load("final_ssd_mobilenet_caltech101_resumed.pth", map_location=device))
    model.eval()

    # Get class names
    class_names = dataset.classes

    # Evaluate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for batchidx,(images, labels) in enumerate(test_loader):
            print("\r[Batch {}/{}]".format(batchidx + 1, len(test_loader)), end="")
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy on Caltech101: {accuracy:.2f}%")

    # Predict and print class name for a sample image from test set
    sample_img_path, _ = test_dataset.dataset.samples[test_dataset.indices[0]]
    sample_img = Image.open(sample_img_path).convert("RGB")
    input_tensor = transform(sample_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        print(f"Sample image: {os.path.basename(sample_img_path)}")
        print(f"Predicted class: {class_names[pred_idx]}")

if __name__ == "__main__":
    main()