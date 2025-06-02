import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from dataset import get_transform  # You can use your get_transform or define transforms here
from model import SSDMobileNetClassifier
import utils
import os
import urllib.request
import zipfile
import random

def download_and_extract_tiny_imagenet(data_dir="./data"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_dir, "tiny-imagenet-200")
    if not os.path.exists(extract_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Done!")
    else:
        print("Tiny ImageNet already downloaded and extracted.")
    return extract_path

def get_random_split(dataset, total_images=30000, train_ratio=0.7, seed=42):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:total_images]
    split = int(train_ratio * total_images)
    train_indices = selected_indices[:split]
    test_indices = selected_indices[split:]
    return train_indices, test_indices

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print("Epoch:", epoch+1)
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            print(inputs.shape)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Save model checkpoint
        utils.save_checkpoint(model, optimizer, epoch, epoch_loss, f"checkpoint_epoch_{epoch+1}.pth")

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 25

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load Tiny ImageNet dataset
    tiny_imagenet_dir = download_and_extract_tiny_imagenet()
    train_dir = os.path.join(tiny_imagenet_dir, "train")
    full_dataset = ImageFolder(root=train_dir, transform=transform)

    # Get 30,000 random images and split
    train_indices, test_indices = get_random_split(full_dataset, total_images=30000, train_ratio=0.7)
    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # === 2. Re-load pretrained SSD + MobileNet heads ===
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    ssd_backbone = ssd_model.backbone
    mobilenet_classifier = mobilenet.classifier
    print(mobilenet_classifier)
    # exit()
    conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
    model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)

    # Save the final model
    torch.save(model.state_dict(), "final_ssd_mobilenet_Imagenet.pth")

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

if __name__ == '__main__':
    main()