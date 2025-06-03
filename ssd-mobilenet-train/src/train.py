import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from dataset import CustomDataset, get_transform  # Assuming you have a dataset class defined in dataset.py
from model import SSDMobileNetClassifier
import utils

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print("Epoch:", epoch+1)
        print("Dataloader size:", dataloader.__len__())
        for index, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            print("\r[Batch {}]".format(index), end='', flush=True)
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
        utils.save_checkpoint(model, optimizer, epoch, epoch_loss, f"checkpoint_epoch_Cifar_{epoch+1}.pth")

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 15

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Download CIFAR-10 dataset and use only 7,000 samples
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset_indices = list(range(7000))
    train_subset = Subset(train_dataset, subset_indices)
    print(train_subset)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

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
    torch.save(model.state_dict(), "final_ssd_mobilenet_Cifar.pth")

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

if __name__ == '__main__':
    main()