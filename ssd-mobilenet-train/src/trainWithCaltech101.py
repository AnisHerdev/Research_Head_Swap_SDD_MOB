import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Caltech101
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from model import SSDMobileNetClassifier
from torchvision.datasets import ImageFolder
import utils

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 20

    # Data transformations
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # No augmentation for validation/test
    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    data_dir = "./data/101_ObjectCategories"
    full_dataset = ImageFolder(root=data_dir, transform=None)  # We'll set transform per split

    # Split into train/test (70% train, 30% test)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Assign transforms to each split
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # === 2. Re-load pretrained SSD + MobileNet heads ===
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    ssd_backbone = ssd_model.backbone
    mobilenet_classifier = mobilenet.classifier
    conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
    model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print("\r[Batch {}/{}]".format(batch_idx + 1, len(train_loader)), end="")
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        utils.save_checkpoint(
            model, optimizer, epoch, epoch_loss,
            f"checkpoints_catech_agumented/caltech101_checkpoint_epoch_{epoch+1}.pth"
        )
        torch.cuda.empty_cache()

    # Save the final model
    torch.save(model.state_dict(), "final_ssd_mobilenet_caltech101.pth")

if __name__ == '__main__':
    main()