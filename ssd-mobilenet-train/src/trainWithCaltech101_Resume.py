import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from model import SSDMobileNetClassifier
import utils

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 32
    learning_rate = 0.0001
    extra_epochs = 10  # Number of extra epochs to train
    resume_checkpoint = "caltech101_checkpoint_epoch_10.pth"  # Change to your checkpoint

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_dir = "./data/101_ObjectCategories"
    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # === Rebuild model ===
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    ssd_backbone = ssd_model.backbone
    mobilenet_classifier = mobilenet.classifier
    conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
    model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === Load checkpoint ===
    start_epoch = 0
    if resume_checkpoint:
        print(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # === Continue training ===
    for epoch in range(start_epoch, start_epoch + extra_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch [{epoch+1}/{start_epoch + extra_epochs}]")
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
        print(f"Epoch [{epoch+1}/{start_epoch + extra_epochs}], Loss: {epoch_loss:.4f}")
        utils.save_checkpoint(model, optimizer, epoch, epoch_loss, f"caltech101_checkpoint_epoch_resumed_{epoch+1}.pth")
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), "final_ssd_mobilenet_caltech101_resumed.pth")

if __name__ == '__main__':
    main()