import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from dataset import CustomDataset, get_transform  # Assuming you have a dataset class defined in dataset.py
from model import SSDMobileNetClassifier
import utils
import random

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

def uncertainty_training(model, dataset, criterion, optimizer, device, num_epochs=25, initial_labelled_size=1000, batch_size=16, add_to_labelled_size=4000, max_accuracy=0.95):
    # Step 1: Randomly select 1k images for labelled dataset (L)
    labelled_indices = random.sample(range(len(dataset)), initial_labelled_size)
    unlabelled_indices = list(set(range(len(dataset))) - set(labelled_indices))
    
    labelled_dataset = torch.utils.data.Subset(dataset, labelled_indices)
    unlabelled_dataset = torch.utils.data.Subset(dataset, unlabelled_indices)
    
    labelled_loader = DataLoader(labelled_dataset, batch_size=batch_size, shuffle=True)
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False)
    
    for iteration in range(num_epochs):
        print(f"Iteration {iteration+1}/{num_epochs}")
        
        # Step 3: Train model on labelled dataset (L)
        train_model(model, labelled_loader, criterion, optimizer, device, num_epochs=1)
        
        # Step 4: Get confidence scores for unlabelled dataset (U)
        model.eval()
        confidence_scores = []
        with torch.no_grad():
            for inputs, _ in unlabelled_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probabilities, dim=1)
                confidence_scores.extend(max_probs.cpu().numpy())
        
        # Rank images in ascending order of confidence scores
        ranked_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i])
        
        # Select the first 4k images with lowest confidence scores
        selected_indices = [unlabelled_indices[i] for i in ranked_indices[:add_to_labelled_size]]
        
        # Step 5: Add selected images to labelled dataset (L) and remove from unlabelled dataset (U)
        labelled_indices.extend(selected_indices)
        unlabelled_indices = list(set(unlabelled_indices) - set(selected_indices))
        
        labelled_dataset = torch.utils.data.Subset(dataset, labelled_indices)
        unlabelled_dataset = torch.utils.data.Subset(dataset, unlabelled_indices)
        
        labelled_loader = DataLoader(labelled_dataset, batch_size=batch_size, shuffle=True)
        unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Labelled dataset size: {len(labelled_indices)}, Unlabelled dataset size: {len(unlabelled_indices)}")
        
        # Test the model every 2 iterations
        if (iteration + 1) % 2 == 0:
            accuracy = test_model(model, labelled_loader, device)
            print(f"Accuracy after iteration {iteration+1}: {accuracy:.4f}")
            if accuracy >= max_accuracy:
                print(f"Stopping training as accuracy reached {accuracy:.4f}")
                break
    
    return model

def test_model(model, dataloader, device):
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
    return correct / total

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 5

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Download CIFAR-10 dataset
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Load pretrained SSD + MobileNet heads
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    ssd_backbone = ssd_model.backbone
    mobilenet_classifier = mobilenet.classifier
    conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
    model = SSDMobileNetClassifier(ssd_backbone, conv_512_to_960, mobilenet_classifier)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Perform uncertainty training
    model = uncertainty_training(model, dataset, criterion, optimizer, device, num_epochs, initial_labelled_size=1000, batch_size=batch_size, add_to_labelled_size=4000)

    # Save the final model
    torch.save(model.state_dict(), "final_ssd_mobilenet_uncertainty.pth")

if __name__ == '__main__':
    main()
