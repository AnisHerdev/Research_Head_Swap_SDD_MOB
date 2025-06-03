import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import random

# ============ Device Setup ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============ Your Custom Model ============
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
        x = self.classifier(x)
        return x

# Load pretrained parts
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)
model = SSDMobileNetClassifier(ssd_model.backbone, conv_512_to_960, mobilenet.classifier)
model.to(device)

# Optional: Load pretrained weights from file
# model.load_state_dict(torch.load('ssd_mobilenet_combined_pretrained_state_dict.pth', map_location=device))

# ============ Data Preparation ============
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split full dataset into 5% labeled, 70% unlabeled
total_size = len(full_dataset)
indices = list(range(total_size))
random.shuffle(indices)

labeled_size = int(0.05 * total_size)
unlabeled_size = int(0.70 * total_size)

labeled_indices = indices[:labeled_size]
unlabeled_indices = indices[labeled_size:labeled_size+unlabeled_size]

labeled_dataset = Subset(full_dataset, labeled_indices)
unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

# ============ Training and Evaluation ============
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ============ Active Learning Loop ============
batch_size = 64
query_batch = 100  # number of samples to label per iteration
n_iterations = 10
criterion = nn.CrossEntropyLoss()
test_loader = DataLoader(test_dataset, batch_size=batch_size)

accuracies = []

for i in range(n_iterations):
    print(f"\n=== Active Learning Iteration {i+1} ===")
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    train_loss = train(model, labeled_loader, optimizer, criterion)
    acc = evaluate(model, test_loader)
    accuracies.append(acc)
    
    print(f"Train Loss: {train_loss:.4f} | Test Accuracy: {acc:.4f} | Labeled Size: {len(labeled_dataset)}")
    
    # Save model after each iteration
    torch.save(model.state_dict(), f"active_learning_model_iter_{i+1}.pth")
    
    # Stop if no more unlabeled data
    if len(unlabeled_dataset) == 0:
        break
    
    # Select new uncertain samples to label
    model.eval()
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size)
    
    uncertainties = []
    all_inputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in unlabeled_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            uncertainty = 1 - max_probs
            uncertainties.extend(uncertainty.cpu().numpy())
            all_inputs.extend(inputs.cpu())
            all_targets.extend(targets.cpu())

    # Select top-K most uncertain samples
    uncertainties = np.array(uncertainties)
    query_indices = np.argsort(-uncertainties)[:query_batch]

    # Update labeled and unlabeled datasets
    new_data = [(all_inputs[idx], all_targets[idx]) for idx in query_indices]
    new_dataset = torch.utils.data.TensorDataset(
        torch.stack([item[0] for item in new_data]),
        torch.tensor([item[1] for item in new_data])
    )
    
    labeled_dataset = ConcatDataset([labeled_dataset, new_dataset])

    # Remove selected from unlabeled dataset
    remaining_indices = [i for j, i in enumerate(unlabeled_dataset.indices) if j not in query_indices]
    unlabeled_dataset = Subset(full_dataset, remaining_indices)

# ============ Plot Accuracy ============
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
plt.xlabel("Active Learning Iteration")
plt.ylabel("Test Accuracy")
plt.title("Active Learning Performance (Fine-Tuned SSD-MobileNet Classifier)")
plt.grid(True)
plt.show()
