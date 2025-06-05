import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import matplotlib.pyplot as plt

# ============ Configuration ============
resume_iter = 15  # <-- Change this to the iteration you want to resume from
total_iterations = 30
query_batch = 300
batch_size = 32

# ============ Device Setup ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ============ Custom Model ============
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
        return self.classifier(x)

# ============ Load Data ============
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ============ Load Saved State ============
labeled_indices = np.load(f"saved_state/labeled_indices_iter_{resume_iter}.npy").tolist()
unlabeled_indices = np.load(f"saved_state/unlabeled_indices_iter_{resume_iter}.npy").tolist()
accuracies = np.load("saved_state/accuracies.npy").tolist()

labeled_dataset = Subset(full_dataset, labeled_indices)
unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

# ============ Model ============
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

mobilenet.classifier[-1] = nn.Linear(mobilenet.classifier[-1].in_features, 10)
conv_512_to_960 = nn.Conv2d(512, 960, kernel_size=1)

model = SSDMobileNetClassifier(ssd_model.backbone, conv_512_to_960, mobilenet.classifier)
model.load_state_dict(torch.load(f"saved_state/model_iter_{resume_iter}.pth"))
model.to(device)

# ============ Training and Evaluation ============
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for idx, (inputs, labels) in enumerate(dataloader):
        print("\r[Training Batch {}/{}]".format(idx + 1, len(dataloader)), end="")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    print("Evaluating...")
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            print("\r[Eval Batch {}/{}]".format(idx + 1, len(dataloader)), end="")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print()
    return correct / total

# ============ Resume Active Learning ============
criterion = nn.CrossEntropyLoss()

for i in range(resume_iter, total_iterations):
    print(f"\n=== Active Learning Iteration {i+1} ===")

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.05)

    train_loss = train(model, labeled_loader, optimizer, criterion)
    acc = evaluate(model, test_loader)
    scheduler.step(acc)

    accuracies.append(acc)
    print(f"Train Loss: {train_loss:.4f} | Test Accuracy: {acc:.4f} | Labeled Size: {len(labeled_dataset)}")

    # Save everything again
    torch.save(model.state_dict(), f"saved_state/model_iter_{i+1}.pth")
    np.save(f"saved_state/labeled_indices_iter_{i+1}.npy", np.array(labeled_dataset.indices))
    np.save(f"saved_state/unlabeled_indices_iter_{i+1}.npy", np.array(unlabeled_dataset.indices))
    np.save("saved_state/accuracies.npy", np.array(accuracies))

    if len(unlabeled_dataset) == 0:
        print("No more unlabeled data to label.")
        break

    # Estimate uncertainty
    model.eval()
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size)
    uncertainties = []
    unlabeled_data_indices = []

    print("Computing uncertainties on unlabeled data...")
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(unlabeled_loader):
            print("\r[Uncertainty Batch {}/{}]".format(idx + 1, len(unlabeled_loader)), end="")
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            uncertainty = 1 - max_probs
            uncertainties.extend(uncertainty.cpu().numpy())

            start_idx = idx * batch_size
            end_idx = start_idx + inputs.size(0)
            batch_indices = unlabeled_dataset.indices[start_idx:end_idx]
            unlabeled_data_indices.extend(batch_indices)
    print()

    # Select top uncertain samples
    uncertainties = np.array(uncertainties)
    query_indices_local = np.argsort(-uncertainties)[:query_batch]
    query_indices_global = [unlabeled_data_indices[i] for i in query_indices_local]
    query_uncertainties = uncertainties[query_indices_local]

    # Log queried samples
    log_file = f"saved_state/label_log_iter_{i+1}.txt"
    with open(log_file, "w") as f:
        for idx, uncert in zip(query_indices_global, query_uncertainties):
            f.write(f"Index: {idx} | Uncertainty: {uncert:.4f}\n")

    # Update datasets
    labeled_dataset = Subset(full_dataset, labeled_dataset.indices + query_indices_global)
    unlabeled_dataset = Subset(full_dataset, [idx for idx in unlabeled_dataset.indices if idx not in query_indices_global])

    torch.cuda.empty_cache()

# ============ Final Accuracy Plot ============
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
plt.xlabel("Active Learning Iteration")
plt.ylabel("Test Accuracy")
plt.title("Resumed Active Learning Performance")
plt.grid(True)
plt.savefig("saved_state/resumed_accuracy_plot.png")
plt.show()
