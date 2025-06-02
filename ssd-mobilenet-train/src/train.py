import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset  # Assuming you have a dataset class defined in dataset.py
from model import SSDMobileNetClassifier
import utils

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Save model checkpoint
        utils.save_checkpoint(model, optimizer, epoch)

def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 25

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = CustomDataset(transform=transform)  # Implement your dataset class in dataset.py
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = SSDMobileNetClassifier()  # Initialize your model with appropriate parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)

if __name__ == '__main__':
    main()