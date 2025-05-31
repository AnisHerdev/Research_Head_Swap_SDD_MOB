import torch
from torchvision.models import mobilenet_v3_large
from torchvision import transforms
from PIL import Image
import urllib.request

# === Step 1: Load Pretrained MobileNetV3 Model ===
model = mobilenet_v3_large(pretrained=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# === Step 2: Preprocess Input Image ===
image_path = 'elephant.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

# === Step 3: Run Inference ===
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# === Step 4: Get Top-1 Prediction with Label ===
top_prob, top_idx = torch.topk(probabilities, 1)

# Download ImageNet class labels if not present
label_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
label_path = 'imagenet_classes.txt'
urllib.request.urlretrieve(label_url, label_path)

with open(label_path) as f:
    classes = [line.strip() for line in f.readlines()]

predicted_class = classes[top_idx.item()]
print(f"Predicted: {predicted_class} ({top_prob.item():.4f})")

# === Step 5: Save the Model ===
# Option 1: Save full model
torch.save(model, 'mobilenetv3_large_full.pth')

# Option 2: Save only the state_dict (recommended)
# torch.save(model.state_dict(), 'mobilenetv3_large_state_dict.pth')