import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO class labels for SSD model
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load pre-trained model
model = ssd300_vgg16(pretrained=True)
model.eval()

# Load image and preprocess
image_path = 'elephant.jpg'
image = Image.open(image_path).convert("RGB")
image_resized = F.resize(image, [300, 300])
input_tensor = F.to_tensor(image_resized).unsqueeze(0)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)

# Process predictions
threshold = 0.5
pred_boxes = outputs[0]['boxes'].cpu()
pred_labels = outputs[0]['labels'].cpu()
pred_scores = outputs[0]['scores'].cpu()

# Draw detections
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
    if score > threshold:
        class_name = COCO_CLASSES[label]
        print(f"Detected: {class_name} ({score:.2f})")

        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{class_name} {score:.2f}', color='white',
                bbox=dict(facecolor='red', alpha=0.5))

plt.axis('off')
plt.show()

torch.save(model, 'ssd300_vgg16_full.pth')
torch.save(model.state_dict(), 'ssd300_vgg16_state_dict.pth')