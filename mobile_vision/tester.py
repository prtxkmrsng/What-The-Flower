import torch
import torch.nn as nn
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

import json
class_names_path = "class_names.json"  
with open(class_names_path, 'r') as f:
    class_names = json.load(f)


val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_dataset = datasets.INaturalist(root="data", version='2021_valid', download=False, transform=val_transforms)

def get_flower_name(flower_id):
    try:
        # Access by index directly if it's a list
        return class_names[flower_id] 
    except IndexError:
        return "Unknown Flower"


# Get the number of classes
num_classes = len(val_dataset.all_categories)
# 1. Re-build the same MobileNet-V3 structure
model = models.mobilenet_v3_large()
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

# 2. Load the weights you saved during training
model.load_state_dict(torch.load("best_plant_model.pth"))
model.eval() # Set to "I'm just guessing, not learning" mode

# 3. Prepare the image preprocessing (Must match the training steps!)
preprocess = transforms.Compose([
     transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0) # Add a "batch" dimension

    with torch.no_grad():
        output = model(batch_t)
    
    # Get the ID of the flower with the highest score
    _, index = torch.max(output, 1)
    return index.item()

# Example usage (Point it to any flower image on your D: drive)
flower_id = predict("image.png")
print(f"Predicted Plant: {get_flower_name(flower_id)}")

