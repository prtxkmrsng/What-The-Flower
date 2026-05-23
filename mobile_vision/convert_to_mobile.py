import torch
import torch.nn as nn
from torchvision import models
import json
import os

def convert():
    # 1. Dynamically read your num_classes matching your class_names mapping file
    if not os.path.exists('class_names.json'):
        raise FileNotFoundError("class_names.json not found! Please make sure it's in this directory.")
        
    with open('class_names.json', 'r') as f:
        categories = json.load(f)
    num_classes = len(categories)
    print(f"Detected {num_classes} categories from class_names.json")

    # 2. Rebuild the precise architecture configured in your training loop
    model = models.mobilenet_v3_large(weights=None) # Start without default ImageNet weights
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    # 3. Load your trained local checkpoint weights
    checkpoint_path = 'best_plant_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find model file: {checkpoint_path}")
        
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    # 4. Critical Wrapper Block: Mobile platforms need probability distributions (0 to 1), 
    # but CrossEntropyLoss processes raw un-normalized logits. We bundle Softmax into the model export.
    class MobileWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            logits = self.base_model(x)
            return self.softmax(logits)

    mobile_ready_model = MobileWrapper(model)
    mobile_ready_model.eval()

    # 5. Create a fake image tensor mapping your exact validation shape (Batch 1, 3 Channels, 224x224)
    example_input = torch.rand(1, 3, 224, 224)

    # 6. Trace and compile structural execution paths
    print("Tracing execution architecture...")
    traced_script_module = torch.jit.trace(mobile_ready_model, example_input)

    # 7. Apply hardware-level mobile kernel optimizations
    print("Applying mobile structural optimization engines...")
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_script_module)

    # 8. Save the cross-platform bytecode flat file
    output_filename = "best_plant_model.ptl"
    optimized_model._save_for_lite_interpreter(output_filename)
    print(f"✨ Success! Your PyTorch Mobile Lite file has been generated: '{output_filename}'")

if __name__ == '__main__':
    convert()