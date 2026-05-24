import torch
import torch.nn as nn
from torchvision import models
import os

class MobileWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base_model(x)
        return self.softmax(logits)

def export_clean():
    print("Preparing to export clean PTL model...")
    num_classes = 10000
    
    # 1. Rebuild architecture
    model = models.mobilenet_v3_large()
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    # 2. Load weights
    weight_path = "best_plant_model.pth"
    if not os.path.exists(weight_path):
        print(f"ERROR: {weight_path} not found in current directory! Please run this script in the directory containing your .pth model.")
        return

    print("Loading weights...")
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    
    # 3. CRITICAL: Eval mode
    model.eval()

    # 4. Wrap with Softmax (matches your current PTL structure)
    mobile_ready = MobileWrapper(model)
    mobile_ready.eval()

    # 5. Trace
    print("Tracing model...")
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(mobile_ready, example_input)

    # 6. Save DIRECTLY without optimize_for_mobile (avoids weight corruption bugs!)
    output_path = "assets/best_plant_model.ptl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_model._save_for_lite_interpreter(output_path)
    
    print(f"SUCCESS! Clean model exported to {output_path} (Bypassed optimize_for_mobile)")
    print("Now run 'python scratch_test.py' again to see if they match!")

if __name__ == "__main__":
    export_clean()
