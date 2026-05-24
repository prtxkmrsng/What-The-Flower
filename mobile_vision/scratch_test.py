import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import json

def test():
    print("Testing models compatibility...")
    image_path = "D:\Programming\Projects\What-The-Flower\mobile_vision\image.png"
    
    # 1. Rebuild models
    num_classes = 10000
    model_pth = models.mobilenet_v3_large()
    in_features = model_pth.classifier[3].in_features
    model_pth.classifier[3] = nn.Linear(in_features, num_classes)
    
    model_pth.load_state_dict(torch.load("best_plant_model.pth", map_location="cpu"))
    model_pth.eval()
    
    # Load .ptl
    model_ptl = torch.jit.load("best_plant_model.ptl")
    model_ptl.eval()
    
    # 2. Preprocess image
    img = Image.open(image_path).convert("RGB")
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = val_transforms(img).unsqueeze(0)
    
    # Print first few tensor elements to verify
    print("Tensor first 5 values (Ch 0):", tensor[0, 0, 0, :5].tolist())
    
    # 3. Predict PTH
    with torch.no_grad():
        out_pth = model_pth(tensor)
        prob_pth = torch.softmax(out_pth, dim=1)
        max_val_pth, max_idx_pth = torch.max(prob_pth, 1)
        
    print(f"PTH Prediction Index: {max_idx_pth.item()}, Prob: {max_val_pth.item():.4f}")
    
    # 4. Predict PTL
    with torch.no_grad():
        out_ptl = model_ptl(tensor)
        # Note: If .ptl already has softmax inside, we shouldn't apply it again!
        # Let's see the outputs
        print("PTL Output Raw shape:", out_ptl.shape)
        print("PTL Output Sum:", out_ptl.sum().item())
        
        # If output sum is ~1.0, it's already softmaxed
        if abs(out_ptl.sum().item() - 1.0) < 0.1:
            print("PTL Output is ALREADY SOFTMAXED")
            prob_ptl = out_ptl
        else:
            print("PTL Output is LOGITS, applying softmax")
            prob_ptl = torch.softmax(out_ptl, dim=1)
            
        max_val_ptl, max_idx_ptl = torch.max(prob_ptl, 1)
        
    print(f"PTL Prediction Index: {max_idx_ptl.item()}, Prob: {max_val_ptl.item():.4f}")
    
    if max_idx_pth.item() == max_idx_ptl.item():
        print("SUCCESS: PTH and PTL predictions MATCH perfectly!")
    else:
        print("FAIL: Predictions MISMATCH!")

if __name__ == "__main__":
    test()
