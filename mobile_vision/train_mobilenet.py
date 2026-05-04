import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json

# ==========================================
# 1. EARLY STOPPING LOGIC
# ==========================================
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
            print(f"   [+] Validation Loss improved. Model saved to {path}")
        else:
            self.counter += 1
            print(f"   [-] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

# ==========================================
# 2. MAIN EXECUTION BLOCK (THE GUARD)
# ==========================================
if __name__ == '__main__': 
    
    # CONFIGURATION
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001      # Lowered for fine-tuning
    EPOCHS = 100
    PATIENCE = 10               # Increased patience
    DATA_DIR = './data'
    LOG_DIR = './runs/inaturalist_mobilenet_v3'
    SAVE_PATH = 'best_plant_model.pth'
    
    # --- NEW: RESUME SETTINGS ---
    RESUME_TRAINING = True
    START_EPOCH = 38           # Start counting from 11 so TensorBoard graphs align

    print("Preparing datasets...")
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Loading iNaturalist
    train_dataset = datasets.INaturalist(root=DATA_DIR, version='2021_train_mini', download=True, transform=train_transforms)
    val_dataset = datasets.INaturalist(root=DATA_DIR, version='2021_valid', download=True, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    with open('class_names.json', 'w') as f:
        json.dump(train_dataset.all_categories, f)

    num_classes = len(train_dataset.all_categories)
    print(f"Setup complete. Training on {num_classes} classes.")

    # MODEL SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    # --- LOAD SAVED WEIGHTS BEFORE MOVING TO GPU ---
    if RESUME_TRAINING and os.path.exists(SAVE_PATH):
        print(f"Loading existing weights from {SAVE_PATH} to resume training...")
        model.load_state_dict(torch.load(SAVE_PATH))
    else:
        print("Starting completely fresh training...")

    model = model.to(device)

    # Note: Using the new, lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=PATIENCE)
    writer = SummaryWriter(LOG_DIR)
    scaler = torch.amp.GradScaler('cuda')

    # TRAINING LOOP
    print(f"Resuming training on {torch.cuda.get_device_name(0)} starting at Epoch {START_EPOCH}...")

    # --- LOOP STARTS AT START_EPOCH ---
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train_top1 = 0
        correct_train_top5 = 0  
        total_train = 0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            # --- TOP-1 and TOP-5 ACCURACY CALCULATION ---
            total_train += labels.size(0)
            
            # Top-1
            _, predicted = torch.max(outputs.data, 1)
            correct_train_top1 += (predicted == labels).sum().item()
            
            # Top-5
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            labels_reshaped = labels.view(-1, 1).expand_as(top5_preds)
            correct_train_top5 += (top5_preds == labels_reshaped).sum().item()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS-1}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Summary & Validation
        avg_train_loss = running_loss / len(train_loader)
        train_acc_top1 = 100 * correct_train_top1 / total_train
        train_acc_top5 = 100 * correct_train_top5 / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val_top1 = 0
        correct_val_top5 = 0 
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                total_val += labels.size(0)
                
                # Validation Top-1
                _, predicted = torch.max(outputs.data, 1)
                correct_val_top1 += (predicted == labels).sum().item()
                
                # Validation Top-5
                _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
                labels_reshaped = labels.view(-1, 1).expand_as(top5_preds)
                correct_val_top5 += (top5_preds == labels_reshaped).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc_top1 = 100 * correct_val_top1 / total_val
        val_acc_top5 = 100 * correct_val_top5 / total_val
        epoch_time = time.time() - start_time

        print(f"\n>> Epoch {epoch} | Time: {epoch_time:.2f}s")
        print(f"   Train Top-1: {train_acc_top1:.2f}% | Train Top-5: {train_acc_top5:.2f}%")
        print(f"   Val Top-1:   {val_acc_top1:.2f}% | Val Top-5:   {val_acc_top5:.2f}%")

        # Write to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy_Top1/Train', train_acc_top1, epoch)
        writer.add_scalar('Accuracy_Top1/Validation', val_acc_top1, epoch)
        writer.add_scalar('Accuracy_Top5/Train', train_acc_top5, epoch)
        writer.add_scalar('Accuracy_Top5/Validation', val_acc_top5, epoch)

        # --- EARLY STOPPING & SECOND WIND LOGIC ---
        early_stopper(avg_val_loss, model, SAVE_PATH)
        
        if early_stopper.early_stop:
            if avg_val_loss >= 1.0 and epoch < 95:
                print("\n   [!] Plateau detected, but goals not met. Initiating Second Wind!")
                
                # 1. Drop the learning rate by a factor of 10
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr / 10
                    new_lr = param_group['lr']
                print(f"   [!] Learning Rate dropped: {old_lr:.6f} -> {new_lr:.6f}")
                
                # 2. Increase patience by 5
                early_stopper.patience += 5
                print(f"   [!] Patience increased to: {early_stopper.patience}")
                
                # 3. Reset the early stopper to allow training to continue
                early_stopper.counter = 0
                early_stopper.early_stop = False
                
                # 4. Reload the best weights to escape the stagnant state
                print(f"   [!] Reloading best weights from {SAVE_PATH} to resume...")
                model.load_state_dict(torch.load(SAVE_PATH))
            else:
                # Actual stop if loss < 1.0 or we reached the epoch limit
                print("\n!!! True Early stopping triggered. Training complete. !!!")
                break

    writer.close()