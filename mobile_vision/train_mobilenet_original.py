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
    def __init__(self, patience=5, min_delta=0.01):
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
if __name__ == '__main__': # <--- THIS IS THE MAGIC LINE
    
    # CONFIGURATION
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 5
    DATA_DIR = './data'
    LOG_DIR = './runs/inaturalist_mobilenet_v3'
    SAVE_PATH = 'best_plant_model.pth'

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
    train_dataset = datasets.INaturalist(root=DATA_DIR, version='2021_train_mini', download=False, transform=train_transforms)
    val_dataset = datasets.INaturalist(root=DATA_DIR, version='2021_valid', download=False, transform=val_transforms)

    # Note: num_workers=4 is fine NOW because of the 'if __name__' guard
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
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=PATIENCE)
    writer = SummaryWriter(LOG_DIR)
    scaler = torch.amp.GradScaler('cuda')

    # TRAINING LOOP
    print(f"Starting training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
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
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Summary & Validation
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        epoch_time = time.time() - start_time

        print(f"\n>> Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        early_stopper(avg_val_loss, model, SAVE_PATH)
        if early_stopper.early_stop:
            print("!!! Early stopping triggered. !!!")
            break

    writer.close()