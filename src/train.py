import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchgeo.models import resnet50

# ---- CONFIG ----
data_dir = "../data/"
batch_size = 16  # Reduced batch size for M1 memory constraints
num_epochs = 10
learning_rate = 1e-4
num_classes = 2  # solar vs no_solar

# ---- DEVICE SETUP FOR M1 ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---- DATA TRANSFORMS ----
# ResNet50 expects 224x224 inputs, normalized with ImageNet stats
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- DATASETS ----
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
val_dataset   = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)
test_dataset  = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

# ---- DATA LOADERS WITH M1 OPTIMIZATIONS ----
# Use fewer workers on M1 to avoid thread contention
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,  # Reduced from 4
    pin_memory=False  # Disable for MPS
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=False
)

print(f"Training on {len(train_dataset)} images, Validation on {len(val_dataset)} images")

# ---- MODEL ----
# TorchGeo provides ResNet variants with pretrained weights
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace classifier head

# Move model to device
model = model.to(device)

# ---- LOSS & OPTIMIZER ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---- TRAINING LOOP ----
print(f"Starting training on {device}...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

    train_acc = 100. * correct / total

    # ---- VALIDATION ----
    model.eval()
    val_correct, val_total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_acc:.2f}%")

# ---- TEST ----
model.eval()
test_correct, test_total = 0, 0
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100. * test_correct / test_total
avg_test_loss = test_loss / len(test_loader)

print(f"\n=== FINAL RESULTS ===")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ---- SAVE MODEL ----
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'accuracy': test_acc,
}, 'solar_panel_model_m1.pth')
print("Model saved as 'solar_panel_model_m1.pth'")