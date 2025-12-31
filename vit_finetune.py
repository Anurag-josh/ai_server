import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Load datasets
print("Loading datasets...")
train_data = ImageFolder("dataset/train", transform=transform)
val_data = ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

num_classes = len(train_data.classes)
print(f"Detected classes: {train_data.classes} ({num_classes})")

# Save class names to file for reference
with open("class_names.txt", "w") as f:
    f.write("\n".join(train_data.classes))

# Prepare id2label and label2id dicts
id2label = {i: label for i, label in enumerate(train_data.classes)}
label2id = {label: i for i, label in enumerate(train_data.classes)}

# Load pre-trained ViT model with correct number of labels and mappings
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id
).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
print("Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1} Batch {batch_idx}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} finished. Average Loss = {avg_loss:.4f}")

# Save the fine-tuned model and feature extractor properly
save_dir = "vit_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
feature_extractor.save_pretrained(save_dir)

print(f"Model and feature extractor saved to '{save_dir}'")
