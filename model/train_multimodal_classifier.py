#!/usr/bin/env python3
"""
Simple multi-modal classifier for PDDL label classification (grocery vs liquid)
Uses both images and text descriptions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json
import os
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 15
IMG_SIZE = 224
TEXT_DIM = 384  # Sentence transformer dimension
MODEL_DIR = "model/checkpoints"
DATASET_DIR = "dataset/multimodal_binary_classification"

class MultiModalDataset(Dataset):
    """Custom dataset for loading images + text"""
    
    def __init__(self, data_list, dataset_dir, split='train', transform=None, text_encoder=None):
        self.data_list = data_list
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.text_encoder = text_encoder
        self.label_to_idx = {'grocery': 0, 'liquid': 1}
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load image
        img_path = os.path.join(self.dataset_dir, self.split, item['image_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load text
        text_path = os.path.join(self.dataset_dir, self.split, item['text_path'])
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Encode text to embedding
        if self.text_encoder:
            text_embedding = self.text_encoder.encode(text, convert_to_tensor=True)
        else:
            text_embedding = torch.zeros(TEXT_DIM)  # Fallback
        
        # Get label
        label = self.label_to_idx[item['label']]
        
        return {
            'image': image,
            'text_embedding': text_embedding,
            'label': torch.tensor(label, dtype=torch.long),
            'class_name': item['class_name'],
            'coarse_class': item['coarse_class']
        }

class MultiModalClassifier(nn.Module):
    """Simple multi-modal classifier combining vision + text"""
    
    def __init__(self, num_classes=2, text_dim=TEXT_DIM):
        super(MultiModalClassifier, self).__init__()
        
        # Vision encoder (ResNet18)
        self.vision_encoder = models.resnet18(pretrained=True)
        vision_dim = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Identity()  # Remove final layer
        
        # Text encoder (already encoded by sentence transformer)
        self.text_processor = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, text_embedding):
        # Extract vision features
        vision_features = self.vision_encoder(image)
        
        # Process text features
        text_features = self.text_processor(text_embedding)
        
        # Concatenate and classify
        combined = torch.cat([vision_features, text_features], dim=1)
        output = self.fusion(combined)
        
        return output

def create_data_loaders():
    """Create train and validation data loaders"""
    
    # Load dataset manifests
    with open(os.path.join(DATASET_DIR, 'train_manifest.json'), 'r') as f:
        train_data = json.load(f)
    
    with open(os.path.join(DATASET_DIR, 'val_manifest.json'), 'r') as f:
        val_data = json.load(f)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Initialize text encoder
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MultiModalDataset(
        train_data, DATASET_DIR, split='train', transform=train_transform, text_encoder=text_encoder
    )
    
    val_dataset = MultiModalDataset(
        val_data, DATASET_DIR, split='val', transform=val_transform, text_encoder=text_encoder
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader

def train_model():
    """Train the multi-modal classifier"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders()
    
    # Initialize model
    model = MultiModalClassifier(num_classes=2)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            text_embeddings = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, text_embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = {'grocery': 0, 'liquid': 0}
        class_total = {'grocery': 0, 'liquid': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                text_embeddings = batch['text_embedding'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, text_embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_name = 'grocery' if label == 0 else 'liquid'
                    class_total[class_name] += 1
                    if label == pred:
                        class_correct[class_name] += 1
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        
        # Per-class accuracies
        grocery_acc = 100.0 * class_correct['grocery'] / class_total['grocery'] if class_total['grocery'] > 0 else 0
        liquid_acc = 100.0 * class_correct['liquid'] / class_total['liquid'] if class_total['liquid'] > 0 else 0
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_acc:.2f}%')
        print(f'  Grocery Accuracy: {grocery_acc:.2f}%')
        print(f'  Liquid Accuracy: {liquid_acc:.2f}%')
        print('-' * 60)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_accuracies': {'grocery': grocery_acc, 'liquid': liquid_acc}
            }, os.path.join(MODEL_DIR, 'best_multimodal_model.pth'))
            print(f'✓ New best model saved! Val Acc: {val_acc:.2f}%')
    
    print(f'Training complete! Best validation accuracy: {best_val_acc:.2f}%')
    return model, best_val_acc

def test_single_prediction(model_path, image_path, text_description, device):
    """Test single prediction"""
    
    # Load model
    model = MultiModalClassifier(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load text encoder
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process image
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Process text
    text_embedding = text_encoder.encode(text_description, convert_to_tensor=True)
    text_embedding = text_embedding.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image, text_embedding)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
    
    class_names = ['grocery', 'liquid']
    predicted_class = class_names[prediction.item()]
    confidence = probabilities[0][prediction.item()].item()
    
    return predicted_class, confidence

if __name__ == "__main__":
    model, best_acc = train_model()
    print(f"Training finished with best accuracy: {best_acc:.2f}%")