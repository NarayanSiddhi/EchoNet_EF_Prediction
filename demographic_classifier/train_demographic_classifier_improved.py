"""
IMPROVED VERSION: Demographic Classifier with better architecture
- Uses pretrained 3D ResNet backbone
- Better handling of class imbalance
- Improved data augmentation
- Learning rate scheduling
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score, classification_report
import json
import torchvision.models as models
from torchvision.models.video import r3d_18, R3D_18_Weights


# Import the dataset class (same as before)
from train_demographic_classifier import DemographicVideoDataset


class ImprovedDemographicClassifier3D(nn.Module):
    """Improved 3D CNN using pretrained R3D-18 backbone"""
    def __init__(self, num_age_bins=8, num_bmi_cats=4, pretrained=True):
        super().__init__()
        
        # Use pretrained R3D-18 (3D ResNet) as backbone
        if pretrained:
            backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        else:
            backbone = r3d_18(weights=None)
        
        # Remove the final classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Get feature dimension (512 for R3D-18)
        feature_dim = 512
        
        # Classification heads with better architecture
        self.sex_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # F, M, O
        )
        
        self.age_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_age_bins)
        )
        
        self.bmi_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_bmi_cats)
        )
    
    def forward(self, x):
        # x: [B, 1, T, H, W]
        # R3D expects [B, 3, T, H, W] - repeat grayscale to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        
        # Extract features
        features = self.backbone(x)
        # Global average pooling
        features = features.view(features.size(0), -1)
        
        sex_logits = self.sex_head(features)
        age_logits = self.age_head(features)
        bmi_logits = self.bmi_head(features)
        
        return sex_logits, age_logits, bmi_logits


def get_class_weights(df, column):
    """Calculate class weights for imbalanced classes"""
    value_counts = df[column].value_counts()
    total = len(df)
    weights = {}
    for class_val, count in value_counts.items():
        weights[class_val] = total / (len(value_counts) * count)
    return weights


def train_epoch_improved(model, dataloader, criterion, optimizer, device, class_weights=None):
    """Train for one epoch with weighted loss"""
    model.train()
    total_loss = 0
    sex_preds, sex_labels = [], []
    age_preds, age_labels = [], []
    bmi_preds, bmi_labels = [], []
    
    for video, sex_label, age_label, bmi_label in tqdm(dataloader, desc="Training"):
        video = video.to(device)
        sex_label = sex_label.to(device)
        age_label = age_label.to(device)
        bmi_label = bmi_label.to(device)
        
        sex_logits, age_logits, bmi_logits = model(video)
        
        # Weighted loss if class weights provided
        if class_weights:
            sex_weights = torch.tensor([class_weights['sex'].get(i, 1.0) for i in range(3)], device=device)
            age_weights = torch.tensor([class_weights['age'].get(i, 1.0) for i in range(len(class_weights['age']))], device=device)
            bmi_weights = torch.tensor([class_weights['bmi'].get(i, 1.0) for i in range(4)], device=device)
            
            loss_sex = nn.CrossEntropyLoss(weight=sex_weights)(sex_logits, sex_label)
            loss_age = nn.CrossEntropyLoss(weight=age_weights)(age_logits, age_label)
            loss_bmi = nn.CrossEntropyLoss(weight=bmi_weights)(bmi_logits, bmi_label)
        else:
            loss_sex = criterion(sex_logits, sex_label)
            loss_age = criterion(age_logits, age_label)
            loss_bmi = criterion(bmi_logits, bmi_label)
        
        loss = loss_sex + loss_age + loss_bmi
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        
        sex_preds.extend(sex_logits.argmax(dim=1).cpu().numpy())
        sex_labels.extend(sex_label.cpu().numpy())
        age_preds.extend(age_logits.argmax(dim=1).cpu().numpy())
        age_labels.extend(age_label.cpu().numpy())
        bmi_preds.extend(bmi_logits.argmax(dim=1).cpu().numpy())
        bmi_labels.extend(bmi_label.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    sex_acc = accuracy_score(sex_labels, sex_preds)
    age_acc = accuracy_score(age_labels, age_preds)
    bmi_acc = accuracy_score(bmi_labels, bmi_preds)
    
    return avg_loss, sex_acc, age_acc, bmi_acc


def evaluate_improved(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    sex_preds, sex_labels = [], []
    age_preds, age_labels = [], []
    bmi_preds, bmi_labels = [], []
    
    with torch.no_grad():
        for video, sex_label, age_label, bmi_label in tqdm(dataloader, desc="Evaluating"):
            video = video.to(device)
            sex_label = sex_label.to(device)
            age_label = age_label.to(device)
            bmi_label = bmi_label.to(device)
            
            sex_logits, age_logits, bmi_logits = model(video)
            
            loss_sex = criterion(sex_logits, sex_label)
            loss_age = criterion(age_logits, age_label)
            loss_bmi = criterion(bmi_logits, bmi_label)
            loss = loss_sex + loss_age + loss_bmi
            
            total_loss += loss.item()
            
            sex_preds.extend(sex_logits.argmax(dim=1).cpu().numpy())
            sex_labels.extend(sex_label.cpu().numpy())
            age_preds.extend(age_logits.argmax(dim=1).cpu().numpy())
            age_labels.extend(age_label.cpu().numpy())
            bmi_preds.extend(bmi_logits.argmax(dim=1).cpu().numpy())
            bmi_labels.extend(bmi_label.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    sex_acc = accuracy_score(sex_labels, sex_preds)
    age_acc = accuracy_score(age_labels, age_preds)
    bmi_acc = accuracy_score(bmi_labels, bmi_preds)
    
    return {
        'loss': avg_loss,
        'sex_accuracy': sex_acc,
        'age_accuracy': age_acc,
        'bmi_accuracy': bmi_acc,
        'sex_preds': sex_preds,
        'sex_labels': sex_labels,
        'age_preds': age_preds,
        'age_labels': age_labels,
        'bmi_preds': bmi_preds,
        'bmi_labels': bmi_labels
    }


def main():
    print("\n" + "="*80)
    print("IMPROVED DEMOGRAPHIC CLASSIFIER TRAINING")
    print("Using pretrained R3D-18 backbone + weighted loss")
    print("="*80 + "\n")
    
    # Load config
    config_path = "ef_prediction/config.yaml"
    if not os.path.exists(config_path):
        config_path = "../ef_prediction/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create datasets
    train_manifest = "data/processed_full/train_manifest_filtered_clean.csv"
    val_manifest = "data/processed_full/val_manifest.csv"
    if not os.path.exists(train_manifest):
        train_manifest = "../data/processed_full/train_manifest_filtered_clean.csv"
        val_manifest = "../data/processed_full/val_manifest.csv"
    video_dir = cfg["data"]["original_video_dir"]
    if not os.path.exists(video_dir):
        video_dir = f"../{video_dir}"
    
    print("Loading datasets...")
    train_ds = DemographicVideoDataset(
        manifest_path=train_manifest,
        video_root_dir=video_dir,
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        augment=True
    )
    
    val_ds = DemographicVideoDataset(
        manifest_path=val_manifest,
        video_root_dir=video_dir,
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        augment=False
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}\n")
    
    # Calculate class weights for imbalanced classes
    print("Calculating class weights...")
    class_weights = {
        'sex': get_class_weights(train_ds.df, 'sex_label'),
        'age': get_class_weights(train_ds.df, 'age_label'),
        'bmi': get_class_weights(train_ds.df, 'bmi_label')
    }
    print("✓ Class weights calculated\n")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)  # Smaller batch for R3D
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    # Model
    num_age_bins = len(train_ds.df['age_bin'].unique())
    num_bmi_cats = 4
    model = ImprovedDemographicClassifier3D(num_age_bins=num_age_bins, num_bmi_cats=num_bmi_cats, pretrained=True).to(device)
    
    # Loss and optimizer with lower learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)  # Lower LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Training
    num_epochs = 50
    best_val_loss = float('inf')
    checkpoint_dir = Path("checkpoints_improved")
    if not checkpoint_dir.exists():
        checkpoint_dir = Path("../demographic_classifier/checkpoints_improved")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [], 'train_sex_acc': [], 'train_age_acc': [], 'train_bmi_acc': [],
        'val_loss': [], 'val_sex_acc': [], 'val_age_acc': [], 'val_bmi_acc': []
    }
    
    print("Starting training...\n")
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_sex_acc, train_age_acc, train_bmi_acc = train_epoch_improved(
            model, train_loader, criterion, optimizer, device, class_weights
        )
        
        # Validate
        val_results = evaluate_improved(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_sex_acc'].append(train_sex_acc)
        history['train_age_acc'].append(train_age_acc)
        history['train_bmi_acc'].append(train_bmi_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_sex_acc'].append(val_results['sex_accuracy'])
        history['val_age_acc'].append(val_results['age_accuracy'])
        history['val_bmi_acc'].append(val_results['bmi_accuracy'])
        
        print(f"\nTrain - Loss: {train_loss:.4f} | Sex Acc: {train_sex_acc:.4f} | Age Acc: {train_age_acc:.4f} | BMI Acc: {train_bmi_acc:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f} | Sex Acc: {val_results['sex_accuracy']:.4f} | Age Acc: {val_results['age_accuracy']:.4f} | BMI Acc: {val_results['bmi_accuracy']:.4f}")
        
        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save(model.state_dict(), checkpoint_dir / "best.pth")
            print("✓ Saved best model")
    
    # Save final model and history
    torch.save(model.state_dict(), checkpoint_dir / "final.pth")
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n🎉 Training completed!\n")


if __name__ == "__main__":
    main()
