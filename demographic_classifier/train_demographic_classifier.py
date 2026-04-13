"""
Train a demographic classifier to predict sex, age bin, and BMI category from videos.
This validates that synthetic videos encode demographic signals correctly.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json


class DemographicVideoDataset(Dataset):
    """Dataset for demographic classification from videos"""
    def __init__(self, manifest_path, video_root_dir, video_length=32, video_size=128, augment=False):
        self.df = pd.read_csv(manifest_path)
        self.video_root_dir = video_root_dir
        self.video_length = video_length
        self.video_size = video_size
        self.augment = augment
        
        # Encode demographics
        self._encode_labels()
        
    def _encode_labels(self):
        """Encode sex, age_bin, and BMI into categorical labels"""
        # Sex: F=0, M=1, O=2
        sex_map = {'F': 0, 'M': 1, 'O': 2}
        self.df['sex_label'] = self.df['sex'].map(sex_map).fillna(2)
        
        # Age bins: map age_bin string to integer
        age_bins = sorted(self.df['age_bin'].unique())
        age_map = {bin_name: idx for idx, bin_name in enumerate(age_bins)}
        self.df['age_label'] = self.df['age_bin'].map(age_map)
        
        # BMI category: calculate from weight/height if not present
        if 'bmi_category' not in self.df.columns:
            # Calculate BMI (handle missing values)
            self.df['bmi'] = self.df.apply(
                lambda row: row['weight'] / ((row['height'] / 100) ** 2) 
                if pd.notna(row['weight']) and pd.notna(row['height']) and row['height'] > 0 
                else np.nan, axis=1
            )
            # Categorize: Underweight < 18.5, Normal 18.5-25, Overweight 25-30, Obese > 30
            self.df['bmi_category'] = pd.cut(
                self.df['bmi'],
                bins=[0, 18.5, 25, 30, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                include_lowest=True
            )
            # Fill NaN with 'Normal' as default
            self.df['bmi_category'] = self.df['bmi_category'].fillna('Normal')
        
        bmi_map = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        self.df['bmi_label'] = self.df['bmi_category'].map(bmi_map).fillna(1)
        
        # Drop rows with missing labels
        self.df = self.df.dropna(subset=['sex_label', 'age_label', 'bmi_label']).reset_index(drop=True)
        
    def load_video(self, path):
        """Load and preprocess video"""
        if not os.path.exists(path):
            return None
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.video_size, self.video_size))
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            return None
            
        frames = np.stack(frames)
        
        # Sample or pad to video_length
        if len(frames) >= self.video_length:
            idx = np.linspace(0, len(frames) - 1, self.video_length).astype(int)
            frames = frames[idx]
        else:
            pad = self.video_length - len(frames)
            frames = np.pad(frames, ((0, pad), (0, 0), (0, 0)), mode='edge')
        
        frames = frames.astype(np.float32) / 255.0
        
        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                frames = np.flip(frames, axis=2)  # Horizontal flip
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.02, frames.shape).astype(np.float32)
                frames = np.clip(frames + noise, 0, 1)
        
        frames = torch.from_numpy(frames).float().unsqueeze(0)  # [1, T, H, W] - ensure float32
        return frames
    
    def _resolve_path(self, root, path):
        """Resolve video path"""
        if os.path.isabs(path):
            return path
        if root is None:
            return path
        if path.startswith(root):
            return path
        return os.path.join(root, path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                row = self.df.iloc[idx]
                
                # Resolve video path
                if 'processed_path' in row and pd.notna(row['processed_path']):
                    video_path = self._resolve_path(self.video_root_dir, row['processed_path'])
                elif 'file_path' in row and pd.notna(row['file_path']):
                    video_path = self._resolve_path(self.video_root_dir, row['file_path'])
                elif 'synthetic_path' in row and pd.notna(row['synthetic_path']):
                    video_path = self._resolve_path(self.video_root_dir, row['synthetic_path'])
                else:
                    # Try next sample
                    idx = (idx + 1) % len(self.df)
                    retry_count += 1
                    continue
                
                video = self.load_video(video_path)
                if video is None:
                    # Try next sample
                    idx = (idx + 1) % len(self.df)
                    retry_count += 1
                    continue
                
                # Success - extract labels
                break
            except Exception as e:
                # Try next sample on any error
                idx = (idx + 1) % len(self.df)
                retry_count += 1
                continue
        
        # If we exhausted retries, return a dummy sample (shouldn't happen often)
        if retry_count >= max_retries:
            # Return a dummy video with default labels (ensure float32)
            dummy_video = torch.zeros(1, self.video_length, self.video_size, self.video_size, dtype=torch.float32)
            return dummy_video, torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long), torch.tensor(1, dtype=torch.long)
        
        sex_label = torch.tensor(int(row['sex_label']), dtype=torch.long)
        age_label = torch.tensor(int(row['age_label']), dtype=torch.long)
        bmi_label = torch.tensor(int(row['bmi_label']), dtype=torch.long)
        
        return video, sex_label, age_label, bmi_label


class DemographicClassifier3D(nn.Module):
    """3D CNN for demographic classification"""
    def __init__(self, num_age_bins=8, num_bmi_cats=4):
        super().__init__()
        
        # 3D CNN backbone
        self.conv3d_layers = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            # Block 2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            # Block 3
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            # Block 4
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Classification heads
        self.sex_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # F, M, O
        )
        
        self.age_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_age_bins)
        )
        
        self.bmi_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_bmi_cats)
        )
    
    def forward(self, x):
        # x: [B, 1, T, H, W]
        features = self.conv3d_layers(x)
        features = features.view(features.size(0), -1)  # [B, 512]
        
        sex_logits = self.sex_head(features)
        age_logits = self.age_head(features)
        bmi_logits = self.bmi_head(features)
        
        return sex_logits, age_logits, bmi_logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
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
        
        loss_sex = criterion(sex_logits, sex_label)
        loss_age = criterion(age_logits, age_label)
        loss_bmi = criterion(bmi_logits, bmi_label)
        
        loss = loss_sex + loss_age + loss_bmi
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
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


def evaluate(model, dataloader, criterion, device):
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
    print("DEMOGRAPHIC CLASSIFIER TRAINING")
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
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)  # num_workers=0 to avoid recursion in workers
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    # Model
    num_age_bins = len(train_ds.df['age_bin'].unique())
    num_bmi_cats = 4
    model = DemographicClassifier3D(num_age_bins=num_age_bins, num_bmi_cats=num_bmi_cats).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    num_epochs = 50
    best_val_loss = float('inf')
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        checkpoint_dir = Path("../demographic_classifier/checkpoints")
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
        train_loss, train_sex_acc, train_age_acc, train_bmi_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_results['loss'])
        
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
    
    # Final evaluation with detailed metrics
    print("\n" + "="*80)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*80)
    final_results = evaluate(model, val_loader, criterion, device)
    
    print(f"\nSex Classification:")
    print(classification_report(final_results['sex_labels'], final_results['sex_preds'], 
                                target_names=['Female', 'Male', 'Other']))
    
    print(f"\nAge Classification:")
    age_bins = sorted(val_ds.df['age_bin'].unique())
    print(classification_report(final_results['age_labels'], final_results['age_preds'],
                                target_names=[str(b) for b in age_bins]))
    
    print(f"\nBMI Classification:")
    print(classification_report(final_results['bmi_labels'], final_results['bmi_preds'],
                                target_names=['Underweight', 'Normal', 'Overweight', 'Obese']))
    
    # Save results
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir = Path("../demographic_classifier/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "real_videos_metrics.json", "w") as f:
        json.dump({
            'sex_accuracy': float(final_results['sex_accuracy']),
            'age_accuracy': float(final_results['age_accuracy']),
            'bmi_accuracy': float(final_results['bmi_accuracy']),
            'overall_accuracy': float(np.mean([
                final_results['sex_accuracy'],
                final_results['age_accuracy'],
                final_results['bmi_accuracy']
            ]))
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir / 'real_videos_metrics.json'}")
    print("\n🎉 Training completed!\n")


if __name__ == "__main__":
    main()
