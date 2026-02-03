"""
Conditional 3D GAN for Pediatric Echocardiogram Video Generation
Best approach: Simple, stable, and proven to work!

Conditions: Sex, Age, BMI
Resolution: 64x64 or 128x128 (configurable)
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EchoDataset(Dataset):
    """Dataset for preprocessed videos (MP4 or numpy)"""
    def __init__(self, manifest_csv: str, video_length=32, video_size=128):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['processed_path'].notna()].reset_index(drop=True)
        self.video_length = video_length
        self.video_size = video_size
        
        # Condition encodings
        self.sex_map = {'F': 0, 'M': 1, 'O': 0}
        self.age_map = {'0-1': 0, '2-5': 1, '6-10': 2, '11-15': 3, '16-18': 4}
        self.bmi_map = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
        
        # Compute BMI categories if not present
        if 'bmi_category' not in self.df.columns:
            self._compute_bmi_categories()
        
        print(f"✓ Dataset: {len(self.df)} videos loaded")

    def _compute_bmi_categories(self):
        """Compute BMI categories from weight and height"""
        def categorize_bmi(row):
            if pd.isna(row.get('weight')) or pd.isna(row.get('height')) or row['height'] <= 0:
                return 'normal'  # Default
            
            # BMI = weight (kg) / (height (cm) / 100)^2
            bmi = row['weight'] / ((row['height'] / 100.0) ** 2)
            
            # Pediatric BMI categories (approximate)
            if bmi < 18.5:
                return 'underweight'
            elif bmi < 25:
                return 'normal'
            elif bmi < 30:
                return 'overweight'
            else:
                return 'obese'
        
        self.df['bmi_category'] = self.df.apply(categorize_bmi, axis=1)

    def _load_video(self, video_path):
        """Load video from MP4 or numpy file"""
        video_path = Path(video_path)
        
        # Handle relative paths
        if not video_path.is_absolute():
            # Try relative to current directory
            base_path = Path(__file__).parent.parent
            video_path = base_path / video_path
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Check if it's a numpy file
        if video_path.suffix == '.npy':
            video = np.load(video_path)
        else:
            # Load MP4 using cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frames = []
            while len(frames) < self.video_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Resize to target size
                if frame.shape[0] != self.video_size or frame.shape[1] != self.video_size:
                    frame = cv2.resize(frame, (self.video_size, self.video_size))
                
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                # Return zeros if video couldn't be loaded
                video = np.zeros((self.video_length, self.video_size, self.video_size), dtype=np.uint8)
            else:
                video = np.array(frames)
                # Pad or truncate to exact length
                if len(video) < self.video_length:
                    padding = np.zeros((self.video_length - len(video), self.video_size, self.video_size), dtype=video.dtype)
                    video = np.concatenate([video, padding], axis=0)
                elif len(video) > self.video_length:
                    video = video[:self.video_length]
        
        return video

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load video [T, H, W]
        try:
            video = self._load_video(row['processed_path'])
        except Exception as e:
            print(f"Warning: Failed to load {row['processed_path']}: {e}")
            video = np.zeros((self.video_length, self.video_size, self.video_size), dtype=np.uint8)
        
        # Normalize to [-1, 1]
        video = video.astype(np.float32) / 127.5 - 1.0
        
        # To tensor [C, T, H, W]
        x = torch.from_numpy(video).unsqueeze(0)
        
        # Conditions - handle both 'Sex' and 'sex' column names
        sex_col = 'Sex' if 'Sex' in row else 'sex'
        sex = self.sex_map.get(row[sex_col], 0)
        age = self.age_map.get(row['age_bin'], 0)
        bmi = self.bmi_map.get(row.get('bmi_category', 'normal'), 1)  # Default to normal
        
        # One-hot encode for better conditioning
        sex_onehot = torch.zeros(2)
        sex_onehot[sex] = 1
        
        age_onehot = torch.zeros(5)
        age_onehot[age] = 1
        
        bmi_onehot = torch.zeros(4)
        bmi_onehot[bmi] = 1
        
        # Concatenate: 2 + 5 + 4 = 11 dimensions
        cond = torch.cat([sex_onehot, age_onehot, bmi_onehot])
        
        return x, cond


class Generator(nn.Module):
    """
    3D Generator with conditioning
    Generates: [B, 1, 32, H, W] videos
    """
    def __init__(self, z_dim=128, cond_dim=11, size=64):
        super().__init__()
        self.size = size
        
        # Project noise + condition
        self.fc = nn.Linear(z_dim + cond_dim, 512 * 4 * 4 * 4)
        
        # 3D Convolution layers
        layers = []
        
        # 4 -> 8
        layers.extend([
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True)
        ])
        
        # 8 -> 16
        layers.extend([
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True)
        ])
        
        # 16 -> 32
        layers.extend([
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True)
        ])
        
        if size == 64:
            # 32 -> 64 (spatial only, keep temporal at 32)
            layers.extend([
                nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(True),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ])
        elif size == 128:
            # 32 -> 64
            layers.extend([
                nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(True)
            ])
            # 64 -> 128
            layers.extend([
                nn.ConvTranspose3d(32, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(True),
                nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ])
        else:  # size == 32
            layers.extend([
                nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ])
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, z, cond):
        # Concatenate noise and condition
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4, 4)
        return self.main(x)


class Discriminator(nn.Module):
    """
    3D Discriminator with conditioning
    Input: [B, 1, 32, H, W] videos + conditions
    Output: [B, 1] real/fake score
    """
    def __init__(self, cond_dim=11, size=64):
        super().__init__()
        self.size = size
        
        layers = []
        
        # Input layer
        if size == 128:
            # 128 -> 64
            layers.extend([
                nn.Conv3d(1, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            # 64 -> 32
            layers.extend([
                nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        elif size == 64:
            # 64 -> 32
            layers.extend([
                nn.Conv3d(1, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        else:  # size == 32
            layers.extend([
                nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # 32 -> 16
        layers.extend([
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 16 -> 8
        layers.extend([
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 8 -> 4
        layers.extend([
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        self.features = nn.Sequential(*layers)
        
        # Classifier with conditioning
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4 + cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, cond):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        # Concatenate features with condition
        x = torch.cat([features, cond], dim=1)
        return self.classifier(x)


def weights_init(m):
    """Initialize weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(cfg):
    # Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    
    # Dataset
    dataset = EchoDataset(cfg['manifest'], video_length=32, video_size=cfg['size'])
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=(device == "cuda")
    )
    print(f"✓ Batches per epoch: {len(loader)}")
    
    # Models
    netG = Generator(cfg['z_dim'], cfg['cond_dim'], cfg['size']).to(device)
    netD = Discriminator(cfg['cond_dim'], cfg['size']).to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    print(f"✓ Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"✓ Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=cfg['lr_d'], betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg['lr_g'], betas=(0.5, 0.999))
    
    # Labels
    real_label = 1.0
    fake_label = 0.0
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Resolution: {cfg['size']}x{cfg['size']}x32")
    print(f"  LR (G): {cfg['lr_g']}, LR (D): {cfg['lr_d']}")
    print(f"  Conditioning: Sex (2) + Age (5) + BMI (4) = 11 dims")
    print(f"  Latent dim: {cfg['z_dim']}")
    print(f"{'='*60}\n")
    
    # Training loop
    G_losses = []
    D_losses = []
    
    for epoch in range(cfg['epochs']):
        netG.train()
        netD.train()
        
        epoch_G_loss = []
        epoch_D_loss = []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        for i, (real_videos, conditions) in enumerate(pbar):
            real_videos = real_videos.to(device)
            conditions = conditions.to(device)
            b_size = real_videos.size(0)
            
            # ========== Train Discriminator ==========
            netD.zero_grad()
            
            # Real videos
            label = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)
            output = netD(real_videos, conditions)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = torch.sigmoid(output).mean().item()
            
            # Fake videos
            noise = torch.randn(b_size, cfg['z_dim'], device=device)
            fake = netG(noise, conditions)
            label.fill_(fake_label)
            output = netD(fake.detach(), conditions)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = torch.sigmoid(output).mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # ========== Train Generator ==========
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake, conditions)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = torch.sigmoid(output).mean().item()
            optimizerG.step()
            
            # Save losses
            epoch_D_loss.append(errD.item())
            epoch_G_loss.append(errG.item())
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{errD.item():.3f}',
                'G_loss': f'{errG.item():.3f}',
                'D(x)': f'{D_x:.3f}',
                'D(G(z))': f'{D_G_z2:.3f}'
            })
        
        # Epoch statistics
        avg_D_loss = np.mean(epoch_D_loss)
        avg_G_loss = np.mean(epoch_G_loss)
        
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)
        
        print(f"\n[Epoch {epoch+1}/{cfg['epochs']}] D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f}")
        
        # Save checkpoints
        os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
        torch.save(netG.state_dict(), f"{cfg['checkpoint_dir']}/generator_epoch{epoch}.pt")
        torch.save(netD.state_dict(), f"{cfg['checkpoint_dir']}/discriminator_epoch{epoch}.pt")
        
        # Save best model
        if epoch == 0 or avg_G_loss < min(G_losses[:-1]):
            torch.save(netG.state_dict(), f"{cfg['checkpoint_dir']}/generator_best.pt")
            print(f"✓ Saved best generator (epoch {epoch+1})")
        
        print()
    
    print("="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best G_loss: {min(G_losses):.4f}")
    print(f"Final G_loss: {G_losses[-1]:.4f}")
    print(f"Final D_loss: {D_losses[-1]:.4f}")
    print(f"\nCheckpoints saved in: {cfg['checkpoint_dir']}/")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train C3D-GAN for video generation")
    
    # Data
    parser.add_argument("--manifest", type=str, default="data/processed_full/manifest_full.csv")
    parser.add_argument("--size", type=int, default=128, choices=[32, 64, 128])
    
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    
    # Model
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=11)  # 2 + 5 + 4
    
    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_c3dgan")
    
    args = parser.parse_args()
    train(vars(args))

