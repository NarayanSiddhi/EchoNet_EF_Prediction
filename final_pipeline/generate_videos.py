"""
Generate synthetic videos using trained C3D-GAN
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm


class Generator(nn.Module):
    """Same architecture as training"""
    def __init__(self, z_dim=128, cond_dim=11, size=64):
        super().__init__()
        self.size = size
        
        self.fc = nn.Linear(z_dim + cond_dim, 512 * 4 * 4 * 4)
        
        layers = []
        
        layers.extend([
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True)
        ])
        
        layers.extend([
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True)
        ])
        
        layers.extend([
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True)
        ])
        
        if size == 64:
            layers.extend([
                nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(True),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ])
        elif size == 128:
            layers.extend([
                nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(True)
            ])
            layers.extend([
                nn.ConvTranspose3d(32, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(16),
                nn.ReLU(True),
                nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ])
        else:
            layers.extend([
                nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ])
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4, 4)
        return self.main(x)


def create_condition_vector(sex, age, bmi):
    """
    Create condition vector from categorical values
    sex: 0=F, 1=M
    age: 0=0-1, 1=2-5, 2=6-10, 3=11-15, 4=16-18
    bmi: 0=underweight, 1=normal, 2=overweight, 3=obese
    """
    cond = torch.zeros(11)
    
    # Sex one-hot (2 dims)
    cond[sex] = 1
    
    # Age one-hot (5 dims)
    cond[2 + age] = 1
    
    # BMI one-hot (4 dims)
    cond[7 + bmi] = 1
    
    return cond


def save_video(tensor, path, fps=30):
    """Save video tensor - saves as numpy array (more reliable)"""
    # tensor: [1, 1, T, H, W]
    video = tensor.squeeze().detach().cpu().numpy()  # [T, H, W]
    
    # Denormalize [-1, 1] -> [0, 255]
    video = (video + 1.0) * 127.5
    video = video.clip(0, 255).astype(np.uint8)
    
    # Save as numpy array (works everywhere!)
    np.save(path.replace('.mp4', '.npy'), video)
    
    # Also try to save as MP4 using a different method
    try:
        import imageio
        # Convert to list of frames
        frames = [video[t] for t in range(video.shape[0])]
        # Save with imageio (more reliable than OpenCV on Mac)
        imageio.mimsave(path, frames, fps=fps, codec='libx264', pixelformat='gray')
    except:
        # If imageio fails, just keep the numpy file
        pass


def generate_samples(checkpoint, num_samples, output_dir, device, z_dim=128, size=64):
    """Generate synthetic videos"""
    
    # Load generator
    netG = Generator(z_dim=z_dim, cond_dim=11, size=size).to(device)
    netG.load_state_dict(torch.load(checkpoint, map_location=device))
    netG.eval()
    
    print(f"✓ Loaded generator from: {checkpoint}")
    print(f"✓ Generating {num_samples} videos...")
    print(f"✓ Resolution: {size}x{size}x32")
    print(f"✓ Output directory: {output_dir}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Label mappings
    sex_labels = {0: 'F', 1: 'M'}
    age_labels = {0: '0-1y', 1: '2-5y', 2: '6-10y', 3: '11-15y', 4: '16-18y'}
    bmi_labels = {0: 'underweight', 1: 'normal', 2: 'overweight', 3: 'obese'}
    
    # All possible combinations
    combinations = []
    for sex in [0, 1]:
        for age in [0, 1, 2, 3, 4]:
            for bmi in [0, 1, 2, 3]:
                combinations.append((sex, age, bmi))
    
    print(f"✓ Conditioning combinations: {len(combinations)} (Sex × Age × BMI)")
    print()
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Generating"):
            # Cycle through combinations
            sex, age, bmi = combinations[idx % len(combinations)]
            
            # Create condition vector
            cond = create_condition_vector(sex, age, bmi).unsqueeze(0).to(device)
            
            # Random noise
            noise = torch.randn(1, z_dim, device=device)
            
            # Generate
            fake_video = netG(noise, cond)
            
            # Create filename
            sex_label = sex_labels[sex]
            age_label = age_labels[age]
            bmi_label = bmi_labels[bmi]
            
            filename = f"synth_{idx:04d}_sex{sex_label}_age{age_label}_bmi{bmi_label}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            # Save
            save_video(fake_video, filepath)
    
    print(f"\n✅ Generated {num_samples} videos in: {output_dir}")
    print(f"\nCondition mappings:")
    print(f"  Sex: {sex_labels}")
    print(f"  Age: {age_labels}")
    print(f"  BMI: {bmi_labels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos using C3D-GAN")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Generator checkpoint")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of videos to generate")
    parser.add_argument("--output_dir", type=str, default="generated_videos", help="Output directory")
    parser.add_argument("--z_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--size", type=int, default=64, choices=[32, 64, 128], help="Video resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"✓ Using device: {device}\n")
    
    generate_samples(
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=device,
        z_dim=args.z_dim,
        size=args.size
    )

