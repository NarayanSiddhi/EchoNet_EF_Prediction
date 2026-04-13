"""
Data loader for preprocessed echocardiogram videos with class labels.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Tuple, Dict


class ConditionalEchoVideoDataset(Dataset):
    """
    Dataset loader for preprocessed echocardiogram videos with class labels.
    Supports filtering by underrepresented groups.
    """
    
    def __init__(
        self,
        manifest_path: str,
        video_dir: str,
        video_length: int = 96,
        video_size: int = 128,
        class_to_idx: Optional[Dict[str, int]] = None,
        filter_groups: Optional[list] = None,
        min_samples: int = 10
    ):
        """
        Args:
            manifest_path: Path to manifest CSV file
            video_dir: Directory containing processed videos
            video_length: Number of frames to extract
            video_size: Spatial size (height/width)
            class_to_idx: Mapping from class_label to index
            filter_groups: List of class_labels to include (None = all)
            min_samples: Minimum samples per group to include
        """
        self.manifest = pd.read_csv(manifest_path)
        self.video_dir = Path(video_dir)
        self.video_length = video_length
        self.video_size = video_size
        
        # Create class labels if not already present
        if 'class_label' not in self.manifest.columns:
            if 'bmi_bin' in self.manifest.columns:
                self.manifest['class_label'] = (
                    self.manifest['view'].astype(str) + '_' +
                    self.manifest['sex'].astype(str) + '_' +
                    self.manifest['age_bin'].astype(str) + '_' +
                    self.manifest['bmi_bin'].astype(str)
                )
            elif 'view' in self.manifest.columns and 'sex' in self.manifest.columns and 'age_bin' in self.manifest.columns:
                self.manifest['class_label'] = (
                    self.manifest['view'].astype(str) + '_' + 
                    self.manifest['sex'].astype(str) + '_' + 
                    self.manifest['age_bin'].astype(str)
                )
            else:
                raise ValueError("Cannot create class_label: missing required columns (view, sex, age_bin) or class_label")
        
        # Resolve video path (prefer processed_path if available and exists)
        if 'resolved_path' not in self.manifest.columns:
            def resolve_path(row):
                processed_path = row.get('processed_path', None)
                if pd.notna(processed_path) and processed_path and os.path.exists(processed_path):
                    return processed_path
                file_path = row.get('file_path', None)
                if pd.notna(file_path) and file_path and os.path.exists(file_path):
                    return file_path
                return None

            self.manifest['resolved_path'] = self.manifest.apply(resolve_path, axis=1)

        # Filter to videos that exist
        self.manifest = self.manifest[
            self.manifest['resolved_path'].apply(lambda x: x is not None)
        ].reset_index(drop=True)
        
        # Filter by groups if specified
        if filter_groups is not None:
            self.manifest = self.manifest[
                self.manifest['class_label'].isin(filter_groups)
            ].reset_index(drop=True)
        
        # Filter groups with minimum samples
        if min_samples > 0:
            group_counts = self.manifest.groupby('class_label').size()
            valid_groups = group_counts[group_counts >= min_samples].index
            self.manifest = self.manifest[
                self.manifest['class_label'].isin(valid_groups)
            ].reset_index(drop=True)
        
        # Create or use class mapping
        if class_to_idx is None:
            unique_classes = sorted(self.manifest['class_label'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
        
        print(f"Loaded {len(self.manifest)} videos")
        print(f"Number of classes: {len(self.class_to_idx)}")
        # Avoid printing pandas Series directly to prevent NumPy recursion errors
        try:
            class_counts = self.manifest['class_label'].value_counts().head(10)
            print(f"Class distribution (top 10):")
            for cls, count in class_counts.items():
                print(f"  {cls}: {int(count)}")
        except Exception:
            print(f"Class distribution: {len(self.manifest['class_label'].unique())} unique classes")
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a video with its class label.
        Returns:
            video: Tensor of shape (C, T, H, W) where C=1 (grayscale), T=video_length
            label: Class index
        """
        row = self.manifest.iloc[idx]
        class_label = row['class_label']
        
        # Use resolved path (processed_path preferred when available)
        if 'resolved_path' not in row.index:
            raise KeyError(f"'resolved_path' column not found in manifest")
        
        video_path = row['resolved_path']
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Load video (fail-safe for corrupted files)
        try:
            video = self._load_video(video_path)
        except Exception as e:
            print(f"Warning: failed to load video {video_path}: {e}")
            video = np.zeros((self.video_length, self.video_size, self.video_size), dtype=np.float32)
        
        # Ensure it's a numpy array with proper dtype
        video = np.asarray(video, dtype=np.float32)
        
        # Normalize to [0, 1] if not already
        if video.max() > 1.0:
            video = video / 255.0
        
        # Convert to tensor: (T, H, W) -> (C, T, H, W)
        video = torch.tensor(video, dtype=torch.float32)
        if video.dim() == 3:
            video = video.unsqueeze(0)  # Add channel dimension
        
        # Get class index
        label = self.class_to_idx[class_label]
        
        return video, label
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """
        Load video frames from file.
        Returns:
            frames: Array of shape (T, H, W) where T <= video_length
        """
        cap = cv2.VideoCapture(video_path)
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
            
            # Resize if needed
            if frame.shape[0] != self.video_size or frame.shape[1] != self.video_size:
                frame = cv2.resize(frame, (self.video_size, self.video_size), interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
        
        cap.release()
        
        # Pad or truncate to exact length
        if len(frames) < self.video_length:
            # Repeat last frame
            last_frame = frames[-1] if frames else np.zeros((self.video_size, self.video_size), dtype=np.uint8)
            frames.extend([last_frame] * (self.video_length - len(frames)))
        elif len(frames) > self.video_length:
            # Uniformly sample frames
            indices = np.linspace(0, len(frames) - 1, self.video_length, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Stack frames: (T, H, W)
        video = np.stack(frames, axis=0).astype(np.float32)
        
        return video


def create_conditional_dataloader(
    manifest_path: str,
    video_dir: str,
    batch_size: int = 8,
    video_length: int = 96,
    video_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    filter_groups: Optional[list] = None,
    min_samples: int = 10,
    class_to_idx: Optional[Dict[str, int]] = None,
    pin_memory: bool = True,
    balanced_sampling: bool = False,
    prefetch_factor: Optional[int] = None
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Create a DataLoader for training with class labels.
    Returns:
        dataloader: DataLoader instance
        class_to_idx: Class label to index mapping
    """
    dataset = ConditionalEchoVideoDataset(
        manifest_path=manifest_path,
        video_dir=video_dir,
        video_length=video_length,
        video_size=video_size,
        class_to_idx=class_to_idx,
        filter_groups=filter_groups,
        min_samples=min_samples
    )
    
    sampler = None
    if balanced_sampling:
        class_counts = dataset.manifest['class_label'].value_counts().to_dict()
        sample_weights = dataset.manifest['class_label'].map(lambda c: 1.0 / class_counts[c]).astype(np.float32).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory if torch.cuda.is_available() else False,
        "persistent_workers": False,  # Disabled to prevent hangs after resume
    }
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(**dataloader_kwargs)
    
    return dataloader, dataset.class_to_idx
