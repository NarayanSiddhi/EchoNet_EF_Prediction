# Quick Setup Guide for Full Dataset

## Step-by-Step Instructions

### 1. Download and Prepare Dataset

**Option A: If you already have processed videos**

1. Place your processed MP4 videos in: `data/processed/videos/`
2. Create a manifest CSV file (see format below)

**Option B: Preprocess from raw dataset**

1. Download EchoNet Pediatric dataset
2. Run preprocessing:
   ```bash
   python preprocessing/preprocess.py --config preprocessing/config.yaml
   ```
   This creates:
   - Processed videos in `data/processed/videos/`
   - Manifest file at `data/processed_full/manifest_full.csv`

### 2. Manifest File Format

Create a CSV file with these columns:

**Required columns:**
- `processed_path`: Path to your processed video file (relative or absolute)
- `sex`: F, M, or O
- `age_bin`: 0-1, 2-5, 6-10, 11-15, or 16-18
- `weight`: Weight in kg (for BMI calculation)
- `height`: Height in cm (for BMI calculation)

**Example manifest (`my_manifest.csv`):**
```csv
view,file_name,sex,age,weight,height,age_bin,processed_path
PSAX,video1.avi,F,1,7.8,72.0,0-1,data/processed/videos/video1.mp4
A4C,video2.avi,M,3,12.5,95.0,2-5,data/processed/videos/video2.mp4
```

**Important:** 
- The `processed_path` must point to existing MP4 files
- Videos should be 128x128, grayscale, 30fps
- Paths can be relative to project root or absolute

### 3. Run Training

**Basic command (uses default manifest):**
```bash
python final_pipeline/train_c3dgan.py
```

**With custom manifest:**
```bash
python final_pipeline/train_c3dgan.py --manifest path/to/your/manifest.csv
```

**Full example:**
```bash
python final_pipeline/train_c3dgan.py \
    --manifest data/processed_full/manifest_full.csv \
    --size 128 \
    --epochs 200 \
    --batch_size 8 \
    --checkpoint_dir checkpoints_c3dgan
```

### 4. Where to Put Dataset Details

**The dataset location is specified in the manifest file:**

1. **Manifest file location**: Set via `--manifest` argument
   - Default: `data/processed_full/manifest_full.csv`
   - You can use any path: `--manifest /path/to/your/manifest.csv`

2. **Video files location**: Specified in the `processed_path` column of manifest
   - Each row has a `processed_path` pointing to the video file
   - Can be relative: `data/processed/videos/video.mp4`
   - Or absolute: `/full/path/to/video.mp4`

**Example directory structure:**
```
project/
├── data/
│   ├── processed/
│   │   └── videos/           # Your processed videos go here
│   │       ├── video1.mp4
│   │       └── video2.mp4
│   └── processed_full/
│       └── manifest_full.csv  # Manifest file (default location)
└── final_pipeline/
    └── train_c3dgan.py
```

**In the manifest CSV, the `processed_path` column should look like:**
```csv
processed_path
data/processed/videos/video1.mp4
data/processed/videos/video2.mp4
```

### 5. Verify Setup

Before training, verify:
```bash
# Check manifest exists
ls -lh data/processed_full/manifest_full.csv

# Check a few video files exist
ls -lh data/processed/videos/*.mp4 | head -5

# Verify manifest format
head -2 data/processed_full/manifest_full.csv
```

### 6. Common Issues

**"Video not found" error:**
- Check `processed_path` in manifest points to real files
- Verify paths are correct (relative vs absolute)
- Ensure videos are in MP4 format

**"Cannot open video" error:**
- Videos might be corrupted
- Check video format (should be MP4)
- Verify OpenCV can read: `python -c "import cv2; cap = cv2.VideoCapture('path/to/video.mp4'); print(cap.isOpened())"`

**Out of memory:**
- Reduce batch size: `--batch_size 4` or `--batch_size 2`
- Use smaller resolution: `--size 64`

## Summary

**To use the full dataset:**
1. Put processed videos in `data/processed/videos/`
2. Create manifest CSV with `processed_path` column pointing to videos
3. Run: `python final_pipeline/train_c3dgan.py --manifest your_manifest.csv`

**The key is the `processed_path` column in your manifest CSV - it tells the script where to find each video file!**
