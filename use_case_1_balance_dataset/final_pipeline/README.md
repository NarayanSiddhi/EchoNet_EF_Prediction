# Final Pipeline - Training and Generation Guide

This guide explains how to set up and run the C3D-GAN training and video generation scripts.

## 📋 Prerequisites

1. **Dataset**: You need the EchoNet Pediatric dataset downloaded
2. **Processed Videos**: Videos should be preprocessed to 128x128 resolution, grayscale, 30fps
3. **Manifest File**: A CSV file listing all videos with their metadata

## 📁 Dataset Setup

### Option 1: Use Pre-processed Full Dataset (Recommended)

If you have the full dataset already processed:

1. **Place your processed videos** in: `data/processed/videos/`
   - Videos should be MP4 files, 128x128, grayscale, 30fps
   - Example: `data/processed/videos/CR32a95e8-CR32a9abd-000032.mp4`

2. **Create or use manifest file** at: `data/processed_full/manifest_full.csv`

   The manifest CSV must have these columns:
   - `view`: View type (A4C or PSAX)
   - `file_name`: Original filename
   - `file_path`: Path to original video
   - `sex`: Sex (F, M, or O)
   - `age`: Age in years (numeric)
   - `age_bin`: Age bin (0-1, 2-5, 6-10, 11-15, 16-18)
   - `weight`: Weight in kg (numeric)
   - `height`: Height in cm (numeric)
   - `processed_path`: **Path to processed MP4 file** (relative to project root)
   - Other columns are optional

   **Example manifest row:**
   ```csv
   view,file_name,file_path,ef,sex,age,weight,height,split,age_bin,processed_path
   PSAX,CR32a95e8-CR32a9abd-000032.avi,/path/to/original.avi,43.04,F,1,7.8,72.0,7,0-1,data/processed/videos/CR32a95e8-CR32a9abd-000032.mp4
   ```

### Option 2: Preprocess Your Own Dataset

If you need to preprocess the dataset:

1. **Download the EchoNet Pediatric dataset** (see main README.md)

2. **Run preprocessing** to create processed videos and manifest:
   ```bash
   python preprocessing/preprocess.py --config preprocessing/config.yaml
   ```

   This will:
   - Process videos to 128x128, grayscale, 30fps
   - Save processed videos to `data/processed/videos/`
   - Generate manifest at `data/processed_full/manifest_full.csv`

3. **Update `preprocessing/config.yaml`** if needed:
   ```yaml
   paths:
     dataset_root: "Dataset/EchoNet-Pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi"
     output_dir: "data/processed_full"
     manifest_filename: "manifest_full.csv"
   
   processing:
     output_videos_dir: "data/processed/videos"
     width: 128
     height: 128
     target_fps: 30
     max_frames: 96
     grayscale: true
   ```

## 🚀 Running Training

### Basic Usage

```bash
python final_pipeline/train_c3dgan.py
```

This uses default settings:
- **Manifest**: `data/processed_full/manifest_full.csv`
- **Resolution**: 128x128
- **Epochs**: 200
- **Batch size**: 8

### Custom Configuration

```bash
python final_pipeline/train_c3dgan.py \
    --manifest data/processed_full/manifest_full.csv \
    --size 128 \
    --epochs 200 \
    --batch_size 8 \
    --checkpoint_dir checkpoints_c3dgan
```

### All Parameters

```bash
python final_pipeline/train_c3dgan.py \
    --manifest <path_to_manifest.csv> \      # Path to your manifest file
    --size 128 \                              # Resolution: 32, 64, or 128
    --epochs 200 \                            # Number of training epochs
    --batch_size 8 \                          # Batch size (adjust based on GPU memory)
    --lr_g 0.0002 \                           # Generator learning rate
    --lr_d 0.0002 \                           # Discriminator learning rate
    --z_dim 128 \                             # Latent dimension
    --cond_dim 11 \                           # Condition dimension (don't change)
    --checkpoint_dir checkpoints_c3dgan       # Where to save checkpoints
```

## 📝 Manifest File Format

Your manifest CSV file must have these **required columns**:

| Column | Description | Example |
|--------|-------------|---------|
| `processed_path` | Path to processed video (MP4) | `data/processed/videos/video.mp4` |
| `sex` | Sex: F, M, or O | `F` |
| `age_bin` | Age bin: 0-1, 2-5, 6-10, 11-15, 16-18 | `0-1` |
| `weight` | Weight in kg (for BMI calculation) | `7.8` |
| `height` | Height in cm (for BMI calculation) | `72.0` |

**Optional columns** (will be computed if missing):
- `bmi_category`: underweight, normal, overweight, obese (computed from weight/height)

**Example manifest structure:**
```csv
view,file_name,file_path,ef,sex,age,weight,height,split,age_bin,processed_path
PSAX,CR32a95e8-CR32a9abd-000032.avi,/path/to/original.avi,43.04,F,1,7.8,72.0,7,0-1,data/processed/videos/CR32a95e8-CR32a9abd-000032.mp4
A4C,CR32a95c7-CR32a9aa4-000028.avi,/path/to/original.avi,47.86,F,1,8.5,73.0,9,0-1,data/processed/videos/CR32a95c7-CR32a9aa4-000028.mp4
```

## 🎬 Running Video Generation

After training, generate videos using:

```bash
python final_pipeline/generate_videos.py \
    --checkpoint checkpoints_c3dgan/generator_best.pt \
    --num_samples 100 \
    --output_dir generated_videos \
    --size 128
```

This generates videos with the naming pattern:
```
synth_0000_sexF_age0-1y_bmiunderweight.mp4
synth_0001_sexM_age2-5y_bminormal.mp4
...
```

## 🔍 Troubleshooting

### "Video not found" errors

- Check that `processed_path` in manifest points to existing files
- Paths can be relative (to project root) or absolute
- Ensure videos are MP4 format, 128x128, grayscale

### "Cannot open video" errors

- Verify video files are not corrupted
- Check that OpenCV can read the files: `cv2.VideoCapture(path)`
- Ensure videos are in MP4 format

### Out of memory errors

- Reduce `--batch_size` (try 4 or 2)
- Use smaller resolution `--size 64` instead of 128

### Missing columns in manifest

- The script will compute `bmi_category` automatically if `weight` and `height` are present
- If `weight`/`height` are missing, BMI defaults to "normal"

## 📊 Expected Dataset Structure

```
EchoNet-Pediatric-BIGAN-AUGMENTATION/
├── data/
│   ├── processed/
│   │   └── videos/              # Processed MP4 videos (128x128, grayscale)
│   │       ├── video1.mp4
│   │       ├── video2.mp4
│   │       └── ...
│   └── processed_full/
│       └── manifest_full.csv     # Manifest file with all video metadata
├── final_pipeline/
│   ├── train_c3dgan.py          # Training script
│   ├── generate_videos.py       # Generation script
│   └── README.md                # This file
└── checkpoints_c3dgan/          # Saved model checkpoints
    └── generator_best.pt
```

## 💡 Quick Start Checklist

- [ ] Dataset downloaded and placed in correct location
- [ ] Videos preprocessed to 128x128, grayscale, 30fps
- [ ] Manifest CSV created with `processed_path` column pointing to video files
- [ ] Manifest has required columns: `sex`, `age_bin`, `weight`, `height`
- [ ] Run training: `python final_pipeline/train_c3dgan.py`
- [ ] Checkpoints saved in `checkpoints_c3dgan/`
- [ ] Generate videos: `python final_pipeline/generate_videos.py`

## 📞 Need Help?

If you encounter issues:
1. Check that all video paths in manifest exist
2. Verify video format (MP4, 128x128, grayscale)
3. Check GPU memory (reduce batch_size if needed)
4. Ensure all required columns are in manifest CSV
