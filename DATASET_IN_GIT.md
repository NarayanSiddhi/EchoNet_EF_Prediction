# Dataset Files in Git Repository

## ✅ What's Committed to Git

### 1. Dataset Manifest
- **File**: `data/processed_full/manifest_full.csv` (2.4MB)
- **Contains**: Complete metadata for 7,791 preprocessed videos
- **Includes**:
  - Video paths and metadata
  - Patient demographics (sex, age, weight, height)
  - Processing information (resolution, fps)
  - All `processed_path` references

### 2. Training and Generation Code
- **Files**:
  - `final_pipeline/train_c3dgan.py` - Training script
  - `final_pipeline/generate_videos.py` - Generation script
  - `checkpoints_c3dgan/generator_best.pt` - Best trained model (60MB)

### 3. Documentation
- `final_pipeline/README.md` - Complete setup guide
- `final_pipeline/SETUP_GUIDE.md` - Quick setup instructions

## ❌ What's NOT in Git (Gitignored)

### Processed Video Files
- **Location**: `data/processed/videos/*.mp4`
- **Size**: 556MB (7,791 videos)
- **Reason**: Too large for git, stored locally only
- **Status**: Gitignored per `.gitignore`

## 📋 How to Get the Full Dataset

### Option 1: Preprocess from Raw Dataset (Recommended)

1. **Download raw EchoNet Pediatric dataset**
   - See `DOWNLOAD_INSTRUCTIONS.md` for details
   - Dataset hosted on Azure Blob Storage

2. **Run preprocessing**:
   ```bash
   python preprocessing/preprocess.py --config preprocessing/config.yaml
   ```
   
   This will:
   - Process all videos to 128x128, grayscale, 30fps
   - Save to `data/processed/videos/`
   - Generate `data/processed_full/manifest_full.csv` (already in git)

### Option 2: Use Existing Manifest

If you have the processed videos elsewhere:
1. Place videos in `data/processed/videos/`
2. Use the manifest from git: `data/processed_full/manifest_full.csv`
3. Verify paths match between manifest and video files

## 📊 Dataset Statistics

- **Total videos**: 7,791
- **Manifest size**: 2.4MB (in git)
- **Video files size**: 556MB (not in git)
- **Resolution**: 128x128
- **Format**: MP4, grayscale, 30fps
- **Average per video**: ~73KB

## 🔍 Manifest File Format

The manifest CSV contains:
- `processed_path`: Path to processed video file
- `view`: A4C or PSAX
- `sex`: F, M, or O
- `age_bin`: 0-1, 2-5, 6-10, 11-15, 16-18
- `weight`: Weight in kg
- `height`: Height in cm
- `processed_width`: 128
- `processed_height`: 128
- `processed_fps`: 30

## ✅ Ready to Push

**Commits ready:**
1. `22beca5` - Training and generation scripts + checkpoint
2. `b463e85` - Setup documentation
3. `344973f` - Full dataset manifest

**To push:**
```bash
git push origin main
```

**Note**: Requires authentication. The manifest (2.4MB) and code are ready. Video files (556MB) remain local and gitignored.
