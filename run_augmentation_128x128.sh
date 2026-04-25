#!/usr/bin/env bash
# Full 128×128 augmentation pipeline
#
# Launch in byobu:
#   byobu new-session -s aug128 -d \; send-keys "cd /data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION && bash run_augmentation_128x128.sh" Enter
#
set -euo pipefail

ROOT="/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION"
cd "$ROOT"

CKPT="use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt"
MANIFEST="data/processed_full/train_manifest_filtered_clean.csv"
UC2_OUT="use_case_2_demographic_variations/demographic_variations_128x128"
UC3_OUT="perfect_synthetic_copies_128x128"
LOG="$ROOT/augmentation_128x128_pipeline.log"

exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -Is)] $*"; }

log "========================================================"
log "128×128 AUGMENTATION PIPELINE START"
log "Checkpoint : $CKPT"
log "Manifest   : $MANIFEST"
log "UC2 output : $UC2_OUT"
log "UC3 output : $UC3_OUT"
log "========================================================"

# ── STEP 1: UC2 demographic variations at 128×128 ─────────────────────────
log ""
log "===== STEP 1: Generate UC2 demographic variations at 128×128 ====="
python use_case_2_demographic_variations/generate_demographic_variations_fixed.py \
    --manifest "$MANIFEST" \
    --checkpoint "$CKPT" \
    --output_dir "$UC2_OUT" \
    --video_size 128 \
    --video_length 32 \
    --device cuda
log "STEP 1 done"

# ── STEP 2: Perfect copies (UC3 inference) at 128×128 ─────────────────────
log ""
log "===== STEP 2: Generate perfect synthetic copies at 128×128 ====="
mkdir -p "$UC3_OUT"
python - <<'PY'
import sys, torch, numpy as np, cv2, pandas as pd, imageio, json
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, "use_case_3_perfect_reconstruction")
from models import PerfectReconstructionGenerator

CKPT    = "use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt"
MANIFEST= "data/processed_full/train_manifest_filtered_clean.csv"
OUTDIR  = Path("perfect_synthetic_copies_128x128")
OUTDIR.mkdir(parents=True, exist_ok=True)

VL, VS = 32, 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CKPT, map_location=device)
cond  = ckpt.get("conditioning", "concat")
vsz   = int(ckpt.get("video_size", 128))
G = PerfectReconstructionGenerator(base_channels=64, spatial_size=vsz, conditioning=cond).to(device)
G.load_state_dict(ckpt["generator"])
G.eval()
print(f"Model loaded  conditioning={cond}  spatial_size={vsz}")

df = pd.read_csv(MANIFEST)
sex_map  = {"F": 0, "M": 1, "O": 0}
age_map  = {"0-1":0,"1-2":0,"2-3":1,"2-5":1,"3-5":1,"5-8":2,"6-10":2,"8-12":2,
            "11-15":3,"12-15":3,"15-18":4,"16-18":4}
bmi_map  = {"underweight":0,"normal":1,"overweight":2,"obese":3}

def compute_bmi(w, h):
    try:
        b = float(w) / (float(h)/100)**2
        if b < 18.5: return "underweight"
        if b < 25:   return "normal"
        if b < 30:   return "overweight"
        return "obese"
    except: return "normal"

def load_vid(path):
    cap = cv2.VideoCapture(str(path)); frames = []
    while len(frames) < VL:
        ok, fr = cap.read()
        if not ok: break
        if fr.ndim == 3: fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(fr, (VS, VS)))
    cap.release()
    while len(frames) < VL: frames.append(frames[-1] if frames else np.zeros((VS,VS),np.uint8))
    v = np.array(frames[:VL], np.float32)/127.5 - 1.0
    return torch.from_numpy(v).unsqueeze(0)  # [1,T,H,W]

records = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Perfect copies"):
    try:
        vpath = Path(row["processed_path"])
        if not vpath.exists(): continue
        x = load_vid(vpath).unsqueeze(0).to(device)   # [1,1,T,H,W]
        sex_v = sex_map.get(str(row.get("sex","F")), 0)
        age_v = age_map.get(str(row.get("age_bin","11-15")), 3)
        bmi_v = bmi_map.get(row.get("bmi_category",
                    compute_bmi(row.get("weight",50), row.get("height",150))), 1)
        cond_vec = torch.zeros(11, device=device)
        cond_vec[sex_v] = 1.0
        cond_vec[2+age_v] = 1.0
        cond_vec[7+bmi_v] = 1.0
        cond_vec = cond_vec.unsqueeze(0)
        with torch.no_grad():
            out = G(x, cond_vec)
        frames_np = ((out[0,0].cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        out_name = f"perfect_copy_{idx:05d}.mp4"
        out_path = OUTDIR / out_name
        T, H, W = frames_np.shape
        rgb = np.stack([frames_np]*3, axis=-1)  # [T,H,W,3]
        imageio.mimwrite(str(out_path), rgb, fps=30, codec="libx264", quality=8)
        records.append({
            "original_id":   idx,
            "original_path": str(vpath),
            "synthetic_path":str(out_path),
            "EF": row.get("ef", row.get("EF", "")),
            "sex": row.get("sex",""), "age_bin": row.get("age_bin",""),
        })
    except Exception as e:
        pass

out_df = pd.DataFrame(records)
out_df.to_csv(OUTDIR/"perfect_copies_manifest.csv", index=False)
print(f"\nSaved {len(out_df)} perfect copies → {OUTDIR}/perfect_copies_manifest.csv")
PY
log "STEP 2 done"

# ── STEP 3: SSIM / PSNR / MSE on UC2 128×128 variations ──────────────────
log ""
log "===== STEP 3: SSIM/PSNR/MSE — UC2 variations at 128×128 ====="
python - <<'PY'
import pandas as pd, numpy as np, cv2, json
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn

MANIFEST = "use_case_2_demographic_variations/demographic_variations_128x128/variations_manifest.csv"
OUT_JSON = "uc2_128x128_metrics.json"
VL, VS = 32, 128

df = pd.read_csv(MANIFEST)
print(f"Total rows: {len(df)}")

def psnr(a, b):
    mse = np.mean((a.astype(float)-b.astype(float))**2)
    return float('inf') if mse == 0 else 20*np.log10(255.0/np.sqrt(mse))

def load(path):
    cap = cv2.VideoCapture(str(path)); frames=[]
    while len(frames)<VL:
        ok,fr=cap.read()
        if not ok: break
        if fr.ndim==3: fr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(fr,(VS,VS)))
    cap.release()
    while len(frames)<VL: frames.append(frames[-1] if frames else np.zeros((VS,VS),np.uint8))
    return np.array(frames[:VL])

ssims, psnrs, mses, errors = [], [], [], 0
by_type = {}
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        a = load(row['original_path']).astype(np.float32)/255.0
        b = load(row['synthetic_path']).astype(np.float32)/255.0
        T = min(len(a), len(b))
        ss = float(np.mean([ssim_fn(a[t],b[t],data_range=1.0) for t in range(T)]))
        ps = float(np.mean([psnr(a[t]*255,b[t]*255) for t in range(T)]))
        ms = float(np.mean((a[:T]-b[:T])**2))
        ssims.append(ss); psnrs.append(ps); mses.append(ms)
        vt = row.get('variation_type','unknown')
        by_type.setdefault(vt, {'ssims':[],'psnrs':[],'mses':[]})
        by_type[vt]['ssims'].append(ss)
        by_type[vt]['psnrs'].append(ps)
        by_type[vt]['mses'].append(ms)
    except Exception as e:
        errors += 1

psnr_fin = [p for p in psnrs if not np.isinf(p)]
print(f"\n{'='*60}")
print(f"UC2 128×128 Metrics  ({len(ssims)} videos, {errors} errors)")
print(f"{'='*60}")
print(f"  SSIM : {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
print(f"  PSNR : {np.mean(psnr_fin):.2f} ± {np.std(psnr_fin):.2f} dB")
print(f"  MSE  : {np.mean(mses):.6f} ± {np.std(mses):.6f}")
print(f"\nBy variation type:")
for vt, vals in by_type.items():
    pf = [p for p in vals['psnrs'] if not np.isinf(p)]
    print(f"  {vt}:")
    print(f"    SSIM {np.mean(vals['ssims']):.4f}  PSNR {np.mean(pf):.2f} dB  MSE {np.mean(vals['mses']):.6f}")

result = {
    "n": len(ssims), "errors": errors,
    "SSIM": float(np.mean(ssims)),  "SSIM_std": float(np.std(ssims)),
    "PSNR": float(np.mean(psnr_fin)),"PSNR_std": float(np.std(psnr_fin)),
    "MSE":  float(np.mean(mses)),   "MSE_std":  float(np.std(mses)),
    "by_type": {vt: {
        "SSIM": float(np.mean(v['ssims'])),
        "PSNR": float(np.mean([p for p in v['psnrs'] if not np.isinf(p)])),
        "MSE":  float(np.mean(v['mses']))
    } for vt, v in by_type.items()}
}
with open(OUT_JSON,"w") as f: json.dump(result,f,indent=2)
print(f"\nSaved → {OUT_JSON}")
PY
log "STEP 3 done"

# ── STEP 4: SSIM / PSNR / MSE on UC3 perfect copies ──────────────────────
log ""
log "===== STEP 4: SSIM/PSNR/MSE — Perfect copies at 128×128 ====="
python - <<'PY'
import pandas as pd, numpy as np, cv2, json
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn

MANIFEST = "perfect_synthetic_copies_128x128/perfect_copies_manifest.csv"
OUT_JSON = "uc3_128x128_metrics.json"
VL, VS = 32, 128

p = Path(MANIFEST)
if not p.exists():
    print(f"Manifest not found: {MANIFEST}"); exit(0)

df = pd.read_csv(MANIFEST)
print(f"Total rows: {len(df)}")

def psnr(a, b):
    mse = np.mean((a.astype(float)-b.astype(float))**2)
    return float('inf') if mse == 0 else 20*np.log10(255.0/np.sqrt(mse))

def load(path):
    cap = cv2.VideoCapture(str(path)); frames=[]
    while len(frames)<VL:
        ok,fr=cap.read()
        if not ok: break
        if fr.ndim==3: fr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(fr,(VS,VS)))
    cap.release()
    while len(frames)<VL: frames.append(frames[-1] if frames else np.zeros((VS,VS),np.uint8))
    return np.array(frames[:VL])

ssims, psnrs, mses, errors = [], [], [], 0
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        a = load(row['original_path']).astype(np.float32)/255.0
        b = load(row['synthetic_path']).astype(np.float32)/255.0
        T = min(len(a), len(b))
        ss = float(np.mean([ssim_fn(a[t],b[t],data_range=1.0) for t in range(T)]))
        ps = float(np.mean([psnr(a[t]*255,b[t]*255) for t in range(T)]))
        ms = float(np.mean((a[:T]-b[:T])**2))
        ssims.append(ss); psnrs.append(ps); mses.append(ms)
    except: errors += 1

psnr_fin = [p for p in psnrs if not np.isinf(p)]
print(f"\n{'='*60}")
print(f"UC3 128×128 Perfect Copies  ({len(ssims)} videos, {errors} errors)")
print(f"{'='*60}")
print(f"  SSIM : {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
print(f"  PSNR : {np.mean(psnr_fin):.2f} ± {np.std(psnr_fin):.2f} dB")
print(f"  MSE  : {np.mean(mses):.6f} ± {np.std(mses):.6f}")
result = {
    "n": len(ssims), "errors": errors,
    "SSIM": float(np.mean(ssims)),  "SSIM_std": float(np.std(ssims)),
    "PSNR": float(np.mean(psnr_fin)),"PSNR_std": float(np.std(psnr_fin)),
    "MSE":  float(np.mean(mses)),   "MSE_std":  float(np.std(mses)),
}
with open(OUT_JSON,"w") as f: json.dump(result,f,indent=2)
print(f"Saved → {OUT_JSON}")
PY
log "STEP 4 done"

# ── STEP 5: FID — real vs 128×128 demographic variations ─────────────────
log ""
log "===== STEP 5: FID — real vs 128×128 variations ====="
python - <<'PY'
import pandas as pd
from pathlib import Path
mpath = "use_case_2_demographic_variations/demographic_variations_128x128/variations_manifest.csv"
df = pd.read_csv(mpath)
df = df[["original_path","synthetic_path"]].dropna().head(200)
df.to_csv("/tmp/uc2_128_fid_paired.csv", index=False)
print(f"FID paired manifest: {len(df)} rows → /tmp/uc2_128_fid_paired.csv")
PY

python use_case_1_balance_dataset/evaluate_quality_metrics.py \
    --paired_manifest /tmp/uc2_128_fid_paired.csv \
    --real_video_col original_path \
    --synthetic_video_col synthetic_path \
    --video_dir . \
    --num_samples 200 \
    --batch_size 1 \
    --frames_per_chunk 8 \
    --device cuda \
    --output_file uc2_128x128_fid.json || log "FID step failed (non-fatal)"

log "STEP 5 done"

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────
log ""
log "===== FINAL SUMMARY ====="
python - <<'PY'
import json, pathlib

results = {
    "UC2 128×128 SSIM/PSNR/MSE":  "uc2_128x128_metrics.json",
    "UC3 128×128 SSIM/PSNR/MSE":  "uc3_128x128_metrics.json",
    "UC2 128×128 FID":             "uc2_128x128_fid.json",
    "OLD UC1 FID (64×64 baseline)":"fid_sanity_latest.json",
}
for label, fpath in results.items():
    p = pathlib.Path(fpath)
    if p.exists():
        d = json.loads(p.read_text())
        print(f"\n{label}:")
        for k,v in d.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
    else:
        print(f"\n{label}: (not found — {fpath})")
PY

log "========================================================"
log "128×128 AUGMENTATION PIPELINE COMPLETE"
log "========================================================"
log "Log saved to: $LOG"
