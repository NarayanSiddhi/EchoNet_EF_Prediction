#!/usr/bin/env python3
"""
Recalculate SSIM, PSNR, MSE for all 3 use cases from actual video files.
Results saved to recalc_metrics_all_uc.json
"""
import json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path("/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION")


def load_video(path, video_length=32, video_size=128):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < video_length:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.ndim == 3:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(fr, (video_size, video_size)))
    cap.release()
    while len(frames) < video_length:
        frames.append(frames[-1] if frames else np.zeros((video_size, video_size), np.uint8))
    return np.array(frames[:video_length], dtype=np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return np.inf
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_metrics(orig_path, syn_path, video_length=32, video_size=128):
    a = load_video(orig_path, video_length, video_size).astype(np.float32) / 255.0
    b = load_video(syn_path,  video_length, video_size).astype(np.float32) / 255.0
    T = min(len(a), len(b))
    ssims = [ssim_fn(a[t], b[t], data_range=1.0) for t in range(T)]
    psnrs = [psnr(a[t] * 255, b[t] * 255) for t in range(T)]
    mses  = [float(np.mean((a[t] - b[t]) ** 2)) for t in range(T)]
    return float(np.mean(ssims)), float(np.mean(psnrs)), float(np.mean(mses))


def evaluate_manifest(manifest_path, orig_col, syn_col,
                      video_length=32, video_size=128,
                      max_samples=None, label=""):
    df = pd.read_csv(manifest_path)
    print(f"\n{'='*60}")
    print(f"{label}  ({len(df)} rows in manifest)")
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  Sampling {max_samples} rows for speed")
    print(f"{'='*60}")

    ssims, psnrs, mses, errors = [], [], [], 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=label):
        orig = ROOT / str(row[orig_col])
        syn  = ROOT / str(row[syn_col])
        if not orig.exists() or not syn.exists():
            errors += 1
            continue
        try:
            s, p, m = compute_metrics(orig, syn, video_length, video_size)
            ssims.append(s)
            psnrs.append(p)
            mses.append(m)
        except Exception:
            errors += 1

    finite_psnrs = [p for p in psnrs if not np.isinf(p)]
    result = {
        "n":          len(ssims),
        "errors":     errors,
        "SSIM_mean":  float(np.mean(ssims)),
        "SSIM_std":   float(np.std(ssims)),
        "PSNR_mean":  float(np.mean(finite_psnrs)),
        "PSNR_std":   float(np.std(finite_psnrs)),
        "PSNR_inf_count": int(sum(1 for p in psnrs if np.isinf(p))),
        "MSE_mean":   float(np.mean(mses)),
        "MSE_std":    float(np.std(mses)),
    }
    print(f"  Evaluated : {result['n']} videos  |  Errors: {result['errors']}")
    print(f"  SSIM      : {result['SSIM_mean']:.4f} ± {result['SSIM_std']:.4f}")
    print(f"  PSNR      : {result['PSNR_mean']:.2f} ± {result['PSNR_std']:.2f} dB  (inf: {result['PSNR_inf_count']})")
    print(f"  MSE       : {result['MSE_mean']:.6f} ± {result['MSE_std']:.6f}")
    return result


# ── UC1 ──────────────────────────────────────────────────────────────────────
# UC1 synthetic paths are relative to use_case_1_balance_dataset/
UC1_ROOT = ROOT / "use_case_1_balance_dataset"

def evaluate_manifest_uc1(manifest_path, orig_col, syn_col,
                           orig_root, syn_root,
                           video_length=32, video_size=128,
                           max_samples=None, label=""):
    df = pd.read_csv(manifest_path)
    print(f"\n{'='*60}")
    print(f"{label}  ({len(df)} rows in manifest)")
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  Sampling {max_samples} rows for speed")
    print(f"{'='*60}")

    ssims, psnrs, mses, errors = [], [], [], 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=label):
        orig = orig_root / str(row[orig_col])
        syn  = syn_root  / str(row[syn_col])
        if not orig.exists() or not syn.exists():
            errors += 1
            continue
        try:
            s, p, m = compute_metrics(orig, syn, video_length, video_size)
            ssims.append(s); psnrs.append(p); mses.append(m)
        except Exception:
            errors += 1

    finite_psnrs = [p for p in psnrs if not np.isinf(p)]
    result = {
        "n":          len(ssims),
        "errors":     errors,
        "SSIM_mean":  float(np.mean(ssims))       if ssims else None,
        "SSIM_std":   float(np.std(ssims))        if ssims else None,
        "PSNR_mean":  float(np.mean(finite_psnrs)) if finite_psnrs else None,
        "PSNR_std":   float(np.std(finite_psnrs))  if finite_psnrs else None,
        "PSNR_inf_count": int(sum(1 for p in psnrs if np.isinf(p))),
        "MSE_mean":   float(np.mean(mses))        if mses else None,
        "MSE_std":    float(np.std(mses))         if mses else None,
    }
    print(f"  Evaluated : {result['n']} videos  |  Errors: {result['errors']}")
    if result['SSIM_mean'] is not None:
        print(f"  SSIM      : {result['SSIM_mean']:.4f} ± {result['SSIM_std']:.4f}")
        print(f"  PSNR      : {result['PSNR_mean']:.2f} ± {result['PSNR_std']:.2f} dB  (inf: {result['PSNR_inf_count']})")
        print(f"  MSE       : {result['MSE_mean']:.6f} ± {result['MSE_std']:.6f}")
    return result

uc1_result = evaluate_manifest_uc1(
    manifest_path = ROOT / "use_case_1_balance_dataset/synthetic_paired_dataset/synthetic_manifest.csv",
    orig_col      = "original_video_path",
    syn_col       = "synthetic_video_path",
    orig_root     = ROOT,          # original paths start with data/processed/...
    syn_root      = UC1_ROOT,      # synthetic paths start with synthetic_paired_dataset/...
    video_length  = 32,
    video_size    = 128,
    max_samples   = 3000,
    label         = "UC1 — Dataset Balancing (Conditional 3D DCGAN)",
)

# ── UC2 ──────────────────────────────────────────────────────────────────────
uc2_result = evaluate_manifest(
    manifest_path = ROOT / "use_case_2_demographic_variations/demographic_variations_128x128/variations_manifest.csv",
    orig_col      = "original_path",
    syn_col       = "synthetic_path",
    video_length  = 32,
    video_size    = 128,
    max_samples   = None,   # evaluate all 18,681
    label         = "UC2 — Demographic Variations (Conditional 3D U-Net GAN)",
)

# ── UC3 ──────────────────────────────────────────────────────────────────────
uc3_result = evaluate_manifest(
    manifest_path = ROOT / "perfect_synthetic_copies_128x128/perfect_copies_manifest.csv",
    orig_col      = "original_path",
    syn_col       = "synthetic_path",
    video_length  = 32,
    video_size    = 128,
    max_samples   = None,   # evaluate all 6,227
    label         = "UC3 — Perfect Reconstruction (U-Net GAN)",
)

# ── Save ─────────────────────────────────────────────────────────────────────
output = {
    "UC1_balancing":        uc1_result,
    "UC2_demographic_variations": uc2_result,
    "UC3_perfect_copies":   uc3_result,
}
out_path = ROOT / "recalc_metrics_all_uc.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
for uc, res in output.items():
    print(f"\n{uc}:")
    print(f"  SSIM : {res['SSIM_mean']:.4f} ± {res['SSIM_std']:.4f}")
    print(f"  PSNR : {res['PSNR_mean']:.2f} ± {res['PSNR_std']:.2f} dB")
    print(f"  MSE  : {res['MSE_mean']:.6f} ± {res['MSE_std']:.6f}")
print(f"\nSaved → {out_path}")
