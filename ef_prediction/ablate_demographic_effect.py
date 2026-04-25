import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Allow direct execution: python ef_prediction/ablate_demographic_effect.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ef_prediction.dataset import DualVideoEFDataset as RealVideoEFDataset
from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.models.pt_efnet_fused import PTEFNetFused
from ef_prediction.models.pt_efnet_real import PTEFNetReal


def compute_metrics(preds, labels):
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    denom = np.sum((labels - labels.mean()) ** 2)
    r2 = 0.0 if denom == 0 else 1 - np.sum((labels - preds) ** 2) / denom
    return mae, mse, rmse, r2


def apply_demo_mode(demo_vec, mode, rng):
    if mode == "normal":
        return demo_vec
    if mode == "zero":
        return torch.zeros_like(demo_vec)
    if mode == "shuffle":
        if demo_vec.shape[0] < 2:
            return demo_vec
        perm = torch.randperm(demo_vec.shape[0], generator=rng, device=demo_vec.device)
        return demo_vec[perm]
    raise ValueError(f"Unsupported mode: {mode}")


def evaluate_group(csv_file, model, cfg, device, mode, batch_size, rng):
    dataset = DualVideoEFDataset(
        manifest_path=str(csv_file),
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True,
    )

    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels = [], []

    with torch.no_grad():
        for real_video, syn_video, ef, _, _, _, demo_vec in loader:
            real_video = real_video.to(device)
            syn_video = syn_video.to(device)
            ef = ef.to(device)
            demo_vec = demo_vec.to(device).float()

            demo_input = apply_demo_mode(demo_vec, mode, rng)
            pred, _ = model(real_video, syn_video, demo_input)

            preds.extend(pred.cpu().numpy())
            labels.extend(ef.cpu().numpy())

    preds = np.array(preds) * 100.0
    labels = np.array(labels) * 100.0
    mae, mse, rmse, r2 = compute_metrics(preds, labels)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Count": int(len(labels)),
    }


def evaluate_group_real(csv_file, model, cfg, device, mode, batch_size, rng):
    """Same ablation modes as fused, but real-only backbone (video + demo_vec)."""
    dataset = RealVideoEFDataset(
        manifest_path=str(csv_file),
        video_root_dir=cfg["data"]["original_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False,
    )

    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels = [], []

    with torch.no_grad():
        for video, ef, _, _, _, demo_vec in loader:
            video = video.to(device)
            ef = ef.to(device)
            demo_vec = demo_vec.to(device).float()

            demo_input = apply_demo_mode(demo_vec, mode, rng)
            pred, _ = model(video, demo_input)

            preds.extend(pred.cpu().numpy())
            labels.extend(ef.cpu().numpy())

    preds = np.array(preds) * 100.0
    labels = np.array(labels) * 100.0
    mae, mse, rmse, r2 = compute_metrics(preds, labels)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Count": int(len(labels)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate demographic conditioning (normal / zero / shuffle) for fused or real EF models."
    )
    parser.add_argument(
        "--model",
        choices=["fused", "real"],
        default="fused",
        help="fused: real+synthetic videos; real: original video only (late fusion demo_vec).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults: fused -> run_1_best.pth, real -> checkpoints/real/best.pth",
    )
    parser.add_argument(
        "--group-dir",
        default="ef_prediction/grouped_manifests",
        help="Directory containing grouped manifest CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="ef_prediction/group_results",
        help="Directory to save ablation outputs.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["normal", "zero", "shuffle"],
        choices=["normal", "zero", "shuffle"],
        help="Demographic modes to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional subset of group names (without .csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is None:
        if args.model == "fused":
            args.checkpoint = "ef_prediction/checkpoints/fused/run_1_best.pth"
        else:
            args.checkpoint = "ef_prediction/checkpoints/real/best.pth"

    group_dir = Path(args.group_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("ef_prediction/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "fused":
        model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
    else:
        backbone = cfg["model"].get("backbone", "resnet34")
        model = PTEFNetReal(backbone=backbone).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    eval_group_fn = evaluate_group if args.model == "fused" else evaluate_group_real

    csv_files = sorted(group_dir.glob("*.csv"))
    if args.groups:
        wanted = set(args.groups)
        csv_files = [f for f in csv_files if f.stem in wanted]

    if not csv_files:
        raise FileNotFoundError("No group CSV files found for ablation.")

    rng = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    rng.manual_seed(args.seed)

    long_rows = []
    grouped_metrics = {}

    print(f"Model={args.model} checkpoint={args.checkpoint}")
    print(f"Evaluating {len(csv_files)} groups with modes: {args.modes}")
    for csv_file in csv_files:
        group_name = csv_file.stem
        grouped_metrics[group_name] = {}
        print(f"\nGroup: {group_name}")

        for mode in args.modes:
            metrics = eval_group_fn(
                csv_file=csv_file,
                model=model,
                cfg=cfg,
                device=device,
                mode=mode,
                batch_size=args.batch_size,
                rng=rng,
            )
            if metrics is None:
                print(f"  {mode:>7}: skipped (empty dataset)")
                continue
            grouped_metrics[group_name][mode] = metrics
            print(
                f"  {mode:>7}: MAE={metrics['MAE']:.3f} "
                f"RMSE={metrics['RMSE']:.3f} R2={metrics['R2']:.4f} n={metrics['Count']}"
            )

        base = grouped_metrics[group_name].get("normal")
        for mode, metrics in grouped_metrics[group_name].items():
            row = {
                "Group": group_name,
                "Mode": mode,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Count": metrics["Count"],
            }
            if base is not None and mode != "normal":
                row["Delta_MAE_vs_normal"] = metrics["MAE"] - base["MAE"]
                row["Delta_RMSE_vs_normal"] = metrics["RMSE"] - base["RMSE"]
                row["Delta_R2_vs_normal"] = metrics["R2"] - base["R2"]
            else:
                row["Delta_MAE_vs_normal"] = 0.0
                row["Delta_RMSE_vs_normal"] = 0.0
                row["Delta_R2_vs_normal"] = 0.0
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    long_df = long_df.sort_values(["Group", "Mode"]).reset_index(drop=True)
    long_csv = output_dir / "demographic_ablation_long.csv"
    long_df.to_csv(long_csv, index=False)

    summary_rows = []
    for group_name, by_mode in grouped_metrics.items():
        if "normal" not in by_mode:
            continue
        row = {"Group": group_name, "Count": by_mode["normal"]["Count"]}
        for mode in args.modes:
            if mode not in by_mode:
                continue
            row[f"MAE_{mode}"] = by_mode[mode]["MAE"]
            row[f"RMSE_{mode}"] = by_mode[mode]["RMSE"]
            row[f"R2_{mode}"] = by_mode[mode]["R2"]
            if mode != "normal":
                row[f"Delta_MAE_{mode}_vs_normal"] = by_mode[mode]["MAE"] - by_mode["normal"]["MAE"]
                row[f"Delta_RMSE_{mode}_vs_normal"] = by_mode[mode]["RMSE"] - by_mode["normal"]["RMSE"]
                row[f"Delta_R2_{mode}_vs_normal"] = by_mode[mode]["R2"] - by_mode["normal"]["R2"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Group").reset_index(drop=True)
    summary_csv = output_dir / "demographic_ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    metrics_json = output_dir / "demographic_ablation_metrics.json"
    payload = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "groups": grouped_metrics,
    }
    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:")
    print(f"  - {long_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {metrics_json}")

    if not summary_df.empty:
        print("\nMean effects across groups:")
        for mode in args.modes:
            if mode == "normal":
                continue
            key = f"Delta_MAE_{mode}_vs_normal"
            if key in summary_df.columns:
                print(f"  {mode:>7}: mean ΔMAE = {summary_df[key].mean():+.4f}")


if __name__ == "__main__":
    main()
