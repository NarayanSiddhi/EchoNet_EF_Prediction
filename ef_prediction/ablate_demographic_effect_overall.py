"""
Demographic ablation on the *full* validation manifest (same logic as
ablate_demographic_effect.py: normal / zero / shuffle on demo_vec), without
touching grouped subgroup CSVs.

Outputs are `demographic_ablation_overall_{fused|real}_*` so they do not
overwrite subgroup `demographic_ablation_*.csv`, and fused vs real do not
overwrite each other.

Run from repo root:
  python ef_prediction/ablate_demographic_effect_overall.py --model fused
  python ef_prediction/ablate_demographic_effect_overall.py --model real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ef_prediction.ablate_demographic_effect import (  # noqa: E402
    evaluate_group,
    evaluate_group_real,
)
from ef_prediction.models.pt_efnet_fused import PTEFNetFused  # noqa: E402
from ef_prediction.models.pt_efnet_real import PTEFNetReal  # noqa: E402


def resolve_manifest(cfg: dict, model: str, override: str | None) -> tuple[str, Path]:
    if override:
        p = Path(override).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--manifest not found: {p}")
        safe = p.stem.replace(" ", "_")
        return (f"OVERALL_{safe}", p)

    if model == "fused":
        rel = cfg["data"].get("val_manifest_fused")
        key = "OVERALL_VAL_FUSED"
    else:
        rel = cfg["data"].get("val_manifest")
        key = "OVERALL_VAL_REAL"
    if not rel:
        raise ValueError("config.yaml missing val manifest path for this model type.")
    path = Path(rel)
    if not path.is_absolute():
        path = (ROOT / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Val manifest not found: {path}")
    return key, path


def parse_args():
    p = argparse.ArgumentParser(
        description="Demographic ablation (normal/zero/shuffle) on full val manifest."
    )
    p.add_argument("--model", choices=["fused", "real"], default="fused")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults: fused -> checkpoints/fused/run_1_best.pth; real -> checkpoints/real/best.pth",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="Override manifest CSV (default: val_manifest_fused or val_manifest from config).",
    )
    p.add_argument(
        "--output-dir",
        default="ef_prediction/group_results",
        help="Directory for *_overall_* outputs.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["normal", "zero", "shuffle"],
        choices=["normal", "zero", "shuffle"],
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is None:
        if args.model == "fused":
            args.checkpoint = "ef_prediction/checkpoints/fused/run_1_best.pth"
        else:
            args.checkpoint = "ef_prediction/checkpoints/real/best.pth"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("ef_prediction/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    group_key, manifest_path = resolve_manifest(cfg, args.model, args.manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "fused":
        model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
        eval_fn = evaluate_group
    else:
        backbone = cfg["model"].get("backbone", "resnet34")
        model = PTEFNetReal(backbone=backbone).to(device)
        eval_fn = evaluate_group_real

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rng = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    rng.manual_seed(args.seed)

    grouped_metrics: dict = {group_key: {}}
    long_rows = []

    print(f"Model={args.model} checkpoint={args.checkpoint}")
    print(f"Manifest: {manifest_path}")
    print(f"Group key: {group_key}  modes={args.modes}")

    for mode in args.modes:
        m = eval_fn(
            csv_file=manifest_path,
            model=model,
            cfg=cfg,
            device=device,
            mode=mode,
            batch_size=args.batch_size,
            rng=rng,
        )
        if m is None:
            print(f"  {mode:>7}: skipped (empty dataset)")
            continue
        grouped_metrics[group_key][mode] = m
        print(
            f"  {mode:>7}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
            f"R2={m['R2']:.4f} n={m['Count']}"
        )

    base = grouped_metrics[group_key].get("normal")
    for mode, metrics in grouped_metrics[group_key].items():
        row = {
            "Group": group_key,
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

    tag = f"demographic_ablation_overall_{args.model}"
    long_df = pd.DataFrame(long_rows)
    long_csv = out_dir / f"{tag}_long.csv"
    long_df.to_csv(long_csv, index=False)

    by_mode = grouped_metrics[group_key]
    summary_row = {"Group": group_key, "Count": by_mode["normal"]["Count"]}
    for mode in args.modes:
        if mode not in by_mode:
            continue
        summary_row[f"MAE_{mode}"] = by_mode[mode]["MAE"]
        summary_row[f"RMSE_{mode}"] = by_mode[mode]["RMSE"]
        summary_row[f"R2_{mode}"] = by_mode[mode]["R2"]
        if mode != "normal":
            summary_row[f"Delta_MAE_{mode}_vs_normal"] = (
                by_mode[mode]["MAE"] - by_mode["normal"]["MAE"]
            )
            summary_row[f"Delta_RMSE_{mode}_vs_normal"] = (
                by_mode[mode]["RMSE"] - by_mode["normal"]["RMSE"]
            )
            summary_row[f"Delta_R2_{mode}_vs_normal"] = (
                by_mode[mode]["R2"] - by_mode["normal"]["R2"]
            )

    summary_df = pd.DataFrame([summary_row])
    summary_csv = out_dir / f"{tag}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(manifest_path),
        "group_key": group_key,
        "metrics": grouped_metrics[group_key],
    }
    metrics_json = out_dir / f"{tag}_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved:")
    print(f"  {long_csv}")
    print(f"  {summary_csv}")
    print(f"  {metrics_json}")


if __name__ == "__main__":
    main()
