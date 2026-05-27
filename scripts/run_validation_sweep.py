"""Run focused XAI pipeline at multiple dataset sizes for validation.

Stages:
  1. N=500
  2. N=1000
  3. Full dataset (--max-samples -1)

Each stage saves results to a separate output folder.

Usage:
    python scripts/run_validation_sweep.py --dry-run
    python scripts/run_validation_sweep.py --run-n500 --skip-n1000 --skip-full
    python scripts/run_validation_sweep.py --run-n500 --run-n1000 --skip-full
    python scripts/run_validation_sweep.py --run-n500 --run-n1000 --run-full
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Validation sweep: run focused XAI at multiple N")

    # Stage toggles
    p.add_argument("--run-n500", action="store_true", default=False,
                   help="Run N=500 stage")
    p.add_argument("--skip-n500", action="store_true", default=False,
                   help="Skip N=500 stage")
    p.add_argument("--run-n1000", action="store_true", default=False,
                   help="Run N=1000 stage")
    p.add_argument("--skip-n1000", action="store_true", default=False,
                   help="Skip N=1000 stage")
    p.add_argument("--run-full", action="store_true", default=False,
                   help="Run full dataset stage")
    p.add_argument("--skip-full", action="store_true", default=False,
                   help="Skip full dataset stage")

    # Shared parameters
    p.add_argument("--n-boot", type=int, default=2000,
                   help="Bootstrap iterations (default 2000)")
    p.add_argument("--montage-file", type=str,
                   default=str(ROOT / "config" / "dtu_channel_montage.csv"))
    p.add_argument("--base-output-prefix", type=str, default="xai_results_focused",
                   help="Base prefix for output directories")
    p.add_argument("--balanced-by-subject", action="store_true", default=True)
    p.add_argument("--no-balanced-by-subject", dest="balanced_by_subject",
                   action="store_false")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print commands without running")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite existing output folders")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


def resolve_output_dir(base_dir: Path, overwrite: bool) -> Path:
    """Return the output directory, adding a timestamp suffix if it exists
    and --overwrite is not set."""
    if not base_dir.exists():
        return base_dir
    if overwrite:
        return base_dir
    # Add timestamp suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    new_dir = base_dir.parent / f"{base_dir.name}_{ts}"
    print(f"  Output dir exists: {base_dir}")
    print(f"  Using timestamped: {new_dir}")
    return new_dir


def build_stage_commands(stage_name: str, max_samples: int, ig_samples: int,
                         windows_per_subject: int, output_dir: str,
                         args) -> list[list[str]]:
    """Build the command lists for a stage (run_focused_xai + postprocess)."""
    focused_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_focused_xai.py"),
        "--max-samples", str(max_samples),
        "--n-boot", str(args.n_boot),
        "--ig-samples", str(ig_samples),
        "--windows-per-subject", str(windows_per_subject),
        "--montage-file", args.montage_file,
        "--output-dir", output_dir,
        "--random-seed", str(args.seed),
        "--device", args.device,
    ]
    if args.balanced_by_subject:
        focused_cmd.append("--balanced-by-subject")
    else:
        focused_cmd.append("--no-balanced-by-subject")

    postprocess_cmd = [
        sys.executable, str(ROOT / "scripts" / "postprocess_focused_xai.py"),
        "--output-dir", output_dir,
        "--montage-file", args.montage_file,
    ]

    return [focused_cmd, postprocess_cmd]


def run_command(cmd: list[str], log_path: Path, dry_run: bool) -> tuple[int, float]:
    """Run a command, log output, return (exit_code, elapsed_seconds)."""
    cmd_str = " ".join(cmd)
    print(f"\n  $ {cmd_str}")

    if dry_run:
        print("  [DRY-RUN] Skipping execution")
        return 0, 0.0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n{'='*70}\n")
        log_f.write(f"Command: {cmd_str}\n")
        log_f.write(f"Started: {datetime.now().isoformat()}\n")
        log_f.write(f"{'='*70}\n\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        proc.wait()

        elapsed = time.time() - start
        log_f.write(f"\nExit code: {proc.returncode}\n")
        log_f.write(f"Elapsed: {elapsed:.1f}s\n")

    return proc.returncode, elapsed


def estimate_full_dataset_windows() -> int | None:
    """Try to estimate total windows in the full dataset without loading it."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from aad_xai.data.vlaai_dataset import VLAAIDTUDataset
        data_dir = ROOT / "external" / "vlaai" / "evaluation_datasets" / "DTU"
        if not data_dir.exists():
            return None
        ds = VLAAIDTUDataset(data_dir=str(data_dir), window_length=320, hop=64)
        return len(ds)
    except Exception:
        return None


def main():
    args = parse_args()

    print("=" * 70)
    print("VALIDATION SWEEP: Focused XAI at Multiple Dataset Sizes")
    print("=" * 70)
    print(f"  Bootstrap: {args.n_boot}")
    print(f"  Montage: {args.montage_file}")
    print(f"  Base prefix: {args.base_output_prefix}")
    print(f"  Balanced by subject: {args.balanced_by_subject}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Overwrite: {args.overwrite}")
    print("=" * 70)

    # Determine which stages to run
    stages = []

    run_500 = args.run_n500 and not args.skip_n500
    run_1000 = args.run_n1000 and not args.skip_n1000
    run_full = args.run_full and not args.skip_full

    # If none explicitly set, show help
    if not run_500 and not run_1000 and not run_full and not args.dry_run:
        print("\nNo stages selected. Use --run-n500, --run-n1000, --run-full to select stages.")
        print("Use --dry-run to preview commands without running.")
        return

    # If dry-run with no stage flags, show all stages
    if args.dry_run and not run_500 and not run_1000 and not run_full:
        run_500 = run_1000 = run_full = True

    if run_500:
        out_dir_500 = resolve_output_dir(ROOT / f"{args.base_output_prefix}_n500", args.overwrite)
        stages.append({
            "name": "N=500",
            "max_samples": 500,
            "ig_samples": 50,
            "windows_per_subject": 20,
            "output_dir": str(out_dir_500),
        })

    if run_1000:
        out_dir_1000 = resolve_output_dir(ROOT / f"{args.base_output_prefix}_n1000", args.overwrite)
        stages.append({
            "name": "N=1000",
            "max_samples": 1000,
            "ig_samples": 100,
            "windows_per_subject": 30,
            "output_dir": str(out_dir_1000),
        })

    if run_full:
        out_dir_full = resolve_output_dir(ROOT / f"{args.base_output_prefix}_full", args.overwrite)

        # Warn about large dataset
        est_windows = estimate_full_dataset_windows()
        if est_windows is not None:
            print(f"\n  Full dataset estimated windows: {est_windows}")
            if est_windows > 5000:
                print(f"  WARNING: Full dataset has {est_windows} windows (>5000).")
                print("  This may take a long time. Consider using --skip-full for initial testing.")
        stages.append({
            "name": "Full dataset",
            "max_samples": -1,
            "ig_samples": 100,
            "windows_per_subject": 50,
            "output_dir": str(out_dir_full),
        })

    # Print summary
    print(f"\nStages to run: {len(stages)}")
    for s in stages:
        ms = "ALL" if s["max_samples"] == -1 else str(s["max_samples"])
        print(f"  - {s['name']}: max_samples={ms}, output={s['output_dir']}")
    print()

    # Run each stage
    results = []
    summary_path = ROOT / "validation_sweep_summary.txt"

    for si, stage in enumerate(stages):
        print(f"\n{'#'*70}")
        print(f"# STAGE {si+1}/{len(stages)}: {stage['name']}")
        print(f"# Output: {stage['output_dir']}")
        print(f"{'#'*70}")

        out_dir = Path(stage["output_dir"])
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "validation_stage.log"

        cmds = build_stage_commands(
            stage["name"], stage["max_samples"], stage["ig_samples"],
            stage["windows_per_subject"], stage["output_dir"], args
        )

        stage_start = time.time()
        stage_ok = True

        for cmd in cmds:
            exit_code, elapsed = run_command(cmd, log_path, args.dry_run)
            if exit_code != 0:
                print(f"\n  STAGE FAILED with exit code {exit_code}")
                stage_ok = False
                break

        stage_elapsed = time.time() - stage_start

        results.append({
            "stage": stage["name"],
            "max_samples": stage["max_samples"],
            "output_dir": stage["output_dir"],
            "success": stage_ok,
            "elapsed_seconds": round(stage_elapsed, 1),
        })

        if not stage_ok:
            print(f"\n  Stopping sweep due to stage failure: {stage['name']}")
            break

    # Write summary
    print(f"\n{'='*70}")
    print("VALIDATION SWEEP SUMMARY")
    print(f"{'='*70}")

    summary_lines = []
    summary_lines.append("VALIDATION SWEEP SUMMARY")
    summary_lines.append(f"Date: {datetime.now().isoformat()}")
    summary_lines.append(f"Dry run: {args.dry_run}")
    summary_lines.append(f"Bootstrap: {args.n_boot}")
    summary_lines.append(f"Balanced by subject: {args.balanced_by_subject}")
    summary_lines.append("")

    for r in results:
        ms = "ALL" if r["max_samples"] == -1 else str(r["max_samples"])
        status = "OK" if r["success"] else "FAILED"
        summary_lines.append(f"  {r['stage']:20s}: {status:6s} ({r['elapsed_seconds']:.1f}s) -> {r['output_dir']}")
        print(f"  {r['stage']:20s}: {status:6s} ({r['elapsed_seconds']:.1f}s)")

    summary_lines.append("")

    # List available run folders for comparison
    summary_lines.append("Available run folders for comparison:")
    existing = ROOT / f"{args.base_output_prefix}"
    if existing.exists():
        summary_lines.append(f"  {existing}")
    for r in results:
        if r["success"]:
            summary_lines.append(f"  {r['output_dir']}")

    summary_lines.append("")
    summary_lines.append("Next step:")
    run_dirs = []
    if existing.exists():
        run_dirs.append(str(existing))
    for r in results:
        if r["success"]:
            run_dirs.append(r["output_dir"])
    if run_dirs:
        cmd_str = (f"python scripts/compare_validation_runs.py "
                   f"--runs {' '.join(run_dirs)} "
                   f"--output-dir xai_results_validation_comparison")
        summary_lines.append(f"  {cmd_str}")

    summary_text = "\n".join(summary_lines)
    if not args.dry_run:
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"\n  Summary saved to {summary_path}")
    else:
        print(f"\n  [DRY-RUN] Summary would be saved to {summary_path}")

    print()
    all_ok = all(r["success"] for r in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
