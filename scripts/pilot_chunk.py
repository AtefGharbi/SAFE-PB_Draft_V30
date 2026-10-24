#!/usr/bin/env python
"""Run a small subset of (variant, seed) combos from run_v17_experiments.py's
apparatus and APPEND to a persistent episode_results.csv, so a full pilot can
be assembled across several short, separately-invoked calls (each call in
this sandbox is killed a few minutes in, with no cross-call background
process persistence). Reuses train_one/evaluate_one/build_summary/
build_statistics unmodified -- no experiment logic is duplicated here."""
import sys, os, json, time, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_v17_experiments import (VARIANTS, train_one, evaluate_one, build_summary,
                                  build_statistics, sha256, CONFIG)
from safe_pb.feeder_model import build_feeder

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True, help="e.g. 'SAFE-PB:0,SAFE-PB:1'")
ap.add_argument("--train-episodes", type=int, default=70)
ap.add_argument("--updates-per-episode", type=int, default=8)
ap.add_argument("--eval-days", type=int, default=6)
ap.add_argument("--eval-seed-base", type=int, default=9_000_000)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--replay-capacity", type=int, default=5000)
ap.add_argument("--beta-r", type=float, default=0.1)
ap.add_argument("--beta-f", type=float, default=0.05)
ap.add_argument("--cvar-alpha", type=float, default=0.90)
ap.add_argument("--risk-warmup", type=int, default=20)
ap.add_argument("--output-dir", default=str(ROOT / "results" / "v17_pilot"))
args = ap.parse_args()

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
results_csv = out / "episode_results.csv"
manifest_path = out / "run_manifest.json"

manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
    "status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "args": vars(args), "config_sha256": sha256(CONFIG), "runs": [], "note":
        "Assembled incrementally across several short calls (pilot_chunk.py); "
        "each call runs 1-2 (variant,seed) combos and appends to episode_results.csv."}

pairs = [p.split(":") for p in args.pairs.split(",")]
topo = build_feeder()
existing_df = pd.read_csv(results_csv) if results_csv.exists() else None

for variant, seed_s in pairs:
    seed = int(seed_s)
    t0 = time.time()
    print(f"[{variant}] seed={seed}: training")
    tr = train_one(seed, variant, args, topo, out)
    print(f"[{variant}] seed={seed}: held-out evaluation")
    rows = evaluate_one(tr, seed, variant, args, topo)
    df_new = pd.DataFrame(rows)
    existing_df = df_new if existing_df is None else pd.concat([existing_df, df_new], ignore_index=True)
    existing_df.to_csv(results_csv, index=False)
    manifest["runs"].append({"method": variant, "seed": seed, "seconds": time.time() - t0, "status": "passed"})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{variant}] seed={seed}: done in {time.time()-t0:.1f}s, appended to {results_csv}")

print("Chunk complete.", len(existing_df), "total episode rows accumulated so far.")
