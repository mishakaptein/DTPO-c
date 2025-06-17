import os
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

rpo_dir  = "logs_rpo_continuous/"
dtpo_dir = "./logs_dtpo_continuous/"

def load_rpo_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    arr = np.array(data)
    timestamps = arr[:, 0].astype(float)
    timesteps = arr[:, 1].astype(float)
    returns_und = arr[:, 2].astype(float)

    rel_time = timestamps - timestamps[0]
    return timesteps, returns_und, rel_time

def load_dtpo_file(path):
    timesteps_list = []
    returns_list = []
    reltime_list = []

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            timesteps_list.append(float(obj["step"]))
            reltime_list.append(float(obj["relative_time"]))
            returns_list.append(float(obj["undiscounted_return"]))

    timesteps = np.array(timesteps_list, dtype=float)
    returns_und = np.array(returns_list, dtype=float)
    rel_time = np.array(reltime_list, dtype=float)
    return timesteps, returns_und, rel_time

def interp_to_grid(source_x, source_y, grid_x):
    x_min, x_max = source_x[0], source_x[-1]
    x_clipped = np.clip(grid_x, x_min, x_max)
    return np.interp(x_clipped, source_x, source_y)

def compute_mean_sem(all_runs_interp):
    arr = np.stack(all_runs_interp, axis=0)
    mean_curve = np.mean(arr, axis=0)
    sem_curve = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean_curve, sem_curve

if __name__ == "__main__":
    output_dir = "./out_plots/comparison_plots_dtpo_vs_rpo/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rpo_files = sorted(glob.glob(os.path.join(rpo_dir, "*.json")))
    dtpo_files = sorted(glob.glob(os.path.join(dtpo_dir, "*.json")))

    if len(rpo_files) == 0:
        raise RuntimeError(f"No RPO JSON files found in {rpo_dir}/*.json")
    if len(dtpo_files) == 0:
        raise RuntimeError(f"No DTPO JSON files found in {dtpo_dir}/*.json")

    print(f"Found {len(rpo_files)} RPO files, {len(dtpo_files)} DTPO files.")


    # Load RPO runs
    rpo_runs = []
    for fp in rpo_files:
        ts, rets, rt = load_rpo_file(fp)
        sort_idx = np.argsort(ts)
        rpo_runs.append({
            "timesteps": ts[sort_idx],
            "returns": rets[sort_idx],
            "rel_time": rt[sort_idx]
        })

    # Load DTPO runs
    dtpo_runs = []
    for fp in dtpo_files:
        ts, rets, rt = load_dtpo_file(fp)
        sort_idx = np.argsort(ts)
        dtpo_runs.append({
            "timesteps": ts[sort_idx],
            "returns": rets[sort_idx],
            "rel_time": rt[sort_idx]
        })

    # For RPO, find the minimum of (max timesteps across all its seeds)
    max_rpo_ts_per_seed = [run["timesteps"][-1] for run in rpo_runs]
    max_ts_rpo = float(min(max_rpo_ts_per_seed))
    if max_ts_rpo <= 0:
        raise RuntimeError("RPO runs have non‐positive max timesteps")

    # For RPO, find the minimum of (max relative_time across all its seeds)
    max_rpo_time_per_seed = [run["rel_time"][-1] for run in rpo_runs]
    max_time_rpo = float(min(max_rpo_time_per_seed))
    if max_time_rpo <= 0:
        raise RuntimeError("RPO runs have non‐positive max relative_time")

    # For DTPO, do the same
    max_dtpo_ts_per_seed = [run["timesteps"][-1] for run in dtpo_runs]
    max_ts_dtpo = float(min(max_dtpo_ts_per_seed))
    if max_ts_dtpo <= 0:
        raise RuntimeError("DTPO-c runs have non‐positive max timesteps")

    max_dtpo_time_per_seed = [run["rel_time"][-1] for run in dtpo_runs]
    max_time_dtpo = float(min(max_dtpo_time_per_seed))
    if max_time_dtpo <= 0:
        raise RuntimeError("DTPO-c runs have non‐positive max relative_time")

    print(f"RPO: max‐common timesteps = {int(max_ts_rpo)}, max‐common time = {max_time_rpo:.3f}s")
    print(f"DTPO-c: max‐common timesteps = {int(max_ts_dtpo)}, max‐common time = {max_time_dtpo:.3f}s")

    # Build the two grids
    N = 200
    grid_ts_rpo = np.linspace(0.0, max_ts_rpo, N)
    grid_ts_dtpo = np.linspace(0.0, max_ts_dtpo, N)
    grid_time_rpo = np.linspace(0.0, max_time_rpo, N)
    grid_time_dtpo = np.linspace(0.0, max_time_dtpo, N)

    interp_rpo_on_ts = []
    interp_rpo_on_time = []
    for run in rpo_runs:
        xs = run["timesteps"]
        ys = run["returns"]
        xs_t = run["rel_time"]
        interp_rpo_on_ts.append(interp_to_grid(xs, ys, grid_ts_rpo))
        interp_rpo_on_time.append(interp_to_grid(xs_t, ys, grid_time_rpo))

    interp_dtpo_on_ts = []
    interp_dtpo_on_time = []
    for run in dtpo_runs:
        xs = run["timesteps"]
        ys = run["returns"]
        xs_t = run["rel_time"]
        interp_dtpo_on_ts.append(interp_to_grid(xs, ys, grid_ts_dtpo))
        interp_dtpo_on_time.append(interp_to_grid(xs_t, ys, grid_time_dtpo))

    # Compute mean and SEM for RPO and DTPO-c
    rpo_mean_ts, rpo_sem_ts = compute_mean_sem(interp_rpo_on_ts)
    dtpo_mean_ts, dtpo_sem_ts = compute_mean_sem(interp_dtpo_on_ts)

    rpo_mean_time, rpo_sem_time = compute_mean_sem(interp_rpo_on_time)
    dtpo_mean_time, dtpo_sem_time = compute_mean_sem(interp_dtpo_on_time)

    # Determine the plot ranges
    ts_xlim_max = max(max_ts_rpo, max_ts_dtpo)
    time_xlim_max = max(max_time_rpo, max_time_dtpo)

    # Plot 1: RPO Return vs Timesteps
    plt.figure(figsize=(8, 5))

    for run in interp_rpo_on_ts:
        plt.plot(grid_ts_rpo, run, color="C0", alpha=0.2, linewidth=0.8)

    plt.plot(grid_ts_rpo, rpo_mean_ts, color="C0", linewidth=2, label="RPO (mean)")
    plt.fill_between(grid_ts_rpo,
        rpo_mean_ts - rpo_sem_ts,
        rpo_mean_ts + rpo_sem_ts,
        color="C0", alpha=0.2)
    plt.xlabel("Cumulative Timesteps")
    plt.ylabel("Undiscounted Return")
    plt.title("RPO: Return vs Timesteps")
    plt.xlim(0, max_ts_rpo)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "return_vs_timesteps_rpo.png"), dpi=300)
    print("Saved: return_vs_timesteps_rpo.png")
    plt.close()

    # Plot 2: DTPO-c Return vs Timesteps
    plt.figure(figsize=(8, 5))

    for run in interp_dtpo_on_ts:
        plt.plot(grid_ts_dtpo, run, color="C1", alpha=0.2, linewidth=0.8)

    plt.plot(grid_ts_dtpo, dtpo_mean_ts, color="C1", linewidth=2, label="DTPO-c (mean)")
    plt.fill_between(grid_ts_dtpo,
        dtpo_mean_ts - dtpo_sem_ts,
        dtpo_mean_ts + dtpo_sem_ts,
        color="C1", alpha=0.2)
    plt.xlabel("Cumulative Timesteps")
    plt.ylabel("Undiscounted Return")
    plt.title("DTPO-c: Return vs Timesteps")
    plt.xlim(0, max_ts_dtpo)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "return_vs_timesteps_dtpo-c.png"), dpi=300)
    print("Saved: return_vs_timesteps_dtpo-c.png")
    plt.close()

    # Plot 3: Return vs Runtime
    plt.figure(figsize=(8, 5))

    for run in interp_rpo_on_time:
        plt.plot(grid_time_rpo, run, color="C0", alpha=0.2, linewidth=0.8)

    for run in interp_dtpo_on_time:
        plt.plot(grid_time_dtpo, run, color="C1", alpha=0.2, linewidth=0.8)

    plt.plot(grid_time_rpo,   rpo_mean_time, color="C0", linewidth=2, label="RPO (mean)")
    plt.fill_between(grid_time_rpo,
                     rpo_mean_time - rpo_sem_time,
                     rpo_mean_time + rpo_sem_time,
                     color="C0", alpha=0.2)

    plt.plot(grid_time_dtpo,  dtpo_mean_time, color="C1", linewidth=2, label="DTPO-c (mean)")
    plt.fill_between(grid_time_dtpo,
                     dtpo_mean_time - dtpo_sem_time,
                     dtpo_mean_time + dtpo_sem_time,
                     color="C1", alpha=0.2)

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Undiscounted Return")
    plt.title("Average Return vs Runtime")
    plt.xlim(0, time_xlim_max)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "return_vs_time.png"), dpi=300)
    print("Saved: return_vs_time.png")
    plt.close()
    print("Finished")