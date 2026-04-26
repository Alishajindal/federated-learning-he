# #!/usr/bin/env python3
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # === CONFIG ===
# # Change this to the folder where FeSViBS saved the CSV (the same save_dir used when running training)
# save_dir = "vit_base_r50_s16_224_0.0001lr_bloodmnist_6Clients_1to6Blocks_32Batch_FeSViBS"
# csv_path = os.path.join(save_dir, "he_encryption_log.csv")

# # === LOAD CSV ===
# if not os.path.exists(csv_path):
#     raise FileNotFoundError(f"CSV not found: {csv_path}\nMake sure save_dir points to your experiment output folder.")
# df = pd.read_csv(csv_path)

# # convert numeric columns (they might be strings if written by earlier code)
# for col in ["round", "client_id", "last_block_chunks", "cls_chunks", "pos_chunks", "enc_time_sec", "he_time_sec", "cpu_mem_mb", "gpu_mem_mb"]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

# # === SUMMARY TABLE ===
# summary = df.groupby("client_id").agg(
#     enc_time_mean = ("enc_time_sec", "mean"),
#     enc_time_median = ("enc_time_sec", "median"),
#     enc_time_max = ("enc_time_sec", "max"),
#     cpu_mem_mean_mb = ("cpu_mem_mb", "mean"),
#     gpu_mem_mean_mb = ("gpu_mem_mb", "mean"),
#     last_block_chunks = ("last_block_chunks", "mean")
# ).reset_index()

# print("Per-client summary (mean/median/max enc time, mean mem):")
# print(summary.to_string(index=False))

# # make output dir for plots
# plots_dir = os.path.join(save_dir, "he_plots")
# os.makedirs(plots_dir, exist_ok=True)

# # === PLOT 1: Mean encryption time per client (bar) ===
# plt.figure(figsize=(8,5))
# x = summary["client_id"].astype(int).tolist()
# y = summary["enc_time_mean"].tolist()
# plt.bar(x, y)
# plt.xlabel("Client ID")
# plt.ylabel("Mean encryption time (sec)")
# plt.title("Mean Encryption Time per Client")
# plt.xticks(x)
# plt.tight_layout()
# out1 = os.path.join(plots_dir, "mean_encryption_time_per_client.png")
# plt.savefig(out1)
# print(f"Saved: {out1}")
# plt.show()

# # === PLOT 2: Mean CPU memory (RSS) per client (bar) ===
# plt.figure(figsize=(8,5))
# y_cpu = summary["cpu_mem_mean_mb"].tolist()
# plt.bar(x, y_cpu)
# plt.xlabel("Client ID")
# plt.ylabel("Mean CPU RSS (MB)")
# plt.title("Mean CPU Memory (RSS) per Client During Encryption")
# plt.xticks(x)
# plt.tight_layout()
# out2 = os.path.join(plots_dir, "mean_cpu_mem_per_client.png")
# plt.savefig(out2)
# print(f"Saved: {out2}")
# plt.show()

# # === PLOT 3: Mean GPU reserved memory per client (bar) ===
# plt.figure(figsize=(8,5))
# y_gpu = summary["gpu_mem_mean_mb"].fillna(0).tolist()
# plt.bar(x, y_gpu)
# plt.xlabel("Client ID")
# plt.ylabel("Mean GPU reserved memory (MB)")
# plt.title("Mean GPU Reserved Memory per Client During Encryption")
# plt.xticks(x)
# plt.tight_layout()
# out3 = os.path.join(plots_dir, "mean_gpu_mem_per_client.png")
# plt.savefig(out3)
# print(f"Saved: {out3}")
# plt.show()

# # === OPTIONAL: time-series of enc_time per client across rounds (line plot) ===
# # If you want a per-round trend, uncomment below.
# if True:
#     plt.figure(figsize=(10,6))
#     for cid, group in df.groupby("client_id"):
#         plt.plot(group["round"], group["enc_time_sec"], marker="o", label=f"client {int(cid)}")
#     plt.xlabel("Round")
#     plt.ylabel("Encryption time (sec)")
#     plt.title("Encryption time per client across rounds")
#     plt.legend()
#     plt.tight_layout()
#     out_ts = os.path.join(plots_dir, "enc_time_trends_per_client.png")
#     plt.savefig(out_ts)
#     print(f"Saved: {out_ts}")
#     plt.show()



#!/usr/bin/env python3
"""
Plot HE encryption diagnostics from CSV.

Usage:
    python plot_he_metrics.py --csv he_encryption_log.csv --outdir he_plots
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def ensure_numeric_columns(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(safe_float)
        else:
            df[c] = np.nan
    return df

def make_dirs(outdir):
    os.makedirs(outdir, exist_ok=True)

def savefig(fig, outdir, name, dpi=200):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")

def main(csv_path, outdir):
    make_dirs(outdir)

    # load
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # expected numeric columns (may contain blanks)
    numeric_cols = ["round", "client_id", "chosen_block",
                    "last_block_chunks", "cls_chunks", "pos_chunks",
                    "enc_time_sec", "he_time_sec", "cpu_mem_mb", "gpu_mem_mb"]
    df = ensure_numeric_columns(df, numeric_cols)

    # Basic cleaning: drop rows with no encryption time and no chunk info
    if df["enc_time_sec"].isnull().all():
        print("Warning: enc_time_sec column is empty or non-numeric.")
    df = df.dropna(subset=["client_id"], how="all")

    # Convert client_id to integer for grouping if possible
    if df["client_id"].notnull().any():
        df["client_id"] = df["client_id"].astype(pd.Int64Dtype())

    # ===== Summary table =====
    summary = df.groupby("client_id").agg({
        "enc_time_sec": ["mean", "median", "std", "count"],
        "cpu_mem_mb": ["mean", "median", "std"],
        "gpu_mem_mb": ["mean", "median", "std"],
        "last_block_chunks": "mean",
    })
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(outdir, "he_summary_by_client.csv"), index=False)
    print(f"Saved summary CSV: {os.path.join(outdir, 'he_summary_by_client.csv')}")

    # ===== Mean bar charts per client =====
    clients = df["client_id"].dropna().unique()
    clients = np.sort(clients)

    # Mean encryption time per client
    fig = plt.figure(figsize=(9,5))
    ax = sns.barplot(x="client_id", y="enc_time_sec", data=df, estimator=np.mean, ci=None, order=clients)
    ax.set_title("Mean Encryption Time per Client")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Mean encryption time (sec)")
    savefig(fig, outdir, "mean_enc_time_per_client")

    # Mean CPU memory
    if df["cpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(9,5))
        ax = sns.barplot(x="client_id", y="cpu_mem_mb", data=df, estimator=np.mean, ci=None, order=clients)
        ax.set_title("Mean CPU Memory (RSS) per Client During Encryption")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Mean CPU RSS (MB)")
        savefig(fig, outdir, "mean_cpu_mem_per_client")

    # Mean GPU memory
    if df["gpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(9,5))
        ax = sns.barplot(x="client_id", y="gpu_mem_mb", data=df, estimator=np.mean, ci=None, order=clients)
        ax.set_title("Mean GPU Reserved Memory per Client During Encryption")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Mean GPU reserved memory (MB)")
        savefig(fig, outdir, "mean_gpu_mem_per_client")

    # ===== Boxplots to show spread & outliers =====
    fig = plt.figure(figsize=(10,6))
    ax = sns.boxplot(x="client_id", y="enc_time_sec", data=df, order=clients)
    ax.set_title("Encryption Time Distribution per Client (boxplot)")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Encryption time (sec)")
    savefig(fig, outdir, "boxplot_enc_time_per_client")

    if df["cpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(10,6))
        ax = sns.boxplot(x="client_id", y="cpu_mem_mb", data=df, order=clients)
        ax.set_title("CPU Memory (RSS) Distribution per Client (boxplot)")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("CPU RSS (MB)")
        savefig(fig, outdir, "boxplot_cpu_mem_per_client")

    if df["gpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(10,6))
        ax = sns.boxplot(x="client_id", y="gpu_mem_mb", data=df, order=clients)
        ax.set_title("GPU Reserved Memory Distribution per Client (boxplot)")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("GPU reserved mem (MB)")
        savefig(fig, outdir, "boxplot_gpu_mem_per_client")

    # ===== Histograms / KDE =====
    fig = plt.figure(figsize=(8,5))
    ax = sns.histplot(df["enc_time_sec"].dropna(), kde=True, bins=20)
    ax.set_title("Distribution of Encryption Time (all clients)")
    ax.set_xlabel("Encryption time (sec)")
    savefig(fig, outdir, "hist_enc_time_all")

    if df["cpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(8,5))
        ax = sns.histplot(df["cpu_mem_mb"].dropna(), kde=True, bins=20)
        ax.set_title("Distribution of CPU RSS during encryption (all clients)")
        ax.set_xlabel("CPU RSS (MB)")
        savefig(fig, outdir, "hist_cpu_mem_all")

    if df["gpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(8,5))
        ax = sns.histplot(df["gpu_mem_mb"].dropna(), kde=True, bins=20)
        ax.set_title("Distribution of GPU reserved mem during encryption (all clients)")
        ax.set_xlabel("GPU reserved mem (MB)")
        savefig(fig, outdir, "hist_gpu_mem_all")

    # ===== Scatter plots: enc_time vs memory =====
    if df["cpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(8,6))
        ax = sns.regplot(x="enc_time_sec", y="cpu_mem_mb", data=df.dropna(subset=["enc_time_sec","cpu_mem_mb"]))
        ax.set_title("Encryption time vs CPU RSS (per record)")
        ax.set_xlabel("Encryption time (sec)")
        ax.set_ylabel("CPU RSS (MB)")
        savefig(fig, outdir, "scatter_enc_time_cpu_mem")

    if df["gpu_mem_mb"].notnull().any():
        fig = plt.figure(figsize=(8,6))
        ax = sns.regplot(x="enc_time_sec", y="gpu_mem_mb", data=df.dropna(subset=["enc_time_sec","gpu_mem_mb"]))
        ax.set_title("Encryption time vs GPU reserved mem (per record)")
        ax.set_xlabel("Encryption time (sec)")
        ax.set_ylabel("GPU reserved mem (MB)")
        savefig(fig, outdir, "scatter_enc_time_gpu_mem")

    # ===== Correlation heatmap =====
    numeric_df = df[["enc_time_sec","he_time_sec","cpu_mem_mb","gpu_mem_mb","last_block_chunks","cls_chunks","pos_chunks"]].copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method="pearson")
    fig = plt.figure(figsize=(8,6))
    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix (numeric HE metrics)")
    savefig(fig, outdir, "corr_heatmap_he_metrics")

    # ===== Per-round trends (optional) =====
    round_stats = df.groupby("round").agg({
        "enc_time_sec": "mean",
        "cpu_mem_mb": "mean",
        "gpu_mem_mb": "mean"
    }).reset_index()

    if "enc_time_sec" in round_stats.columns:
        fig = plt.figure(figsize=(10,5))
        ax = sns.lineplot(x="round", y="enc_time_sec", data=round_stats, marker="o")
        ax.set_title("Mean Encryption Time per Round")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean enc_time_sec")
        savefig(fig, outdir, "mean_enc_time_per_round")

    if "cpu_mem_mb" in round_stats.columns:
        fig = plt.figure(figsize=(10,5))
        ax = sns.lineplot(x="round", y="cpu_mem_mb", data=round_stats, marker="o")
        ax.set_title("Mean CPU RSS per Round")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean CPU RSS (MB)")
        savefig(fig, outdir, "mean_cpu_mem_per_round")

    if "gpu_mem_mb" in round_stats.columns:
        fig = plt.figure(figsize=(10,5))
        ax = sns.lineplot(x="round", y="gpu_mem_mb", data=round_stats, marker="o")
        ax.set_title("Mean GPU reserved memory per Round")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean GPU reserved mem (MB)")
        savefig(fig, outdir, "mean_gpu_mem_per_round")

    print("All plots generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HE encryption metrics from CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to he_encryption_log.csv")
    parser.add_argument("--outdir", type=str, default="he_plots", help="Output directory for PNGs/CSVs")
    args = parser.parse_args()
    main(args.csv, args.outdir)
