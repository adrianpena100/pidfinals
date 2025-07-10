import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.ticker import ScalarFormatter

# Strategies and their colors for plotting
strategies = {
    "FedAvg": "black",
    #"FedPIDAvg_default": "blue",
    "FedPIDAvg_tuned": "red",
    "Krum": "green",
    "Bulyan": "orange",
    # Add more as needed
}

plt.figure(figsize=(10, 6))

# Plot Loss (log scale)
plotted_any = False
for name, color in strategies.items():
    fname = f"{name}_log.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        if "round" in df.columns and "loss" in df.columns:
            plt.plot(df["round"], df["loss"], label=name, color=color, linewidth=2)
            plotted_any = True
        else:
            print(f"Warning: {fname} does not have required columns, skipping.")
    else:
        print(f"Warning: {fname} not found, skipping.")
if not plotted_any:
    print("No data was plotted. Please check that your *_log.csv files exist and contain 'round' and 'loss' columns.")
plt.xlabel("Communication Round", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Loss vs Communication Round", fontsize=16)
plt.grid(True, which="both", ls=":", lw=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("comparison_plot.png")
plt.show()

# --- Plot Accuracy ---
plt.figure(figsize=(10, 6))
plotted_any = False
for name, color in strategies.items():
    fname = f"{name}_log.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        if "round" in df.columns and "accuracy" in df.columns:
            plt.plot(df["round"], df["accuracy"], label=name, color=color, linewidth=2)
            plotted_any = True
        else:
            print(f"Warning: {fname} does not have required columns for accuracy, skipping.")
    else:
        print(f"Warning: {fname} not found, skipping.")
if not plotted_any:
    print("No accuracy data was plotted. Please check that your *_log.csv files exist and contain 'round' and 'accuracy' columns.")
plt.xlabel("Communication Round", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.ylim(0, 1)
plt.title("Accuracy vs Communication Round", fontsize=16)
plt.grid(True, which="both", ls=":", lw=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("comparison_accuracy.png")
plt.show()

# --- Plot Recall ---
plt.figure(figsize=(10, 6))
plotted_any = False
for name, color in strategies.items():
    fname = f"{name}_log.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        if "round" in df.columns and "recall" in df.columns:
            plt.plot(df["round"], df["recall"], label=name, color=color, linewidth=2)
            plotted_any = True
        else:
            print(f"Warning: {fname} does not have required columns for recall, skipping.")
    else:
        print(f"Warning: {fname} not found, skipping.")
if not plotted_any:
    print("No recall data was plotted. Please check that your *_log.csv files exist and contain 'round' and 'recall' columns.")
plt.xlabel("Communication Round", fontsize=14)
plt.ylabel("Recall", fontsize=14)
plt.ylim(0, 1)
plt.title("Recall vs Communication Round", fontsize=16)
plt.grid(True, which="both", ls=":", lw=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("comparison_recall.png")
plt.show()

# --- Plot Precision ---
plt.figure(figsize=(10, 6))
plotted_any = False
for name, color in strategies.items():
    fname = f"{name}_log.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        if "round" in df.columns and "precision" in df.columns:
            plt.plot(df["round"], df["precision"], label=name, color=color, linewidth=2)
            plotted_any = True
        else:
            print(f"Warning: {fname} does not have required columns for precision, skipping.")
    else:
        print(f"Warning: {fname} not found, skipping.")
if not plotted_any:
    print("No precision data was plotted. Please check that your *_log.csv files exist and contain 'round' and 'precision' columns.")
plt.xlabel("Communication Round", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.ylim(0, 1)
plt.title("Precision vs Communication Round", fontsize=16)
plt.grid(True, which="both", ls=":", lw=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("comparison_precision.png")
plt.show()
