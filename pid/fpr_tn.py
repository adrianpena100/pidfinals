import pandas as pd
import numpy  as np
import glob, os, warnings, pathlib

# ─── 1. EDIT THESE TWO NUMBERS FOR *THIS* BATCH OF LOGS ────────────────
TOTAL_CLIENTS     = 30  # total clients in this run
MALICIOUS_CLIENTS = 0     # 0 ⇒ non-malicious run, else # malicious
# ────────────────────────────────────────────────────────────────────────
BENIGN_CLIENTS = TOTAL_CLIENTS - MALICIOUS_CLIENTS
P, N = MALICIOUS_CLIENTS, BENIGN_CLIENTS

# destination folder (inside logs/)
OUT_DIR = pathlib.Path("logs") / "fprtn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _col(df, name):
    return df[name] if name in df.columns else None

def enrich(path: str, P: int, N: int) -> None:
    df  = pd.read_csv(path)
    out = pd.DataFrame()
    out["round"] = df.index if "round" not in df.columns else df["round"]

    if P == 0:
        acc = _col(df, "accuracy")
        if acc is not None:
            TN = (acc * N).round().astype(int)
            FP = N - TN
            fpr = FP / N
        else:
            warnings.warn(
                f"[{os.path.basename(path)}] No accuracy column; "
                "assuming detector never flags benign clients (FP=0)."
            )
            FP = 0
            TN = N
            fpr = 0.0
        out["FP"], out["TN"], out["fpr"] = FP, TN, fpr

    else:
        recall    = _col(df, "recall")
        precision = _col(df, "precision")
        accuracy  = _col(df, "accuracy")

        if recall is None:
            raise ValueError(f"{path}: 'recall' column required.")
        TP = (recall * P).round().astype(int)

        if precision is not None and (precision > 0).any():
            FP = np.where(
                precision == 0,
                0,
                (TP * (1 / precision - 1)).round().astype(int),
            )
        elif accuracy is not None:
            total = P + N
            TN = (accuracy * total - TP).round().astype(int)
            FP = N - TN
        else:
            raise ValueError(
                f"{path}: need 'precision' or 'accuracy' to derive FP."
            )

        TN  = N - FP
        fpr = FP / N
        out["FP"], out["TN"], out["fpr"] = FP, TN, fpr

    # ----- write to logs/fprtn/<same-basename>_fp_tn.csv -------------
    fname = os.path.basename(path).replace(".csv", "_fp_tn.csv")
    out_path = OUT_DIR / fname
    out.to_csv(out_path, index=False)
    print(f"✅  wrote {out_path}")

if __name__ == "__main__":
    for csv_file in glob.glob(os.path.join("logs", "*_log_*.csv")):
        enrich(csv_file, P, N)
