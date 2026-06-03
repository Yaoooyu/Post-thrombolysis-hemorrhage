from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "acc": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "n_test": int(len(y_true)),
    }


def _load_pred(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    need = {"global_id", "y_true", "y_prob"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {sorted(need - set(df.columns))}")
    return df[["global_id", "y_true", "y_prob"]].copy()


def _safe_load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    outdir = "/root/autodl-fs/lyy/task4_ehr_ct"

    model_files = {
        "logreg": os.path.join(outdir, "test_pred_logreg.csv"),
        "mlp": os.path.join(outdir, "test_pred_mlp.csv"),
        "catboost": os.path.join(outdir, "test_pred_catboost.csv"),
        "lightgbm": os.path.join(outdir, "test_pred_lightgbm.csv"),
        "xgboost": os.path.join(outdir, "test_pred_xgboost.csv"),
    }

    rows = []
    base_df = None

    for model_name, pred_path in model_files.items():
        df = _load_pred(pred_path)
        if df.empty:
            continue

        if base_df is None:
            base_df = df[["global_id", "y_true"]].copy()
        else:
            chk = base_df.merge(df[["global_id", "y_true"]], on="global_id", how="inner", suffixes=("", "_m"))
            if len(chk) != len(base_df):
                raise ValueError(f"{model_name} prediction rows not aligned with base test set")
            if int((chk["y_true"] != chk["y_true_m"]).sum()) != 0:
                raise ValueError(f"{model_name} y_true mismatch with base test set")

        m = _calc_metrics(df["y_true"].astype(int).to_numpy(), df["y_prob"].to_numpy(), threshold=0.5)

        # attach val roc if exists in metrics json
        metric_json_map = {
            "logreg": os.path.join(outdir, "metrics_logreg.json"),
            "mlp": os.path.join(outdir, "metrics_mlp.json"),
            "catboost": os.path.join(outdir, "metrics_catboost.json"),
            "lightgbm": os.path.join(outdir, "metrics_lightgbm.json"),
            "xgboost": os.path.join(outdir, "metrics_xgboost.json"),
        }
        j = _safe_load_json(metric_json_map[model_name])
        m["model"] = model_name
        m["val_roc_auc"] = j.get("val_roc_auc")
        m["val_pr_auc"] = j.get("val_pr_auc")
        rows.append(m)

    if not rows:
        raise RuntimeError("No model prediction files found. Please run model scripts first.")

    # Mean ensemble (no leakage): average probabilities from available models
    pred_cols = []
    merged = base_df.copy()
    for model_name, pred_path in model_files.items():
        df = _load_pred(pred_path)
        if df.empty:
            continue
        col = f"p_{model_name}"
        pred_cols.append(col)
        merged = merged.merge(df[["global_id", "y_prob"]].rename(columns={"y_prob": col}), on="global_id", how="left")
    if pred_cols:
        merged["y_prob_mean_ens"] = merged[pred_cols].mean(axis=1)
        ens = _calc_metrics(
            merged["y_true"].astype(int).to_numpy(),
            merged["y_prob_mean_ens"].to_numpy(),
            threshold=0.5,
        )
        ens["model"] = "mean_ensemble"
        ens["val_roc_auc"] = None
        ens["val_pr_auc"] = None
        rows.append(ens)

    tbl = pd.DataFrame(rows)
    tbl = tbl[
        [
            "model",
            "val_roc_auc",
            "val_pr_auc",
            "roc_auc",
            "pr_auc",
            "acc",
            "precision",
            "recall",
            "f1",
            "specificity",
            "tn",
            "fp",
            "fn",
            "tp",
            "threshold",
            "n_test",
        ]
    ].sort_values("roc_auc", ascending=False)

    csv_path = os.path.join(outdir, "task4_model_metrics_table.csv")
    xlsx_path = os.path.join(outdir, "task4_model_metrics_table.xlsx")
    tbl.to_csv(csv_path, index=False, encoding="utf-8-sig")
    tbl.to_excel(xlsx_path, index=False)

    print("Saved:", csv_path)
    print("Saved:", xlsx_path)
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()

