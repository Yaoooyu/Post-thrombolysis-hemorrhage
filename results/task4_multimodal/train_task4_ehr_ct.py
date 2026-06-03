from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


EHR_COLS_24 = [
    "htn_history",
    "prior_stroke_history",
    "hd_CHD",
    "hd_MI",
    "hd_VAF",
    "hd_NVAF",
    "hd_Other_HD",
    "alcohol_history",
    "diabetes_history",
    "pre_stroke_mrs",
    "dnt_time",
    "arterial_plaque",
    "admission_glucose",
    "stent_implantation",
    "hypertension_drug",
    "diabetes_drug",
    "gender",
    "admission_nihss",
    "admission_sbp",
    "smoking_history",
    "mechanical_thrombectomy",
    "age",
    "dyslipidemia_history",
    "admission_dbp",
]


def _img_feat_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("img_feat_")]
    cols = sorted(cols, key=lambda x: int(str(x).split("_")[-1]))
    return cols


def _seed(seed: int):
    np.random.seed(seed)


def _load_ct488_fused(
    finaltext_xlsx: str,
    mapping_csv: str,
    manifest_ct488_csv: str,
    image_features_csv: str,
) -> pd.DataFrame:
    """
    Build CT488 cohort table for Task4 (EHR + CT image embedding):
    - EHR: 24 structured cols from finaltext.xlsx
    - CT: img_feat_0..511 from image_features_ct488.csv
    - Split/label: from data_split_manifest_ct488.csv (source of truth)
    - Join key: finaltext.xlsx has id (001..), map to global_id via ehr_ct_id_mapping.csv
    """
    ehr = pd.read_excel(finaltext_xlsx)
    manifest = pd.read_csv(manifest_ct488_csv)[["global_id", "label", "split"]]

    img = pd.read_csv(image_features_csv).copy()
    # Keep only embeddings + global_id; split/label (if present) are ignored
    img = img.drop(columns=["label", "split"], errors="ignore")

    if "global_id" in ehr.columns:
        # Table already has globally-unique key
        ehr = ehr.drop(columns=["split"], errors="ignore")
        # Keep EHR label only for sanity checks; final label comes from manifest
        df = manifest.merge(ehr.drop(columns=["label"], errors="ignore"), on="global_id", how="inner").merge(
            img, on="global_id", how="inner"
        )
    else:
        # finaltext.xlsx: id is group-local and repeats across label.
        # Correct join key is (label, local_id) -> global_id.
        if "id" not in ehr.columns or "label" not in ehr.columns:
            raise ValueError("finaltext.xlsx must contain id and label when global_id is absent")

        ehr = ehr.copy()
        ehr["local_id"] = ehr["id"].astype(str).str.zfill(3)
        ehr["label"] = pd.to_numeric(ehr["label"], errors="coerce").astype(int)

        mapping = pd.read_csv(mapping_csv)[["global_id", "local_id", "label"]].copy()
        mapping["local_id"] = mapping["local_id"].astype(str).str.zfill(3)
        mapping["label"] = pd.to_numeric(mapping["label"], errors="coerce").astype(int)

        ehr = ehr.drop(columns=["split"], errors="ignore").merge(
            mapping, on=["label", "local_id"], how="inner"
        )

        # Now align to CT488 manifest and override label from manifest (sanity-check first)
        tmp = manifest.merge(ehr, on="global_id", how="inner", suffixes=("", "_ehr"))
        if "label_ehr" in tmp.columns:
            mism = int((tmp["label"] != tmp["label_ehr"]).sum())
            if mism != 0:
                raise ValueError(f"Label mismatch between manifest and EHR after mapping: {mism}")
            tmp = tmp.drop(columns=["label_ehr"], errors="ignore")
        df = tmp.merge(img, on="global_id", how="inner")

    if len(df) != 488:
        raise ValueError(f"Expected 488 rows after merge, got {len(df)}")
    return df


def _median_impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    return med


def _median_impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    out = X.copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.take(med, np.where(bad)[1])
    return out


@dataclass(frozen=True)
class Metrics:
    method: str
    test_roc_auc: float
    val_roc_auc: float
    test_pr_auc: float
    val_pr_auc: float
    test_acc_0_5: float
    val_acc_0_5: float
    test_precision_0_5: float
    val_precision_0_5: float
    test_recall_0_5: float
    val_recall_0_5: float
    test_f1_0_5: float
    val_f1_0_5: float
    test_specificity_0_5: float
    val_specificity_0_5: float
    test_confusion_0_5: list[list[int]]
    val_confusion_0_5: list[list[int]]
    # Threshold tuned on val via Youden's J
    val_best_threshold_youden: float
    test_acc_youden: float
    test_precision_youden: float
    test_recall_youden: float
    test_f1_youden: float
    test_specificity_youden: float
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    seed: int


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = float((tp + tn) / max(1, tp + tn + fp + fn))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / max(1, tn + fp))
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # maximize (sensitivity + specificity - 1)
    # candidate thresholds: unique probs
    thr_list = np.unique(y_prob)
    best_thr = 0.5
    best_j = -1e9
    for thr in thr_list:
        m = _metrics_at_threshold(y_true, y_prob, float(thr))
        j = m["recall"] + m["specificity"] - 1.0
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return float(best_thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finaltext", default="/root/autodl-fs/lyy/finaltext.xlsx")
    ap.add_argument("--mapping", default="/root/autodl-fs/lyy/ehr_ct_id_mapping.csv")
    ap.add_argument("--manifest", default="/root/autodl-fs/lyy/data_split_manifest_ct488.csv")
    ap.add_argument("--image-features", default="/root/autodl-fs/lyy/unified_runs/image_features_ct488.csv")
    ap.add_argument("--outdir", default="/root/autodl-fs/lyy/task4_ehr_ct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _seed(args.seed)

    df = _load_ct488_fused(args.finaltext, args.mapping, args.manifest, args.image_features)

    img_cols = _img_feat_cols(df)
    if len(img_cols) != 512:
        raise ValueError(f"Expected 512 img_feat_* cols, got {len(img_cols)}")
    missing = [c for c in EHR_COLS_24 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing EHR cols in final merged table: {missing}")

    feat_cols = EHR_COLS_24 + img_cols
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy()

    idx_tr = df["split"].values == "train"
    idx_va = df["split"].values == "val"
    idx_te = df["split"].values == "test"

    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xva, yva = X[idx_va], y[idx_va]
    Xte, yte = X[idx_te], y[idx_te]

    # train-only preprocess
    med = _median_impute_fit(Xtr)
    Xtr = _median_impute_apply(Xtr, med)
    Xva = _median_impute_apply(Xva, med)
    Xte = _median_impute_apply(Xte, med)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    Xte = sc.transform(Xte)

    clf = LogisticRegression(
        C=args.C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=args.seed,
    )

    t0 = time.time()
    clf.fit(Xtr, ytr)
    train_s = time.time() - t0

    p_val = clf.predict_proba(Xva)[:, 1]
    p_test = clf.predict_proba(Xte)[:, 1]
    val_auc = float(roc_auc_score(yva, p_val))
    test_auc = float(roc_auc_score(yte, p_test))
    val_pr = float(average_precision_score(yva, p_val))
    test_pr = float(average_precision_score(yte, p_test))

    m_val_05 = _metrics_at_threshold(yva, p_val, 0.5)
    m_test_05 = _metrics_at_threshold(yte, p_test, 0.5)

    thr_y = _best_threshold_youden(yva, p_val)
    m_test_y = _metrics_at_threshold(yte, p_test, thr_y)

    metrics = Metrics(
        method="ehr24 + img512 -> logreg (median-impute+zscore, train-only)",
        test_roc_auc=test_auc,
        val_roc_auc=val_auc,
        test_pr_auc=test_pr,
        val_pr_auc=val_pr,
        test_acc_0_5=m_test_05["acc"],
        val_acc_0_5=m_val_05["acc"],
        test_precision_0_5=m_test_05["precision"],
        val_precision_0_5=m_val_05["precision"],
        test_recall_0_5=m_test_05["recall"],
        val_recall_0_5=m_val_05["recall"],
        test_f1_0_5=m_test_05["f1"],
        val_f1_0_5=m_val_05["f1"],
        test_specificity_0_5=m_test_05["specificity"],
        val_specificity_0_5=m_val_05["specificity"],
        test_confusion_0_5=m_test_05["confusion"],
        val_confusion_0_5=m_val_05["confusion"],
        val_best_threshold_youden=thr_y,
        test_acc_youden=m_test_y["acc"],
        test_precision_youden=m_test_y["precision"],
        test_recall_youden=m_test_y["recall"],
        test_f1_youden=m_test_y["f1"],
        test_specificity_youden=m_test_y["specificity"],
        n_train=int(len(ytr)),
        n_val=int(len(yva)),
        n_test=int(len(yte)),
        n_features=int(Xtr.shape[1]),
        seed=int(args.seed),
    )

    out = {
        **asdict(metrics),
        "train_seconds": train_s,
        "C": float(args.C),
    }

    out_path = os.path.join(args.outdir, "metrics_logreg.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Save val/test predictions for downstream unified metric table
    val_pred = pd.DataFrame(
        {
            "global_id": df.loc[idx_va, "global_id"].astype(str).values,
            "y_true": yva,
            "y_prob": p_val,
        }
    )
    test_pred = pd.DataFrame(
        {
            "global_id": df.loc[idx_te, "global_id"].astype(str).values,
            "y_true": yte,
            "y_prob": p_test,
        }
    )
    val_pred.to_csv(os.path.join(args.outdir, "val_pred_logreg.csv"), index=False, encoding="utf-8-sig")
    test_pred.to_csv(os.path.join(args.outdir, "test_pred_logreg.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

