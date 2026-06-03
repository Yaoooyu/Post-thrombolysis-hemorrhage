from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


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


def _load_ct488_fused(
    finaltext_xlsx: str,
    mapping_csv: str,
    manifest_ct488_csv: str,
    image_features_csv: str,
) -> pd.DataFrame:
    ehr = pd.read_excel(finaltext_xlsx)
    manifest = pd.read_csv(manifest_ct488_csv)[["global_id", "label", "split"]]

    img = pd.read_csv(image_features_csv).drop(columns=["label", "split"], errors="ignore")

    if "global_id" in ehr.columns:
        ehr = ehr.drop(columns=["split"], errors="ignore")
        df = manifest.merge(ehr.drop(columns=["label"], errors="ignore"), on="global_id", how="inner").merge(
            img, on="global_id", how="inner"
        )
    else:
        if "id" not in ehr.columns or "label" not in ehr.columns:
            raise ValueError("finaltext.xlsx must contain id and label when global_id is absent")
        ehr = ehr.copy()
        ehr["local_id"] = ehr["id"].astype(str).str.zfill(3)
        ehr["label"] = pd.to_numeric(ehr["label"], errors="coerce").astype(int)

        mapping = pd.read_csv(mapping_csv)[["global_id", "local_id", "label"]].copy()
        mapping["local_id"] = mapping["local_id"].astype(str).str.zfill(3)
        mapping["label"] = pd.to_numeric(mapping["label"], errors="coerce").astype(int)

        ehr = ehr.drop(columns=["split"], errors="ignore").merge(mapping, on=["label", "local_id"], how="inner")
        tmp = manifest.merge(ehr, on="global_id", how="inner", suffixes=("", "_ehr"))
        if "label_ehr" in tmp.columns:
            mism = int((tmp["label"] != tmp["label_ehr"]).sum())
            if mism:
                raise ValueError(f"Label mismatch between manifest and EHR after mapping: {mism}")
            tmp = tmp.drop(columns=["label_ehr"], errors="ignore")
        df = tmp.merge(img, on="global_id", how="inner")

    if len(df) != 488:
        raise ValueError(f"Expected 488 rows after merge, got {len(df)}")
    return df


def _bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, seed: int, n_boot: int = 2000):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        pp = p[idx]
        # need both classes
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
    aucs = np.array(aucs, dtype=float)
    if len(aucs) == 0:
        return {"median": None, "p2_5": None, "p97_5": None, "n_boot_effective": 0}
    return {
        "median": float(np.median(aucs)),
        "p2_5": float(np.percentile(aucs, 2.5)),
        "p97_5": float(np.percentile(aucs, 97.5)),
        "n_boot_effective": int(len(aucs)),
    }


@dataclass(frozen=True)
class Metrics:
    model: str
    calibrated: bool
    val_roc_auc: float
    test_roc_auc: float
    val_pr_auc: float
    test_pr_auc: float
    test_auc_bootstrap_ci95: dict
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    seed: int


def _make_model(name: str, seed: int):
    if name == "catboost":
        return CatBoostClassifier(
            iterations=4000,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            l2_leaf_reg=5.0,
            auto_class_weights="Balanced",
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            class_weight="balanced",
        )
    if name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=8,
        )
    raise ValueError(f"Unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finaltext", default="/root/autodl-fs/lyy/finaltext.xlsx")
    ap.add_argument("--mapping", default="/root/autodl-fs/lyy/ehr_ct_id_mapping.csv")
    ap.add_argument("--manifest", default="/root/autodl-fs/lyy/data_split_manifest_ct488.csv")
    ap.add_argument("--image-features", default="/root/autodl-fs/lyy/unified_runs/image_features_ct488.csv")
    ap.add_argument("--outdir", default="/root/autodl-fs/lyy/task4_ehr_ct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["catboost", "lightgbm", "xgboost"], default="catboost")
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _seed(args.seed)

    df = _load_ct488_fused(args.finaltext, args.mapping, args.manifest, args.image_features)
    img_cols = _img_feat_cols(df)
    if len(img_cols) != 512:
        raise ValueError(f"Expected 512 img_feat_* cols, got {len(img_cols)}")
    missing = [c for c in EHR_COLS_24 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing EHR cols: {missing}")

    feat_cols = EHR_COLS_24 + img_cols
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy()

    idx_tr = df["split"].values == "train"
    idx_va = df["split"].values == "val"
    idx_te = df["split"].values == "test"

    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xva, yva = X[idx_va], y[idx_va]
    Xte, yte = X[idx_te], y[idx_te]

    # train-only impute + zscore (helps linear model; for GBDT it's optional but harmless here)
    med = _median_impute_fit(Xtr)
    Xtr = _median_impute_apply(Xtr, med)
    Xva = _median_impute_apply(Xva, med)
    Xte = _median_impute_apply(Xte, med)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    Xte = sc.transform(Xte)

    model = _make_model(args.model, args.seed)

    t0 = time.time()
    if args.model == "catboost":
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    elif args.model == "lightgbm":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    else:
        # Some xgboost sklearn API builds don't support early_stopping_rounds/callbacks.
        # We train a capped number of estimators for stability/reproducibility.
        model.fit(Xtr, ytr, verbose=False)
    train_s = time.time() - t0

    if args.calibrate != "none":
        # Calibrate on val only (keep test untouched). Use prefit model.
        cal = CalibratedClassifierCV(model, method=args.calibrate, cv="prefit")
        cal.fit(Xva, yva)
        pred_val = cal.predict_proba(Xva)[:, 1]
        pred_test = cal.predict_proba(Xte)[:, 1]
        calibrated = True
    else:
        pred_val = model.predict_proba(Xva)[:, 1]
        pred_test = model.predict_proba(Xte)[:, 1]
        calibrated = False

    val_auc = float(roc_auc_score(yva, pred_val))
    test_auc = float(roc_auc_score(yte, pred_test))
    val_pr = float(average_precision_score(yva, pred_val))
    test_pr = float(average_precision_score(yte, pred_test))
    ci = _bootstrap_auc_ci(yte, pred_test, seed=args.seed, n_boot=int(args.n_boot))

    metrics = Metrics(
        model=args.model,
        calibrated=calibrated,
        val_roc_auc=val_auc,
        test_roc_auc=test_auc,
        val_pr_auc=val_pr,
        test_pr_auc=test_pr,
        test_auc_bootstrap_ci95=ci,
        n_train=int(len(ytr)),
        n_val=int(len(yva)),
        n_test=int(len(yte)),
        n_features=int(Xtr.shape[1]),
        seed=int(args.seed),
    )

    out = {
        **asdict(metrics),
        "train_seconds": train_s,
        "calibration_method": args.calibrate,
    }
    out_path = os.path.join(args.outdir, f"metrics_{args.model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Save test predictions for analysis
    pred_df = pd.DataFrame({"global_id": df.loc[idx_te, "global_id"].astype(str).values, "y_true": yte, "y_prob": pred_test})
    pred_csv = os.path.join(args.outdir, f"test_pred_{args.model}.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {out_path}")
    print(f"Saved test preds: {pred_csv}")


if __name__ == "__main__":
    main()

