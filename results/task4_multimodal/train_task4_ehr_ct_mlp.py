from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class NPDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x)).squeeze(1)


@dataclass(frozen=True)
class Metrics:
    method: str
    hidden: list[int]
    dropout: float
    best_val_roc_auc: float
    test_roc_auc: float
    best_val_pr_auc: float
    test_pr_auc: float
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    seed: int


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        xb = xb.to(device)
        p = torch.sigmoid(model(xb))
        out.append(p.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finaltext", default="/root/autodl-fs/lyy/finaltext.xlsx")
    ap.add_argument("--mapping", default="/root/autodl-fs/lyy/ehr_ct_id_mapping.csv")
    ap.add_argument("--manifest", default="/root/autodl-fs/lyy/data_split_manifest_ct488.csv")
    ap.add_argument("--image-features", default="/root/autodl-fs/lyy/unified_runs/image_features_ct488.csv")
    ap.add_argument("--outdir", default="/root/autodl-fs/lyy/task4_ehr_ct")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--time-limit", type=int, default=1800)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    _seed(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    df = _load_ct488_fused(args.finaltext, args.mapping, args.manifest, args.image_features)
    img_cols = _img_feat_cols(df)
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

    med = _median_impute_fit(Xtr)
    Xtr = _median_impute_apply(Xtr, med)
    Xva = _median_impute_apply(Xva, med)
    Xte = _median_impute_apply(Xte, med)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xva = sc.transform(Xva).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)

    hidden = [int(x.strip()) for x in args.hidden.split(",") if x.strip()]
    model = MLP(in_dim=int(Xtr.shape[1]), hidden=hidden, dropout=float(args.dropout)).to(device)

    # imbalance
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    dl_tr = DataLoader(NPDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_va = DataLoader(NPDataset(Xva, yva), batch_size=max(64, args.batch_size), shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(NPDataset(Xte, yte), batch_size=max(64, args.batch_size), shuffle=False, num_workers=args.num_workers)

    best_auc = -1.0
    best_pr = -1.0
    best_state = None
    no_imp = 0
    t0 = time.time()

    for _epoch in range(1, args.epochs + 1):
        if time.time() - t0 > args.time_limit:
            break
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        p_val = _predict(model, dl_va, device)
        auc = float(roc_auc_score(yva, p_val))
        pr = float(average_precision_score(yva, p_val))
        sched.step(auc)

        if auc > best_auc + 1e-6:
            best_auc = auc
            best_pr = pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint")
    model.load_state_dict(best_state)

    p_val = _predict(model, dl_va, device)
    p_test = _predict(model, dl_te, device)
    test_auc = float(roc_auc_score(yte, p_test))
    test_pr = float(average_precision_score(yte, p_test))

    metrics = Metrics(
        method="ehr24 + img512 -> MLP (median-impute+zscore, train-only)",
        hidden=hidden,
        dropout=float(args.dropout),
        best_val_roc_auc=float(best_auc),
        test_roc_auc=float(test_auc),
        best_val_pr_auc=float(best_pr),
        test_pr_auc=float(test_pr),
        n_train=int(len(ytr)),
        n_val=int(len(yva)),
        n_test=int(len(yte)),
        n_features=int(Xtr.shape[1]),
        seed=int(args.seed),
    )

    metrics_path = os.path.join(args.outdir, "metrics_mlp.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)

    val_pred = pd.DataFrame({"global_id": df.loc[idx_va, "global_id"].astype(str).values, "y_true": yva, "y_prob": p_val})
    test_pred = pd.DataFrame({"global_id": df.loc[idx_te, "global_id"].astype(str).values, "y_true": yte, "y_prob": p_test})
    val_pred.to_csv(os.path.join(args.outdir, "val_pred_mlp.csv"), index=False, encoding="utf-8-sig")
    test_pred.to_csv(os.path.join(args.outdir, "test_pred_mlp.csv"), index=False, encoding="utf-8-sig")

    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

