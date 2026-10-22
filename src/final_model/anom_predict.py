# -*- coding: utf-8 -*-
"""이상탐지 축 적용 함수 — 로우데이터 df → score_if / score_ae / score_anom

남민교 산출물(anomaly_axis_final/)의 아티팩트로부터 PL이 재구성한 재현 스크립트.
AE 구조는 denoising_autoencoder.pt의 state_dict에서 복원:
  encoder: 58→116→(act)→58→(act)→16 / decoder: 16→58→(act)→116→(act)→58
활성함수·IF 점수 방향 등 세부는 validate()가 정답 CSV와 대조해 자동 판별한다.
"""
import json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def _load_pkl(path):
    """joblib.dump(sklearn 표준)와 pickle.dump 모두 지원."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


class DAE(nn.Module):
    def __init__(self, input_dim=58, latent_dim=16, act="relu"):
        super().__init__()
        A = _ACTS[act]
        self.encoder = nn.Sequential(nn.Linear(input_dim, 116), A(), nn.Linear(116, 58), A(),
                                     nn.Linear(58, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 58), A(), nn.Linear(58, 116), A(),
                                     nn.Linear(116, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))


def _get(d, *names, required=True):
    for n in names:
        if n in d: return d[n]
    if required:
        raise KeyError(f"preprocessor에 {names} 키 없음 — 실제 키: {list(d.keys())}")
    return None


class AnomAxis:
    def __init__(self, art_dir=".", act="relu", if_raw_sign=-1, ae_topk_frac=0.10,
                 norm_fn="sigmoid"):
        self.dir = art_dir
        self.prep = _load_pkl(f"{art_dir}/anomaly_preprocessor.pkl")
        self.num_cols = _get(self.prep, "numeric_features", "num_cols")
        self.bin_cols = _get(self.prep, "binary_features", "bin_cols")
        self.medians  = _get(self.prep, "numeric_fill_values", "medians", "num_medians")
        self.bin_maps = _get(self.prep, "binary_maps", "bin_maps", "binary_mapping", required=False)
        self.feat_order = _get(self.prep, "feature_names", required=False)
        self.iforest  = _load_pkl(f"{art_dir}/isolation_forest.pkl")
        self.scaler   = _load_pkl(f"{art_dir}/ae_scaler.pkl")
        self.if_norm  = _load_pkl(f"{art_dir}/if_score_normalization.pkl")
        self.ae_norm  = _load_pkl(f"{art_dir}/ae_score_normalization.pkl")
        cfg = json.load(open(f"{art_dir}/anomaly_config.json"))
        self.w_if, self.w_ae = cfg["if_weight"], cfg["ae_weight"]
        ckpt = torch.load(f"{art_dir}/denoising_autoencoder.pt", map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.ae = DAE(58, 16, act=act); self.ae.load_state_dict(sd); self.ae.eval()
        self.act, self.if_raw_sign, self.ae_topk_frac, self.norm_fn = act, if_raw_sign, ae_topk_frac, norm_fn

    # ---- 전처리 ----
    def _matrix(self, df):
        Xn = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
        med = pd.Series(self.medians) if not isinstance(self.medians, pd.Series) else self.medians
        Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(med[self.num_cols])
        Xb = pd.DataFrame(index=df.index)
        for c in self.bin_cols:
            s = df[c].astype(str)
            mp = self.bin_maps.get(c) if self.bin_maps else None
            if mp:
                v = s.map(mp)
                if v.isna().all():                    # 키 타입/공백 차이 대비
                    mp2 = {str(k).strip(): val for k, val in mp.items()}
                    v = s.str.strip().map(mp2)
                Xb[c + "__bin"] = v.fillna(-1).astype(float)
            else:
                vals = sorted(s.unique())[:2]
                Xb[c + "__bin"] = s.map({v: i for i, v in enumerate(vals)}).fillna(-1).astype(float)
        M = pd.concat([Xn, Xb], axis=1)
        if self.feat_order:                            # 학습 당시 컬럼 순서로 정렬
            missing = [c for c in self.feat_order if c not in M.columns]
            assert not missing, f"feature_names에 있는데 생성 못한 컬럼: {missing}"
            M = M[self.feat_order]
        return M.astype("float32").values              # V2 전처리 사양: float32 통일

    def _norm(self, raw, nd):
        z = (raw - nd["median"]) / max(nd["iqr"], 1e-12)
        if self.norm_fn == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        if self.norm_fn == "clip01":   # (z+3)/6 클리핑 방식 후보
            return np.clip((z + 3) / 6, 0, 1)
        raise ValueError(self.norm_fn)

    # ---- 점수 ----
    def raw_if_scores(self, X, mode="neg_score_samples"):
        if mode == "neg_score_samples":  return -self.iforest.score_samples(X)
        if mode == "score_samples":      return self.iforest.score_samples(X)
        if mode == "neg_decision":       return -self.iforest.decision_function(X)
        if mode == "decision":           return self.iforest.decision_function(X)
        raise ValueError(mode)

    def raw_ae_errors(self, X, k=6, agg="topk_mean", batch=8192):
        Xs = self.scaler.transform(X)
        errs = []
        with torch.no_grad():
            for i in range(0, len(Xs), batch):
                xb = torch.tensor(np.asarray(Xs[i:i+batch], dtype=np.float32))
                rec = self.ae(xb).numpy()
                e2 = (rec - Xs[i:i+batch]) ** 2
                if agg == "topk_mean":
                    kk = max(1, k)
                    errs.append(np.sort(e2, axis=1)[:, -kk:].mean(axis=1))
                elif agg == "mean":
                    errs.append(e2.mean(axis=1))
                elif agg == "max":
                    errs.append(e2.max(axis=1))
        return np.concatenate(errs)

    def score(self, df, if_mode="neg_score_samples", ae_k=6, ae_agg="topk_mean"):
        X = self._matrix(df)
        s_if = self._norm(self.raw_if_scores(X, if_mode), self.if_norm)
        s_ae = self._norm(self.raw_ae_errors(X, ae_k, ae_agg), self.ae_norm)
        return s_if, s_ae, self.w_if * s_if + self.w_ae * s_ae


def predict_anom(df, art_dir=".", **kw):
    """로우데이터 df(분할 parquet 형식) -> (score_if, score_ae, score_anom)"""
    return AnomAxis(art_dir, **kw).score(df)


# ---- 재현 검증: IF/AE 분리 탐색 + 순위보존 판정 ----
def validate(valid_parquet, scores_csv, art_dir=".", n=5000, tol=1e-3):
    df = pd.read_parquet(valid_parquet).head(n)
    ans = pd.read_csv(scores_csv, encoding="utf-8-sig").head(n)
    assert (df["row_id"].values == ans["row_id"].values).all(), "row_id 순서 불일치"

    result = {}
    def spear(a, b):
        ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
        return float(np.corrcoef(ra, rb)[0, 1])

    # ── IF: raw 변형 × 정규화 방식 탐색 (활성함수와 무관) ──
    ax = AnomAxis(art_dir)
    X = ax._matrix(df)
    best_if = None
    for mode in ("neg_score_samples", "score_samples", "neg_decision", "decision"):
        raw = ax.raw_if_scores(X, mode)
        for norm_fn in ("sigmoid", "clip01"):
            ax.norm_fn = norm_fn
            s = ax._norm(raw, ax.if_norm)
            e = float(np.abs(s - ans["score_if"].values).max())
            if best_if is None or e < best_if[0]: best_if = (e, mode, norm_fn, s)
    e, mode, norm_fn, s_if = best_if
    rc = spear(s_if, ans["score_if"].values)
    result["if"] = dict(mode=mode, norm=norm_fn, max_err=round(e, 5), spearman=round(rc, 5))
    print(f"[IF] 최선: mode={mode} norm={norm_fn} | 최대오차 {e:.2e} | 순위상관 {rc:.5f} "
          f"{'✅완전재현' if e < tol else ('✅순위보존(근사)' if rc > 0.999 else '❌')}")

    # ── AE: 활성함수 × k(5/6) × 집계 × 정규화 탐색 ──
    best_ae = None
    for act in _ACTS:
        try:
            axa = AnomAxis(art_dir, act=act)
        except Exception:
            continue
        for k, agg in ((6, "topk_mean"), (5, "topk_mean"), (0, "mean"), (0, "max")):
            raw = axa.raw_ae_errors(X, k=k, agg=agg)
            for norm_fn in ("sigmoid", "clip01"):
                axa.norm_fn = norm_fn
                s = axa._norm(raw, axa.ae_norm)
                e2 = float(np.abs(s - ans["score_ae"].values).max())
                if best_ae is None or e2 < best_ae[0]: best_ae = (e2, act, k, agg, norm_fn, s)
    e2, act, k, agg, norm_fn2, s_ae = best_ae
    rc2 = spear(s_ae, ans["score_ae"].values)
    result["ae"] = dict(act=act, k=k, agg=agg, norm=norm_fn2, max_err=round(e2, 5), spearman=round(rc2, 5))
    print(f"[AE] 최선: act={act} k={k} agg={agg} norm={norm_fn2} | 최대오차 {e2:.2e} | 순위상관 {rc2:.5f} "
          f"{'✅완전재현' if e2 < tol else ('✅순위보존(근사)' if rc2 > 0.999 else '❌')}")

    # ── 종합 score_anom 대조 ──
    prep_cfg = json.load(open(f"{art_dir}/anomaly_config.json"))
    s_an = prep_cfg["if_weight"] * s_if + prep_cfg["ae_weight"] * s_ae
    e3 = float(np.abs(s_an - ans["score_anom"].values).max())
    rc3 = spear(s_an, ans["score_anom"].values)
    result["anom"] = dict(max_err=round(e3, 5), spearman=round(rc3, 5))
    print(f"[ANOM] 결합 최대오차 {e3:.2e} | 순위상관 {rc3:.5f}")
    verdict = ("완전재현" if max(e, e2) < tol
               else "근사재현(순위보존)" if min(rc, rc2, rc3) > 0.999
               else "재현실패")
    result["verdict"] = verdict
    print("종합 판정:", verdict)
    return result
