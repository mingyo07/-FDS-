# -*- coding: utf-8 -*-
"""FDS 하이브리드 병렬 모델 — 단일 진입점

로우데이터 df(분할 parquet 형식, 라벨 불필요) → 최종 부정위험 확률(0~1)과 0~999 위험점수.
계획서 그림 9의 병렬 구조를 하나의 호출로 묶는다:

    메인 LGBM(+관계 피처)  ──┐
    FT-Transformer(딥러닝) ──┼─ 가중합(final_config.json) ─→ 위험점수
    이상탐지(IF+AE)        ──┘

사용:
    from final_predict import predict_final
    out = predict_final(df, model_dir="final_model")   # DataFrame(row_id, score, risk999, flag)

필요 파일(model_dir 안): final_config.json, ml_model_v2.pkl,
  graph_predict.py + graph_feats.pkl,
  dl_predict.py + ft_transformer_v2.pt + dl_preprocess.pkl,
  anom_predict.py + isolation_forest.pkl + denoising_autoencoder.pt + ae_scaler.pkl
  + anomaly_preprocessor.pkl + if/ae_score_normalization.pkl + anomaly_config.json
"""
import os, sys, json, gc, pickle
import numpy as np
import pandas as pd

AMOUNT_COLS = ['GVSPMN_AM','SFALMT_AM','DDCT_PONT_SM_AM','REMNDR_PONT_AM','CREAT_PONT_AM',
               'CREAT_SFALMT_AM','CREAT_GVSPMN_AM','HLTHINS_CALC_AM','F_AMT_PER_MIN']
UNIT_COL = 'VOUCH_UNTPC'
TARGET = 'INJSTUSE_AT'


def _read_like_train(df):
    df = df.copy()
    for c in df.columns:
        dt = str(df[c].dtype)
        if dt in ('string', 'large_string', 'object'):
            df[c] = df[c].astype(str).astype('category')
        elif dt.startswith(('Int', 'Float', 'boolean')):
            df[c] = df[c].astype('float64')
    return df


def _upper_categories(df):
    for c in [c for c in df.columns if str(df[c].dtype) == 'category']:
        s = df[c]
        mp = {x: str(x).strip().upper() for x in s.cat.categories}
        lv = sorted(set(mp.values()))
        pos = {v: i for i, v in enumerate(lv)}
        lut = np.array([pos[mp[x]] for x in s.cat.categories])
        cd = s.cat.codes.to_numpy()
        df[c] = pd.Categorical.from_codes(np.where(cd < 0, -1, lut[np.clip(cd, 0, None)]), categories=lv)
    return df


def _score_ml(df_raw, model_dir, gfeat):
    with open(f'{model_dir}/ml_model_v2.pkl', 'rb') as f:
        ML = pickle.load(f)
    X = _upper_categories(_read_like_train(df_raw))
    X = X[[c for c in X.columns if c not in set(ML['drop_cols']) | {TARGET}]].copy()
    for c in X.columns:
        if str(X[c].dtype) != 'category':
            X[c] = pd.to_numeric(X[c], errors='coerce').astype('float32')
    unit = pd.to_numeric(X[UNIT_COL], errors='coerce').replace(0, np.nan)
    for c in AMOUNT_COLS:
        if c in X.columns:
            X[f'REL_{c}'] = (pd.to_numeric(X[c], errors='coerce') / unit).astype('float32')
    for c in [c for c in gfeat.columns if c != 'row_id']:
        X[c] = pd.to_numeric(gfeat[c], errors='coerce').fillna(0).astype('float32').values
    for c, lv in ML['cat_levels'].items():
        X[c] = X[c].cat.set_categories(lv)
    X = X[ML['features']]
    ens = np.mean([m.predict_proba(X)[:, 1] for m in ML['models']], axis=0)
    del X; gc.collect()
    return ML['calibrator'].transform(ens)


def predict_final(df, model_dir='.', return_parts=False):
    sys.path.insert(0, os.path.abspath(model_dir))
    import graph_predict, dl_predict, anom_predict

    cfg = json.load(open(f'{model_dir}/final_config.json'))
    W = cfg['weights']; thr = cfg['threshold_operating']

    # 1) 관계 피처 (train 룩업 조회, 운영 = leave_one_out=False)
    gfeat = graph_predict.add_graph_feats(df, f'{model_dir}/graph_feats.pkl', leave_one_out=False)

    # 2) 세 축 병렬 스코어링
    s_ml = _score_ml(df, model_dir, gfeat)
    s_dl = dl_predict.predict_dl(df, f'{model_dir}/ft_transformer_v2.pt', f'{model_dir}/dl_preprocess.pkl')
    _, _, s_an = anom_predict.predict_anom(df, art_dir=model_dir, **cfg.get('anom_kwargs', {}))

    # 3) 가중 결합 → 위험점수
    score = W['ml'] * s_ml + W['dl'] * s_dl + W['anom'] * s_an
    out = pd.DataFrame({
        'row_id': df['row_id'].values if 'row_id' in df.columns else np.arange(len(df)),
        'score': score,
        'risk999': np.round(score * 999).astype(int),
        'flag': (score >= thr).astype(int),   # 1 = 조사 대상 후보 (운영 threshold 기준)
    })
    if return_parts:
        out['score_ml'], out['score_dl'], out['score_anom'] = s_ml, s_dl, s_an
    return out
