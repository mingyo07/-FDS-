# -*- coding: utf-8 -*-
"""ML 전용 추론 보조 코드 (신규 편집본).

주의: 이 파일은 원본 학습 코드가 아니다. 최종 하이브리드 final_predict.py에서
ML 전처리·점수 함수 3개를 원문 그대로 발췌하고 ML만 실행하는 함수를 추가한 것이다.
최종 v2 ml_model_v2.pkl과 train 기반 graph_feats.pkl이 필요하다.
"""
import os
import gc
import pickle
import numpy as np
import pandas as pd
from graph_predict import add_graph_feats

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

def predict_ml_v2(df_raw, model_dir='.'):
    """ML 축만 계산한 보정확률 및 0~999 환산점수 (전체 하이브리드 아님)."""
    from pathlib import Path
    d=Path(model_dir)
    gfeat=add_graph_feats(df_raw, str(d/'graph_feats.pkl'), leave_one_out=False)
    score=_score_ml(df_raw, str(d), gfeat)
    row_id=df_raw['row_id'].to_numpy() if 'row_id' in df_raw else np.arange(len(df_raw))
    return pd.DataFrame({'row_id':row_id, 'score_ml':score, 'risk999_ml':np.round(score*999).astype(int)})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='최종 ML v2만 추론 (DL/이상탐지 미실행)')
    p.add_argument('--parquet',required=True)
    p.add_argument('--model-dir',default='.')
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--out',default='ml_only_scores.csv')
    a=p.parse_args()
    df=pd.read_parquet(a.parquet)
    if a.limit is not None:df=df.head(a.limit)
    out=predict_ml_v2(df,model_dir=a.model_dir)
    out.to_csv(a.out,index=False)
    print(f'{len(out):,}건 ML 단독 점수 저장: {a.out}')
