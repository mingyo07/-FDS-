# -*- coding: utf-8 -*-
"""관계 네트워크 피처 적용 함수 — 운영 재현용 (v2)

artifact(graph_feats.pkl)는 train 결제만으로 구성한 노드 룩업 테이블이다.
축: 이용자 / 제공인력 / (이용자,인력) 조합 / 기관(근사) / 지역 / 연결요소 / 커뮤니티

기관 근사 키 = PRINST_TY|PRINST_LOCPLC_SIGNGU_CODE — 정확한 기관 식별이 아님.
신규 노드는 기본값 + IS_NEW_* 플래그로 처리한다.
운영 시 룩업 테이블은 최근 이력 배치로 주기 재계산한다.
"""
import pickle
import numpy as np
import pandas as pd


def add_graph_feats(df, artifact_path, leave_one_out=False):
    """df에 G_* 관계 피처를 붙여 반환.

    df            : TRGTER_ID, PRHNF_ID, PRINST_TY, PRINST_LOCPLC_SIGNGU_CODE 포함
    artifact_path : graph_feats.pkl 경로
    leave_one_out : True면 라벨 피처에서 자기 조합 몫 제외 (train 행 전용)
    """
    with open(artifact_path, 'rb') as f:
        art = pickle.load(f)
    T, P, E = art['trgter'], art['prhnf'], art['pair']
    I, R = art['inst'], art['region']
    dv, gp = art['defaults'], art['global_pos']

    o = df[['TRGTER_ID', 'PRHNF_ID']].copy()
    o['_INST'] = df['PRINST_TY'].astype(str) + '|' + df['PRINST_LOCPLC_SIGNGU_CODE'].astype(str)
    o['_REGION'] = df['PRINST_LOCPLC_SIGNGU_CODE'].astype(str)

    o['IS_NEW_TRGTER'] = (~o.TRGTER_ID.isin(set(T.TRGTER_ID))).astype(np.int8)
    o['IS_NEW_PRHNF'] = (~o.PRHNF_ID.isin(set(P.PRHNF_ID))).astype(np.int8)
    o['IS_NEW_INST'] = (~o._INST.isin(set(I._INST))).astype(np.int8)

    o = o.merge(T, on='TRGTER_ID', how='left')
    o = o.merge(P, on='PRHNF_ID', how='left', suffixes=('_t', '_p'))
    o = o.merge(E, on=['TRGTER_ID', 'PRHNF_ID'], how='left')
    o = o.merge(I, on='_INST', how='left')
    o = o.merge(R, on='_REGION', how='left')

    pc = o['pair_cnt'].fillna(0.0)
    pp = o['pair_pos'].fillna(0.0)
    tc, tp_ = o['t_cnt'], o['t_pos']
    pn, po = o['p_cnt'], o['p_pos']
    cc, cp_ = o['c_cnt_p'].fillna(o['c_cnt_t']), o['c_pos_p'].fillna(o['c_pos_t'])
    mc, mp_ = o['m_cnt_p'].fillna(o['m_cnt_t']), o['m_pos_p'].fillna(o['m_pos_t'])
    ic, ip_ = o['i_cnt'], o['i_pos']
    rc, rp_ = o['r_cnt'], o['r_pos']

    if leave_one_out:
        tc, tp_ = tc - pc, tp_ - pp
        pn, po = pn - pc, po - pp
        cc, cp_ = cc - pc, cp_ - pp
        mc, mp_ = mc - pc, mp_ - pp
        ic, ip_ = ic - pc, ip_ - pp
        rc, rp_ = rc - pc, rp_ - pp

    def safe(num, den):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    g = pd.DataFrame(index=o.index)
    g['G_TRGTER_DEG'] = o['t_deg']
    g['G_PRHNF_DEG'] = o['p_deg']
    g['G_PAIR_CNT'] = o['pair_cnt']
    g['G_PAIR_SHARE_TRGTER'] = safe(o['pair_cnt'], o['t_cnt'])
    g['G_PAIR_SHARE_PRHNF'] = safe(o['pair_cnt'], o['p_cnt'])
    g['G_PRHNF_HHI'] = o['p_hhi']
    g['G_COMP_SIZE'] = o['comp_size_p'].fillna(o['comp_size_t'])
    g['G_PRHNF_TIME_OVERLAP'] = o['p_time_ov']
    g['G_PRHNF_SAME_TIME_PAY'] = o['p_same_pay']
    g['G_INST_PRHNF_CNT'] = o['i_prhnf']
    g['G_INST_TRGTER_CNT'] = o['i_trgter']
    g['G_REGION_PRHNF_CNT'] = o['r_prhnf']
    g['G_REGION_TRGTER_CNT'] = o['r_trgter']
    g['G_PRHNF_EIGCENT'] = o['p_eig']
    g['G_TRGTER_EIGCENT'] = o['t_eig']
    g['G_COMMUNITY_SIZE'] = o['com_size_p'].fillna(o['com_size_t'])
    g['G_PRHNF_FRAUD_NBR'] = safe(po, pn)
    g['G_TRGTER_FRAUD_NBR'] = safe(tp_, tc)
    g['G_COMP_FRAUD_RATE'] = safe(cp_, cc)
    g['G_COMMUNITY_FRAUD_RATE'] = safe(mp_, mc)
    g['G_INST_FRAUD_NBR'] = safe(ip_, ic)
    g['G_REGION_FRAUD_NBR'] = safe(rp_, rc)
    g['IS_NEW_TRGTER'] = o['IS_NEW_TRGTER']
    g['IS_NEW_PRHNF'] = o['IS_NEW_PRHNF']
    g['IS_NEW_INST'] = o['IS_NEW_INST']

    for c in art['feature_order']:
        if c.startswith('IS_NEW'):
            continue
        g[c] = g[c].fillna(dv[c]).astype(np.float32)

    g = g[art['feature_order']]
    head = df[['row_id']].reset_index(drop=True) if 'row_id' in df.columns         else pd.DataFrame(index=range(len(df)))
    return pd.concat([head, g.reset_index(drop=True)], axis=1)
