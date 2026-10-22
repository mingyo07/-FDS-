import argparse
import numpy as np
import pandas as pd

# 원천 274컬럼 -> 최종 110컬럼
# drop_columns -> add_missing_flags -> impute -> add_features -> finalize 순서로 실행

# 결측률 80% 이상 (예외 유지 3개 제외)
DROP_MISSING_80 = [
    "ALL_INSRNC_UN_SBSCRB_AT", "BHF_SE_AT", "CHIEF_PRHNF_NMPR_CO",
    "COFFM_CO", "ENCPT_IHIDNUM", "ETC_FCLTY_CO",
    "EXCLNC_APPN_GRAD_CODE", "EXCLNC_INSTT_AT", "GRNT_SE",
    "HSHLD_INCOME_AM", "HSHLD_TY", "INCOME_STDR",
    "ISLNDS_RMTPLC_AT", "MAIN_ACT_SIGNGU_CODE", "MNGR_CAREER_YYCO",
    "MT_SALARY_AM", "MT_WORK_TIME", "NWNBB_MOTHR_SE",
    "OFFM_CO", "PARTCPTN_PRHNF_NMPR_CO", "PRHNF_DEATH_CNFIRM_RESULT_SE",
    "PRHNF_DEATH_PRSM_DE", "PRHNF_FRST_DEATH_PRSM_DE", "PRHNF_FRST_HSPTLZ_DE",
    "PRHNF_HSPTLZ_DCNT", "PRHNF_IHIDNUM", "PRHNF_IMMI_DE",
    "PRHNF_IMMI_DT", "PRHNF_IMMI_SE", "PRHNF_LAST_STTUS_SE",
    "PRHNF_NATION_SE", "PRHNF_RCPER_BEGIN_DE", "PRHNF_RCPER_END_DE",
    "PRHNF_TOT_HSPTLZ_DCNT", "PRINST_BIZRNO", "PRINST_ID",
    "PRINST_TY3", "PRINST_TY4", "REAL_PYMNT_MT_SALARY_AM",
    "REPRATN_INSRNC_AT", "RL_MT_WORK_TIME", "RPRSNTV_CAREER_YYCO",
    "RPRSNTV_IHIDNUM", "SETLE_FNLT_CODE", "TIME_ACCTO_UNTPC",
    "TRGTER_DEATH_CNFIRM_RESULT_SE", "TRGTER_DEATH_PRSM_DE", "TRGTER_FRST_DEATH_PRSM_DE",
    "TRGTER_FRST_HSPTLZ_DE", "TRGTER_HSPTLZ_DCNT", "TRGTER_IHIDNUM",
    "TRGTER_IMMI_DE", "TRGTER_IMMI_DT", "TRGTER_IMMI_SE",
    "TRGTER_INORG_HSPTLZ_BEGIN_DE", "TRGTER_LAST_STTUS_SE", "TRGTER_NATION_SE",
    "TRGTER_RCPER_BEGIN_DE", "TRGTER_RCPER_END_DE", "TRGTER_TOT_HSPTLZ_DCNT",
    "TRRM_CO", "WRM_CO",
]

# 결측률 60~80% 중 고유값/중요도로 제외
DROP_MISSING_60_80 = [
    "CREAT_VOUCH_NMRS", "CRTAX_PAY_AT", "EMRGNCY_SPORT_AT",
    "EXCLNC_APPN_YEAR", "FAMILY_SPORT_AT", "INSTT_CODE",
    "PRTC_TRGTER_REQST_RQISIT_SE", "VHCLE_POSESN_SE", "VOUCH_KND_SE",
    "VOUCH_QY",
]

# DDIF 중복. 시간창별 대표 6개만 남김
DROP_DDIF = [
    "DPR_DBS_DDIF_10MIN_CNT", "DPR_DBS_DDIF_1MIN_CNT",
    "DPR_DBS_DDIF_3MIN_CNT", "DPR_DBS_TBT_DDIF_10MIN_CNT",
    "DPR_DBS_TBT_DDIF_1MIN_CNT", "DPR_DBS_TBT_DDIF_3MIN_CNT",
    "DPR_DBT_DDIF_10MIN_CNT", "DPR_DBT_DDIF_1MIN_CNT",
    "DPR_DBT_DDIF_3MIN_CNT", "DPR_DTR_DBS_DDIF_10MIN_CNT",
    "DPR_DTR_DBS_DDIF_1MIN_CNT", "DPR_DTR_DBS_DDIF_3MIN_CNT",
    "DPR_DTR_DBS_TBT_DDIF_10MIN_CNT", "DPR_DTR_DBS_TBT_DDIF_1MIN_CNT",
    "DPR_DTR_DBS_TBT_DDIF_3MIN_CNT", "DPR_DTR_DBT_DDIF_10MIN_CNT",
    "DPR_DTR_DBT_DDIF_1MIN_CNT", "DPR_DTR_DBT_DDIF_3MIN_CNT",
    "DPR_DTR_DDIF_10MIN_CNT", "DPR_DTR_DDIF_1MIN_CNT",
    "DPR_DTR_DDIF_3MIN_CNT", "DPR_DTR_TBS_DDIF_10MIN_CNT",
    "DPR_DTR_TBS_DDIF_1MIN_CNT", "DPR_DTR_TBS_DDIF_3MIN_CNT",
    "DPR_DTR_TBT_DDIF_10MIN_CNT", "DPR_DTR_TBT_DDIF_1MIN_CNT",
    "DPR_DTR_TBT_DDIF_3MIN_CNT", "DPR_TBS_DDIF_10MIN_CNT",
    "DPR_TBS_DDIF_1MIN_CNT", "DPR_TBS_DDIF_3MIN_CNT",
    "DPR_TBT_DDIF_10MIN_CNT", "DPR_TBT_DDIF_1MIN_CNT",
    "DPR_TBT_DDIF_3MIN_CNT", "DPR_TTR_DBS_DDIF_10MIN_CNT",
    "DPR_TTR_DBS_DDIF_1MIN_CNT", "DPR_TTR_DBS_DDIF_3MIN_CNT",
    "DPR_TTR_DBS_TBT_DDIF_10MIN_CNT", "DPR_TTR_DBS_TBT_DDIF_1MIN_CNT",
    "DPR_TTR_DBS_TBT_DDIF_3MIN_CNT", "DPR_TTR_DBT_DDIF_10MIN_CNT",
    "DPR_TTR_DBT_DDIF_1MIN_CNT", "DPR_TTR_DBT_DDIF_3MIN_CNT",
    "DPR_TTR_DDIF_10MIN_CNT", "DPR_TTR_DDIF_1MIN_CNT",
    "DPR_TTR_DDIF_3MIN_CNT", "DPR_TTR_TBS_DDIF_10MIN_CNT",
    "DPR_TTR_TBS_DDIF_1MIN_CNT", "DPR_TTR_TBS_DDIF_3MIN_CNT",
    "DPR_TTR_TBT_DDIF_10MIN_CNT", "DPR_TTR_TBT_DDIF_1MIN_CNT",
    "DPR_TTR_TBT_DDIF_3MIN_CNT", "DTR_DBS_DDIF_10MIN_CNT",
    "DTR_DBS_DDIF_1MIN_CNT", "DTR_DBS_DDIF_3MIN_CNT",
    "DTR_DBS_TBT_DDIF_10MIN_CNT", "DTR_DBS_TBT_DDIF_1MIN_CNT",
    "DTR_DBS_TBT_DDIF_3MIN_CNT", "DTR_DBT_DDIF_10MIN_CNT",
    "DTR_DBT_DDIF_1MIN_CNT", "DTR_DBT_DDIF_3MIN_CNT",
    "DTR_TBS_DDIF_10MIN_CNT", "DTR_TBS_DDIF_1MIN_CNT",
    "DTR_TBS_DDIF_3MIN_CNT", "DTR_TBT_DDIF_10MIN_CNT",
    "DTR_TBT_DDIF_1MIN_CNT", "DTR_TBT_DDIF_3MIN_CNT",
    "DTR_TPR_DBS_DDIF_10MIN_CNT", "DTR_TPR_DBS_DDIF_1MIN_CNT",
    "DTR_TPR_DBS_DDIF_3MIN_CNT", "DTR_TPR_DBS_TBT_DDIF_10MIN_CNT",
    "DTR_TPR_DBS_TBT_DDIF_1MIN_CNT", "DTR_TPR_DBS_TBT_DDIF_3MIN_CNT",
    "DTR_TPR_DBT_DDIF_10MIN_CNT", "DTR_TPR_DBT_DDIF_1MIN_CNT",
    "DTR_TPR_DBT_DDIF_3MIN_CNT", "DTR_TPR_DDIF_10MIN_CNT",
    "DTR_TPR_DDIF_1MIN_CNT", "DTR_TPR_DDIF_3MIN_CNT",
    "DTR_TPR_TBS_DDIF_10MIN_CNT", "DTR_TPR_TBS_DDIF_1MIN_CNT",
    "DTR_TPR_TBS_DDIF_3MIN_CNT", "DTR_TPR_TBT_DDIF_10MIN_CNT",
    "DTR_TPR_TBT_DDIF_1MIN_CNT", "DTR_TPR_TBT_DDIF_3MIN_CNT",
]

# 전 행 동일값
DROP_CONST = [
    "BUDGET_SE", "DPR_1MTH_1DAY_AVG_CNT",
    "EMPLYM_STLE_SE", "EXCO_SPORT_RESN_SE",
    "HKNRS_NEED_SVC_1_AT", "HKNRS_NEED_SVC_2_AT",
    "HKNRS_NEED_SVC_3_AT", "INSTT_TY",
    "ODSN_SE", "PRHNF_INORG_HSPTLZ_BEGIN_DE",
    "SVC_STSFDG_CODE", "TROBL_JDGMNT_DE",
    "TRSPMN_PYMNT_AT",
]

# 중복/누수/메타 컬럼
DROP_ETC = [
    "TRMNL_SE_CODE", "SVC_BEGIN_DT_1",
    "SVC_END_DT_1", "USE_RATE",
    "CRTFC_NO_SETLE_AT", "NTWK_CANCL_AT",
    "PRHNF_INCOME_AM", "ABNM_SETLE_AT",
    "INJSTUSE_CHCK_SE",
]

# DDIF 유지분
KEEP_DDIF = [
    "DPR_DDIF_1MIN_CNT", "DPR_DDIF_3MIN_CNT",
    "DPR_DDIF_10MIN_CNT", "DTR_DDIF_1MIN_CNT",
    "DTR_DDIF_3MIN_CNT", "DTR_DDIF_10MIN_CNT",
]

# 최종 컬럼 순서
FINAL_COLS = [
    "DELNG_SN", "CONFM_NO", "BNDE_SETLE_SE",
    "BNDE_SETLE_RESN_SE", "BSNS_SE_ID", "BTYP_ID",
    "SVC_TY_ID", "SVC_KND_SM_CO", "HLDY_SE",
    "NIGHT_SE", "ADIT_SMMNTN_TIME", "DDCT_PONT_SM_AM",
    "DDCT_VOUCH_NMRS_SM", "GVSPMN_AM", "SFALMT_AM",
    "REMNDR_PONT_AM", "REMNDR_VOUCH_NMRS", "PRHNF_NMPR_CO",
    "CNCL_SE_CODE", "CONFM_CANCL_AT", "PUCHAS_CANCL_AT",
    "TRGTER_GRAD_SE", "TRGTER_SIGNGU_CODE", "CREAT_PONT_AM",
    "CREAT_SFALMT_AM", "CREAT_GVSPMN_AM", "SIGNGU_CODE",
    "SRC_TRGTER_GRAD_SE", "INCOME_SE", "HLTHINS_CALC_AM",
    "VOUCH_UNTPC", "EXCPTMAN_SE", "EXCPT_REQST_DE",
    "TROBL_GRAD_SE", "NATIVE_FRGNR_SE", "DEPT_CODE",
    "TROBL_TY", "MBHSHD_CO", "CPR_CHARCT_SE",
    "PRINST_TY", "PRINST_TY2", "PRINST_LOCPLC_SIGNGU_CODE",
    "PRINST_LOCPLC_CTPRVN_CODE", "ESTB_DIV_CODE", "DELNG_DT_BSNDE_AT",
    "SVC_BEGIN_DT_BSNDE_AT", "SALARY_TRGET_AT", "TRGTER_LTR_DPLCT_AT",
    "PRHNF_LTR_DPLCT_AT", "DELNG_DT_CLS", "DELNG_DT_MIDTA_AT",
    "DELNG_DT_MIDTB_AT", "SVC_BEGIN_DT_MIDTA_AT", "SVC_BEGIN_DT_MIDTB_AT",
    "TRGTER_AGE", "TRGTER_AGE_CLS", "TRGTER_SEXDSTN",
    "PRHNF_AGE", "PRHNF_AGE_CLS", "PRHNF_SEXDSTN",
    "SVC_BEGIN_DT_CLS", "SVC_BEGIN_AFTER_DEATH_AT", "IMMI_AT",
    "INORG_HSPTLZ_AT", "PRHNF_HSPTLZ_AT", "DPR_TODAY_CNT",
    "DTR_TODAY_CNT", "DPR_DDT_CLS_CNT", "DTR_DDT_CLS_CNT",
    "DPR_DDIF_1MIN_CNT", "DPR_DDIF_3MIN_CNT", "DPR_DDIF_10MIN_CNT",
    "DELNG_MONTH", "DELNG_QUARTER", "SFALMT_RATE",
    "DPR_1MTH_CNT", "DPR_1MTH_DELDE_CNT", "DPR_1MTH_SVCTM",
    "DPR_1MTH_1DAY_SVCTM_AVG", "DPR_1MTH_1DAY_MAX_CNT", "DTR_DBS_1MTH_CNT",
    "DTR_DBS_1MTH_SVCTM", "DTR_DBS_1MTH_AVG_AMT", "DTR_DBS_1MTH_DIF_AVG_AMT",
    "INJSTUSE_AT", "CPR_CHARCT_SE_missing", "ESTB_DIV_CODE_missing",
    "F_hour_sin", "F_hour_cos", "F_dow_sin",
    "F_dow_cos", "F_DECL_MIN", "F_SVC_DUR_MIN",
    "F_DUR_GAP", "F_DUR_RATIO", "F_CROSS_DATE",
    "F_SUBSIDY_SHARE", "F_AMT_PER_MIN", "F_BURN_RATIO",
    "SVC_BEGIN_DT", "SVC_END_DT", "DELNG_DE",
    "PREDELNG_DE", "DELNG_DT", "TRGTER_ID",
    "PRHNF_ID", "SVC_PROVD_TIME_SM", "DTR_DDIF_1MIN_CNT",
    "DTR_DDIF_3MIN_CNT", "DTR_DDIF_10MIN_CNT",
]

# 최종 자료형
DTYPES = {
    "DELNG_SN": "int64",
    "CONFM_NO": "int64",
    "BNDE_SETLE_SE": "int64",
    "BNDE_SETLE_RESN_SE": "int64",
    "BSNS_SE_ID": "string",
    "BTYP_ID": "string",
    "SVC_TY_ID": "string",
    "SVC_KND_SM_CO": "int64",
    "HLDY_SE": "string",
    "NIGHT_SE": "string",
    "ADIT_SMMNTN_TIME": "int64",
    "DDCT_PONT_SM_AM": "int64",
    "DDCT_VOUCH_NMRS_SM": "int64",
    "GVSPMN_AM": "int64",
    "SFALMT_AM": "int64",
    "REMNDR_PONT_AM": "int64",
    "REMNDR_VOUCH_NMRS": "int64",
    "PRHNF_NMPR_CO": "int64",
    "CNCL_SE_CODE": "string",
    "CONFM_CANCL_AT": "string",
    "PUCHAS_CANCL_AT": "string",
    "TRGTER_GRAD_SE": "string",
    "TRGTER_SIGNGU_CODE": "float64",
    "CREAT_PONT_AM": "float64",
    "CREAT_SFALMT_AM": "float64",
    "CREAT_GVSPMN_AM": "float64",
    "SIGNGU_CODE": "float64",
    "SRC_TRGTER_GRAD_SE": "string",
    "INCOME_SE": "float64",
    "HLTHINS_CALC_AM": "float64",
    "VOUCH_UNTPC": "float64",
    "EXCPTMAN_SE": "int64",
    "EXCPT_REQST_DE": "float64",
    "TROBL_GRAD_SE": "float64",
    "NATIVE_FRGNR_SE": "int64",
    "DEPT_CODE": "string",
    "TROBL_TY": "str",
    "MBHSHD_CO": "float64",
    "CPR_CHARCT_SE": "float64",
    "PRINST_TY": "string",
    "PRINST_TY2": "float64",
    "PRINST_LOCPLC_SIGNGU_CODE": "string",
    "PRINST_LOCPLC_CTPRVN_CODE": "int64",
    "ESTB_DIV_CODE": "float64",
    "DELNG_DT_BSNDE_AT": "string",
    "SVC_BEGIN_DT_BSNDE_AT": "string",
    "SALARY_TRGET_AT": "int64",
    "TRGTER_LTR_DPLCT_AT": "string",
    "PRHNF_LTR_DPLCT_AT": "string",
    "DELNG_DT_CLS": "int64",
    "DELNG_DT_MIDTA_AT": "string",
    "DELNG_DT_MIDTB_AT": "string",
    "SVC_BEGIN_DT_MIDTA_AT": "string",
    "SVC_BEGIN_DT_MIDTB_AT": "string",
    "TRGTER_AGE": "int64",
    "TRGTER_AGE_CLS": "int64",
    "TRGTER_SEXDSTN": "string",
    "PRHNF_AGE": "int64",
    "PRHNF_AGE_CLS": "int64",
    "PRHNF_SEXDSTN": "string",
    "SVC_BEGIN_DT_CLS": "int64",
    "SVC_BEGIN_AFTER_DEATH_AT": "string",
    "IMMI_AT": "string",
    "INORG_HSPTLZ_AT": "string",
    "PRHNF_HSPTLZ_AT": "string",
    "DPR_TODAY_CNT": "float64",
    "DTR_TODAY_CNT": "float64",
    "DPR_DDT_CLS_CNT": "float64",
    "DTR_DDT_CLS_CNT": "float64",
    "DPR_DDIF_1MIN_CNT": "float64",
    "DPR_DDIF_3MIN_CNT": "float64",
    "DPR_DDIF_10MIN_CNT": "float64",
    "DELNG_MONTH": "int64",
    "DELNG_QUARTER": "string",
    "SFALMT_RATE": "float64",
    "DPR_1MTH_CNT": "float64",
    "DPR_1MTH_DELDE_CNT": "float64",
    "DPR_1MTH_SVCTM": "float64",
    "DPR_1MTH_1DAY_SVCTM_AVG": "float64",
    "DPR_1MTH_1DAY_MAX_CNT": "float64",
    "DTR_DBS_1MTH_CNT": "float64",
    "DTR_DBS_1MTH_SVCTM": "float64",
    "DTR_DBS_1MTH_AVG_AMT": "float64",
    "DTR_DBS_1MTH_DIF_AVG_AMT": "float64",
    "INJSTUSE_AT": "int64",
    "CPR_CHARCT_SE_missing": "int64",
    "ESTB_DIV_CODE_missing": "int64",
    "F_hour_sin": "float64",
    "F_hour_cos": "float64",
    "F_dow_sin": "float64",
    "F_dow_cos": "float64",
    "F_DECL_MIN": "int64",
    "F_SVC_DUR_MIN": "float64",
    "F_DUR_GAP": "float64",
    "F_DUR_RATIO": "float64",
    "F_CROSS_DATE": "Int8",
    "F_SUBSIDY_SHARE": "float64",
    "F_AMT_PER_MIN": "float64",
    "F_BURN_RATIO": "float64",
    "SVC_BEGIN_DT": "int64",
    "SVC_END_DT": "int64",
    "DELNG_DE": "int64",
    "PREDELNG_DE": "int64",
    "DELNG_DT": "string",
    "TRGTER_ID": "string",
    "PRHNF_ID": "string",
    "SVC_PROVD_TIME_SM": "int64",
    "DTR_DDIF_1MIN_CNT": "float64",
    "DTR_DDIF_3MIN_CNT": "float64",
    "DTR_DDIF_10MIN_CNT": "float64",
}

DROP_ALL = (DROP_MISSING_80 + DROP_MISSING_60_80 + DROP_DDIF
            + DROP_CONST + DROP_ETC)

# 최빈값 대치
MODE_FILL = {
    "CPR_CHARCT_SE": 20.0, "ESTB_DIV_CODE": 3.0, "INCOME_SE": 2.0,
    "SRC_TRGTER_GRAD_SE": "D001", "SIGNGU_CODE": 4490000,
    "TRGTER_SIGNGU_CODE": 4490000, "TRGTER_GRAD_SE": "D573",
    "PRINST_TY": "B", "PRINST_TY2": 7.0,
    "PRINST_LOCPLC_SIGNGU_CODE": "4490000", "PRINST_LOCPLC_CTPRVN_CODE": 8,
    "NATIVE_FRGNR_SE": 1, "DEPT_CODE": "4490000",
    "TRGTER_SEXDSTN": "m", "PRHNF_SEXDSTN": "f",
    "SVC_BEGIN_DT_BSNDE_AT": "Y", "DELNG_DT_BSNDE_AT": "Y",
    "TROBL_GRAD_SE": 1.0,
}

# 중앙값 대치
MEDIAN_FILL = {
    "HLTHINS_CALC_AM": 18980, "VOUCH_UNTPC": 14020,
    "CREAT_GVSPMN_AM": 2195900, "CREAT_PONT_AM": 2336000,
    "CREAT_SFALMT_AM": 20000, "DTR_DBS_1MTH_AVG_AMT": 66067,
    "DTR_DBS_1MTH_DIF_AVG_AMT": 920, "MBHSHD_CO": 3,
    "TRGTER_AGE": 40, "PRHNF_AGE": 59,
}

# 결측이 '해당 없음'을 뜻해서 0으로 채우는 컬럼
ZERO_FILL = ["SALARY_TRGET_AT", "EXCPTMAN_SE", "EXCPT_REQST_DE"]

# 결측이 '해당 거래 없음'을 뜻하는 집계 컬럼
AGG_ZERO_FILL = (
    ["DPR_TODAY_CNT", "DTR_TODAY_CNT", "DPR_DDT_CLS_CNT", "DTR_DDT_CLS_CNT"]
    + KEEP_DDIF
    + ["DPR_1MTH_CNT", "DPR_1MTH_DELDE_CNT", "DPR_1MTH_SVCTM",
       "DPR_1MTH_1DAY_SVCTM_AVG", "DPR_1MTH_1DAY_MAX_CNT",
       "DTR_DBS_1MTH_CNT", "DTR_DBS_1MTH_SVCTM"]
)

FLAG_COLS = ["CPR_CHARCT_SE", "ESTB_DIV_CODE"]   # 결측 플래그 생성 대상
LOWER_COLS = ["TRGTER_SEXDSTN", "PRHNF_SEXDSTN"] # 최종 저장은 소문자
AGE_BINS = [(-np.inf, 19, 1), (20, 29, 2), (30, 39, 3), (40, 49, 4),
            (50, 59, 5), (60, 69, 6), (70, np.inf, 7)]

DUR_RATIO_FALLBACK = 0.99166667   # F_DECL_MIN이 0인 5건
AMT_PER_MIN_FALLBACK = 276.8      # F_SVC_DUR_MIN이 0 이하인 5건

DT_FMT = "%Y%m%d%H%M%S"
LABEL = "INJSTUSE_AT"


def parse_dt(s):
    return pd.to_datetime(s.astype("string").str.strip(), format=DT_FMT, errors="coerce")


def drop_columns(df):
    # 삭제 178개
    missing = [c for c in DROP_ALL if c not in df.columns]
    if missing:
        raise KeyError("입력에 없는 삭제 대상 컬럼: %s" % missing)
    return df.drop(columns=DROP_ALL)


def add_missing_flags(df):
    # 대치하기 전에 결측 여부부터 남겨둔다
    for c in FLAG_COLS:
        df[c + "_missing"] = df[c].isna().astype("int64")
    return df


def age_class(age):
    # 나이 구간. 양끝 포함
    out = pd.Series(np.nan, index=age.index, dtype="float64")
    for lo, hi, code in AGE_BINS:
        out = out.mask(age.between(lo, hi), code)
    return out.astype("int64")


def fill_trobl_ty(df):
    # 장애유형. 같은 대상자의 앞선 거래값으로 채우고 남는 건 UNKNOWN
    df["TROBL_TY"] = df["TROBL_TY"].astype("string").replace("N", pd.NA)
    order = df.sort_values(["TRGTER_ID", "DELNG_DT"], kind="mergesort").index
    df["TROBL_TY"] = df.loc[order].groupby("TRGTER_ID")["TROBL_TY"].ffill().reindex(df.index)
    df["TROBL_TY"] = df["TROBL_TY"].fillna("UNKNOWN")
    return df


def impute(df):
    # 대치값이 인코딩 뒤에 정의되는 컬럼들
    df["NATIVE_FRGNR_SE"] = df["NATIVE_FRGNR_SE"].map({"H": 1, "F": 0}).astype("Float64")
    df["SALARY_TRGET_AT"] = df["SALARY_TRGET_AT"].map({"Y": 1, "N": 0}).astype("Float64")
    df[LABEL] = (df[LABEL].astype("string").str.strip().str.upper() == "Y").astype("int64")

    for c in LOWER_COLS:
        df[c] = df[c].astype("string").str.lower()

    # 시군구 코드가 소수점으로 남지 않게
    df["PRINST_LOCPLC_SIGNGU_CODE"] = (
        df["PRINST_LOCPLC_SIGNGU_CODE"].astype("Int64").astype("string")
    )

    df = fill_trobl_ty(df)

    for c, v in MODE_FILL.items():
        df[c] = df[c].fillna(v)
    for c, v in MEDIAN_FILL.items():
        df[c] = df[c].fillna(v)

    # 나이를 먼저 채운 뒤 구간을 다시 계산
    df["TRGTER_AGE_CLS"] = age_class(df["TRGTER_AGE"])
    df["PRHNF_AGE_CLS"] = age_class(df["PRHNF_AGE"])

    for c in ZERO_FILL + AGG_ZERO_FILL:
        df[c] = df[c].fillna(0)
    return df


def add_features(df):
    # 파생변수 12개
    begin = parse_dt(df["SVC_BEGIN_DT"])
    end = parse_dt(df["SVC_END_DT"])

    # 시간/요일 순환 표현. 요일은 월요일이 0
    df["F_hour_sin"] = np.sin(2 * np.pi * begin.dt.hour / 24)
    df["F_hour_cos"] = np.cos(2 * np.pi * begin.dt.hour / 24)
    df["F_dow_sin"] = np.sin(2 * np.pi * begin.dt.dayofweek / 7)
    df["F_dow_cos"] = np.cos(2 * np.pi * begin.dt.dayofweek / 7)

    # 신고 서비스시간 HHMM -> 분
    t = df["SVC_PROVD_TIME_SM"].astype("float64")
    df["F_DECL_MIN"] = ((t // 100) * 60 + (t % 100)).astype("int64")

    # 실제 서비스시간과 신고시간 차이
    df["F_SVC_DUR_MIN"] = (end - begin).dt.total_seconds() / 60
    df["F_DUR_GAP"] = df["F_SVC_DUR_MIN"] - df["F_DECL_MIN"]

    decl = df["F_DECL_MIN"].astype("float64")
    df["F_DUR_RATIO"] = (df["F_SVC_DUR_MIN"] / decl.where(decl != 0)).fillna(DUR_RATIO_FALLBACK)

    # 자정 넘김 여부
    df["F_CROSS_DATE"] = (begin.dt.date != end.dt.date).astype("Int8")

    # 정부지원금 비중
    g = df["GVSPMN_AM"].astype("float64")
    s = df["SFALMT_AM"].astype("float64")
    df["F_SUBSIDY_SHARE"] = g / (g + s).where((g + s) != 0)

    # 분당 결제금액
    dur = df["F_SVC_DUR_MIN"]
    dp = df["DDCT_PONT_SM_AM"].astype("float64")
    df["F_AMT_PER_MIN"] = (dp / dur.where(dur > 0)).fillna(AMT_PER_MIN_FALLBACK)

    # 바우처 소진율
    rp = df["REMNDR_PONT_AM"].astype("float64")
    df["F_BURN_RATIO"] = dp / (dp + rp).where((dp + rp) != 0)
    return df


def cast(df):
    for c, t in DTYPES.items():
        if t == "string":
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype("int64")
            df[c] = df[c].astype("string")
        elif t == "str":
            df[c] = df[c].astype("string").astype(object)
        else:
            df[c] = df[c].astype(t)
    return df


def finalize(df):
    # 컬럼 순서와 자료형 맞추기
    missing = [c for c in FINAL_COLS if c not in df.columns]
    if missing:
        raise KeyError("생성되지 않은 컬럼: %s" % missing)
    extra = [c for c in df.columns if c not in FINAL_COLS]
    if extra:
        raise ValueError("최종 목록에 없는 컬럼이 남음: %s" % extra)
    na = df[FINAL_COLS].isna().sum()
    if na.sum():
        raise ValueError("대치 안 된 결측: %s" % na[na > 0].to_dict())
    return cast(df[FINAL_COLS].copy())


def run(df):
    df = df.copy()
    df = drop_columns(df)
    df = add_missing_flags(df)
    df = impute(df)
    df = add_features(df)
    return finalize(df)


def compare(out, ref):
    # 기준 데이터셋과 값 대조
    if list(out.columns) != list(ref.columns):
        print("컬럼 목록/순서 불일치")
        return False
    if len(out) != len(ref):
        print("행 수 불일치 %d vs %d" % (len(out), len(ref)))
        return False

    bad = []
    for c in ref.columns:
        a, b = out[c], ref[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff = (~np.isclose(a.astype("float64"), b.astype("float64"),
                                rtol=0, atol=1e-6, equal_nan=True)).sum()
        else:
            diff = (a.astype("string").fillna("\x00") != b.astype("string").fillna("\x00")).sum()
        if diff:
            bad.append((c, int(diff)))

    print("결측 %d건, 일치 컬럼 %d/%d" % (out.isna().sum().sum(),
                                      len(ref.columns) - len(bad), len(ref.columns)))
    for c, d in bad:
        print("  %s %d건 불일치" % (c, d))
    return not bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verify")
    a = p.parse_args()

    src = pd.read_parquet(a.input)
    print("입력 %d행 %d컬럼" % src.shape)
    out = run(src)
    print("출력 %d행 %d컬럼" % out.shape)
    out.to_parquet(a.output, index=False)

    if a.verify:
        compare(out, pd.read_parquet(a.verify))


if __name__ == "__main__":
    main()
