# ============================================================
# FDS V2 비지도 이상탐지 학습 전체 코드
#
# 필요 파일
#   /content/train.parquet
#   /content/valid.parquet
#   /content/test.parquet
#
# Target
#   INJSTUSE_AT
#   0 = 정상
#   1 = 부정
#
# 모델
#   1. Isolation Forest
#   2. Denoising Autoencoder
#   3. IF + AE weighted ensemble
#
# 학습 원칙
#   - 전처리 FIT: Train 정상만
#   - IF 학습: Train 정상만
#   - AE 학습: Train 정상만
#   - Score normalization: Train 정상만
#   - IF/AE 결합 가중치 선택: Validation
#   - Test: 최종 점수 생성/평가만
# ============================================================


# ============================================================
# 0. 라이브러리
# ============================================================

import os
import json
import copy
import random
import shutil
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. 기본 설정
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("DEVICE:", DEVICE)


# ============================================================
# 2. 파일 경로
# ============================================================

TRAIN_PATH = "/content/train.parquet"
VALID_PATH = "/content/valid.parquet"
TEST_PATH  = "/content/test.parquet"

OUTPUT_DIR = "/content/anomaly_v2_model"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. 데이터 로드
# ============================================================

train = pd.read_parquet(
    TRAIN_PATH
)

valid = pd.read_parquet(
    VALID_PATH
)

test = pd.read_parquet(
    TEST_PATH
)


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    "Train:",
    train.shape
)

print(
    "Valid:",
    valid.shape
)

print(
    "Test :",
    test.shape
)


# ============================================================
# 4. 기본 컬럼
# ============================================================

TARGET = "INJSTUSE_AT"
ROW_ID = "row_id"


for col in [
    TARGET,
    ROW_ID
]:

    if col not in train.columns:

        raise ValueError(
            f"{col} 컬럼이 없습니다."
        )


# ============================================================
# 5. Target
# ============================================================

y_train = (
    train[TARGET]
    .astype(int)
    .reset_index(drop=True)
)

y_valid = (
    valid[TARGET]
    .astype(int)
    .reset_index(drop=True)
)

y_test = (
    test[TARGET]
    .astype(int)
    .reset_index(drop=True)
)


# ============================================================
# 6. 비지도 학습용 정상 Train
# ============================================================

train_normal = (

    train.loc[
        train[TARGET] == 0
    ]

    .copy()

    .reset_index(
        drop=True
    )
)


print()
print(
    "Train 전체:",
    len(train)
)

print(
    "Train 정상:",
    len(train_normal)
)


# ============================================================
# 7. V2 입력 Feature
#
# 49 Numeric + 9 Binary = 58
# ============================================================

NUMERIC_FEATURES = [

    "SVC_KND_SM_CO",
    "ADIT_SMMNTN_TIME",

    "DDCT_PONT_SM_AM",
    "DDCT_VOUCH_NMRS_SM",

    "GVSPMN_AM",
    "SFALMT_AM",

    "REMNDR_PONT_AM",
    "REMNDR_VOUCH_NMRS",

    "PRHNF_NMPR_CO",

    "CREAT_PONT_AM",
    "CREAT_SFALMT_AM",
    "CREAT_GVSPMN_AM",

    "VOUCH_UNTPC",

    "PRHNF_AGE",

    "DPR_TODAY_CNT",
    "DTR_TODAY_CNT",

    "DPR_DDT_CLS_CNT",
    "DTR_DDT_CLS_CNT",

    "DPR_DDIF_1MIN_CNT",
    "DPR_DDIF_3MIN_CNT",
    "DPR_DDIF_10MIN_CNT",

    "SFALMT_RATE",

    "DPR_1MTH_CNT",
    "DPR_1MTH_DELDE_CNT",
    "DPR_1MTH_SVCTM",
    "DPR_1MTH_1DAY_SVCTM_AVG",
    "DPR_1MTH_1DAY_MAX_CNT",

    "DTR_DBS_1MTH_CNT",
    "DTR_DBS_1MTH_SVCTM",
    "DTR_DBS_1MTH_AVG_AMT",
    "DTR_DBS_1MTH_DIF_AVG_AMT",

    "CPR_CHARCT_SE_missing",
    "ESTB_DIV_CODE_missing",

    "F_hour_sin",
    "F_hour_cos",

    "F_dow_sin",
    "F_dow_cos",

    "F_DECL_MIN",
    "F_SVC_DUR_MIN",

    "F_DUR_GAP",
    "F_DUR_RATIO",

    "F_CROSS_DATE",

    "F_SUBSIDY_SHARE",
    "F_AMT_PER_MIN",
    "F_BURN_RATIO",

    "SVC_PROVD_TIME_SM",

    "DTR_DDIF_1MIN_CNT",
    "DTR_DDIF_3MIN_CNT",
    "DTR_DDIF_10MIN_CNT"
]


BINARY_FEATURES = [

    "BNDE_SETLE_SE",

    "HLDY_SE",
    "NIGHT_SE",

    "SVC_BEGIN_DT_BSNDE_AT",

    "SVC_BEGIN_DT_MIDTA_AT",
    "SVC_BEGIN_DT_MIDTB_AT",

    "PRHNF_SEXDSTN",

    "SVC_BEGIN_AFTER_DEATH_AT",

    "PRHNF_HSPTLZ_AT"
]


RAW_FEATURES = (

    NUMERIC_FEATURES
    +
    BINARY_FEATURES
)


print()
print(
    "Numeric:",
    len(NUMERIC_FEATURES)
)

print(
    "Binary:",
    len(BINARY_FEATURES)
)

print(
    "Total:",
    len(RAW_FEATURES)
)


# ============================================================
# 8. Feature 존재 여부 확인
# ============================================================

for dataset_name, df in [

    ("Train", train),
    ("Valid", valid),
    ("Test", test)

]:

    missing = [

        col
        for col in RAW_FEATURES

        if col not in df.columns
    ]


    if len(
        missing
    ) > 0:

        raise ValueError(

            f"{dataset_name}에 "
            f"필요한 컬럼이 없습니다:\n"
            f"{missing}"
        )


# ============================================================
# 9. Numeric 결측치 처리 기준
#
# Train 정상 중앙값만 사용
# ============================================================

numeric_fill_values = {}


for col in NUMERIC_FEATURES:

    x = pd.to_numeric(

        train_normal[col],

        errors="coerce"
    )


    median = x.median()


    if pd.isna(
        median
    ):

        median = 0.0


    numeric_fill_values[
        col
    ] = float(
        median
    )


# ============================================================
# 10. Binary Mapping
#
# Train 정상에서만 mapping 생성
# ============================================================

binary_maps = {}


for col in BINARY_FEATURES:

    values = (

        train_normal[col]

        .fillna(
            "__MISSING__"
        )

        .astype(str)

        .unique()

        .tolist()
    )


    values = sorted(
        values
    )


    if len(values) != 2:

        print(
            f"[WARNING] {col} "
            f"unique={values}"
        )


    binary_maps[
        col
    ] = {

        value: idx

        for idx, value
        in enumerate(
            values[:2]
        )
    }


print()
print("=" * 70)
print("BINARY MAPPING")
print("=" * 70)


for col, mapping in (
    binary_maps.items()
):

    print(
        col,
        mapping
    )


# ============================================================
# 11. 공통 전처리 함수
# ============================================================

def preprocess_dataframe(
    df
):

    output = pd.DataFrame(
        index=df.index
    )


    # ========================================================
    # Numeric
    # ========================================================

    for col in NUMERIC_FEATURES:

        value = pd.to_numeric(

            df[col],

            errors="coerce"
        )


        value = value.fillna(

            numeric_fill_values[
                col
            ]
        )


        output[col] = (
            value.astype(
                np.float32
            )
        )


    # ========================================================
    # Binary
    # ========================================================

    for col in BINARY_FEATURES:

        value = (

            df[col]

            .fillna(
                "__MISSING__"
            )

            .astype(str)
        )


        value = (

            value.map(
                binary_maps[col]
            )

            .fillna(-1)

            .astype(
                np.float32
            )
        )


        output[
            f"{col}__bin"
        ] = value


    return (

        output
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 12. 전처리 실행
# ============================================================

X_train_df = preprocess_dataframe(
    train_normal
)

X_valid_df = preprocess_dataframe(
    valid
)

X_test_df = preprocess_dataframe(
    test
)


FEATURE_NAMES = (
    X_train_df
    .columns
    .tolist()
)


X_train = (

    X_train_df

    .to_numpy(
        dtype=np.float32
    )
)


X_valid = (

    X_valid_df

    .to_numpy(
        dtype=np.float32
    )
)


X_test = (

    X_test_df

    .to_numpy(
        dtype=np.float32
    )
)


print()
print("=" * 70)
print("PREPROCESSING")
print("=" * 70)

print(
    "Train:",
    X_train.shape
)

print(
    "Valid:",
    X_valid.shape
)

print(
    "Test :",
    X_test.shape
)


# ============================================================
# 13. NaN / Inf 확인
# ============================================================

def check_matrix(
    X,
    name
):

    print(

        name,

        "| NaN:",
        np.isnan(
            X
        ).sum(),

        "| Inf:",
        np.isinf(
            X
        ).sum()
    )


check_matrix(
    X_train,
    "Train"
)

check_matrix(
    X_valid,
    "Valid"
)

check_matrix(
    X_test,
    "Test"
)


# ============================================================
# 14. Score normalization
#
# Train-normal 기준
# R -> 0~1 단조변환
# ============================================================

def fit_score_normalizer(
    score
):

    median = float(
        np.median(
            score
        )
    )

    q25 = float(
        np.quantile(
            score,
            0.25
        )
    )

    q75 = float(
        np.quantile(
            score,
            0.75
        )
    )


    iqr = (
        q75
        -
        q25
    )


    if iqr <= 1e-12:

        iqr = float(
            np.std(
                score
            )
        )


    if iqr <= 1e-12:

        iqr = 1.0


    return {

        "median":
            median,

        "iqr":
            iqr
    }


def normalize_score(
    score,
    params
):

    z = (

        (
            score
            -
            params["median"]
        )

        /
        params["iqr"]
    )


    normalized = (

        0.5
        +
        np.arctan(z)
        /
        np.pi
    )


    return normalized.astype(
        np.float32
    )


# ============================================================
# ============================================================
# PART A
# Isolation Forest
# ============================================================
# ============================================================

print()
print("=" * 70)
print("ISOLATION FOREST TRAIN")
print("=" * 70)


iso_model = IsolationForest(

    n_estimators=500,

    max_samples=4096,

    max_features=1.0,

    contamination="auto",

    bootstrap=False,

    n_jobs=-1,

    random_state=SEED
)


# 정상 Train만 학습
iso_model.fit(
    X_train
)


# 높을수록 이상하도록 부호 반전
if_train_raw = (

    -iso_model
    .score_samples(
        X_train
    )
)


if_valid_raw = (

    -iso_model
    .score_samples(
        X_valid
    )
)


if_test_raw = (

    -iso_model
    .score_samples(
        X_test
    )
)


# normalization도 Train normal만
if_norm = (
    fit_score_normalizer(
        if_train_raw
    )
)


score_if_train = (
    normalize_score(
        if_train_raw,
        if_norm
    )
)


score_if_valid = (
    normalize_score(
        if_valid_raw,
        if_norm
    )
)


score_if_test = (
    normalize_score(
        if_test_raw,
        if_norm
    )
)


print(
    "IF Valid PR-AUC:",
    average_precision_score(
        y_valid,
        score_if_valid
    )
)


print(
    "IF Valid ROC-AUC:",
    roc_auc_score(
        y_valid,
        score_if_valid
    )
)


# ============================================================
# ============================================================
# PART B
# Autoencoder preprocessing
# ============================================================
# ============================================================

print()
print("=" * 70)
print("AUTOENCODER PREPROCESSING")
print("=" * 70)


# Train 정상만 scaler fit
ae_scaler = RobustScaler(
    quantile_range=(
        25,
        75
    )
)


X_train_ae = (

    ae_scaler

    .fit_transform(
        X_train
    )

    .astype(
        np.float32
    )
)


INPUT_DIM = (
    X_train_ae.shape[1]
)


# ============================================================
# 15. Autoencoder
# ============================================================

class DenoisingAutoencoder(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        latent_dim
    ):

        super().__init__()


        hidden1 = max(
            32,
            min(
                128,
                input_dim * 2
            )
        )


        hidden2 = max(
            16,
            min(
                64,
                input_dim
            )
        )


        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden1
            ),

            nn.ReLU(),

            nn.Linear(
                hidden1,
                hidden2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden2,
                latent_dim
            )
        )


        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden2,
                hidden1
            ),

            nn.ReLU(),

            nn.Linear(
                hidden1,
                input_dim
            )
        )


    def forward(
        self,
        x
    ):

        return self.decoder(

            self.encoder(
                x
            )
        )


# ============================================================
# 16. Train / internal valid 분리
# ============================================================

train_tensor = (
    torch.from_numpy(
        X_train_ae
    )
)


dataset = TensorDataset(
    train_tensor
)


train_size = int(
    len(dataset)
    * 0.9
)


valid_size = (
    len(dataset)
    -
    train_size
)


generator = (
    torch.Generator()
    .manual_seed(
        SEED
    )
)


train_dataset, internal_valid_dataset = (

    random_split(

        dataset,

        [
            train_size,
            valid_size
        ],

        generator=generator
    )
)


BATCH_SIZE = 2048


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


internal_valid_loader = DataLoader(

    internal_valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# 17. AE Training
# ============================================================

def train_ae(
    latent_dim
):

    model = (

        DenoisingAutoencoder(

            INPUT_DIM,

            latent_dim

        ).to(
            DEVICE
        )
    )


    criterion = (
        nn.SmoothL1Loss()
    )


    optimizer = (
        torch.optim.Adam(

            model.parameters(),

            lr=1e-3,

            weight_decay=1e-5
        )
    )


    MAX_EPOCHS = 30

    PATIENCE = 5

    NOISE_RATE = 0.10


    best_loss = (
        float("inf")
    )

    best_state = None

    patience_count = 0


    history = []


    print()
    print(
        f"AE latent={latent_dim}"
    )


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):


        # ====================================================
        # Train
        # ====================================================

        model.train()


        total = 0

        count = 0


        for (batch,) in (
            train_loader
        ):

            batch = batch.to(
                DEVICE
            )


            # 10% masking noise
            noisy = (
                batch.clone()
            )


            mask = (

                torch.rand_like(
                    noisy
                )
                <
                NOISE_RATE
            )


            noisy[
                mask
            ] = 0


            optimizer.zero_grad()


            output = model(
                noisy
            )


            loss = criterion(
                output,
                batch
            )


            loss.backward()

            optimizer.step()


            total += (
                loss.item()
                *
                len(batch)
            )


            count += (
                len(batch)
            )


        train_loss = (
            total
            /
            count
        )


        # ====================================================
        # Internal Validation
        # ====================================================

        model.eval()


        total = 0

        count = 0


        with torch.no_grad():

            for (batch,) in (
                internal_valid_loader
            ):

                batch = batch.to(
                    DEVICE
                )


                output = model(
                    batch
                )


                loss = criterion(
                    output,
                    batch
                )


                total += (
                    loss.item()
                    *
                    len(batch)
                )


                count += (
                    len(batch)
                )


        val_loss = (
            total
            /
            count
        )


        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "valid_loss":
                val_loss
        })


        print(

            f"Epoch {epoch:02d} | "

            f"Train "
            f"{train_loss:.6f} | "

            f"Valid "
            f"{val_loss:.6f}"
        )


        # ====================================================
        # Early stopping
        # ====================================================

        if (
            val_loss
            <
            best_loss - 1e-6
        ):

            best_loss = (
                val_loss
            )


            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )


            patience_count = 0


        else:

            patience_count += 1


            if (
                patience_count
                >=
                PATIENCE
            ):

                print(
                    "Early stopping"
                )

                break


    model.load_state_dict(
        best_state
    )


    return (
        model,
        best_loss,
        pd.DataFrame(
            history
        )
    )


# ============================================================
# 18. Reconstruction score
# ============================================================

def get_ae_scores(
    X,
    model,
    batch_size=4096
):

    model.eval()


    mean_scores = []

    top10_scores = []

    max_scores = []


    top_k = max(

        1,

        int(
            np.ceil(
                X.shape[1]
                *
                0.10
            )
        )
    )


    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):


            end = min(

                start
                +
                batch_size,

                len(X)
            )


            batch_np = (

                ae_scaler

                .transform(
                    X[
                        start:end
                    ]
                )

                .astype(
                    np.float32
                )
            )


            batch = (

                torch.from_numpy(
                    batch_np
                )

                .to(
                    DEVICE
                )
            )


            output = model(
                batch
            )


            error = (

                output
                -
                batch

            ) ** 2


            mean_error = (
                error.mean(
                    dim=1
                )
            )


            top10_error = (

                torch.topk(

                    error,

                    k=top_k,

                    dim=1

                )[0]

                .mean(
                    dim=1
                )
            )


            max_error = (
                error.max(
                    dim=1
                )[0]
            )


            mean_scores.append(

                mean_error
                .cpu()
                .numpy()
            )


            top10_scores.append(

                top10_error
                .cpu()
                .numpy()
            )


            max_scores.append(

                max_error
                .cpu()
                .numpy()
            )


    return {

        "mean":
            np.concatenate(
                mean_scores
            ),

        "top10":
            np.concatenate(
                top10_scores
            ),

        "max":
            np.concatenate(
                max_scores
            )
    }


# ============================================================
# ============================================================
# PART C
# AE 모델 탐색
#
# latent 8 / 16
# mean / top10 / max
# ============================================================
# ============================================================

AE_LATENTS = [
    8,
    16
]


ae_results = []

ae_models = {}

ae_scores = {}

ae_norms = {}


for latent in AE_LATENTS:


    model, internal_loss, history = (
        train_ae(
            latent
        )
    )


    history.to_csv(

        os.path.join(

            OUTPUT_DIR,

            f"ae_latent_{latent}_history.csv"
        ),

        index=False
    )


    print(
        f"latent={latent} score 계산"
    )


    train_raw = get_ae_scores(
        X_train,
        model
    )


    valid_raw = get_ae_scores(
        X_valid,
        model
    )


    test_raw = get_ae_scores(
        X_test,
        model
    )


    for score_type in [

        "mean",
        "top10",
        "max"

    ]:


        norm = (
            fit_score_normalizer(

                train_raw[
                    score_type
                ]
            )
        )


        train_score = (
            normalize_score(

                train_raw[
                    score_type
                ],

                norm
            )
        )


        valid_score = (
            normalize_score(

                valid_raw[
                    score_type
                ],

                norm
            )
        )


        test_score = (
            normalize_score(

                test_raw[
                    score_type
                ],

                norm
            )
        )


        model_name = (

            f"DAE_latent"
            f"{latent}_"
            f"{score_type}"
        )


        pr_auc = (
            average_precision_score(

                y_valid,

                valid_score
            )
        )


        roc_auc = (
            roc_auc_score(

                y_valid,

                valid_score
            )
        )


        ae_results.append({

            "model":
                model_name,

            "latent":
                latent,

            "score_type":
                score_type,

            "internal_loss":
                internal_loss,

            "pr_auc":
                pr_auc,

            "roc_auc":
                roc_auc
        })


        ae_scores[
            model_name
        ] = {

            "train":
                train_score,

            "valid":
                valid_score,

            "test":
                test_score
        }


        ae_norms[
            model_name
        ] = norm


    # state 저장
    ae_models[
        latent
    ] = {

        key:
            value
            .detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }


    del model


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 19. Best AE 선택
#
# Validation PR-AUC 기준
# ============================================================

ae_result_df = (

    pd.DataFrame(
        ae_results
    )

    .sort_values(

        "pr_auc",

        ascending=False
    )

    .reset_index(
        drop=True
    )
)


print()
print("=" * 70)
print("AE RANKING")
print("=" * 70)

print(
    ae_result_df
)


best_ae_name = (

    ae_result_df.iloc[0][
        "model"
    ]
)


best_ae_latent = int(

    ae_result_df.iloc[0][
        "latent"
    ]
)


best_ae_score_type = (

    ae_result_df.iloc[0][
        "score_type"
    ]
)


score_ae_train = (

    ae_scores[
        best_ae_name
    ]["train"]
)


score_ae_valid = (

    ae_scores[
        best_ae_name
    ]["valid"]
)


score_ae_test = (

    ae_scores[
        best_ae_name
    ]["test"]
)


print()
print(
    "Best AE:",
    best_ae_name
)


# ============================================================
# 20. IF / AE Correlation
# ============================================================

if_ae_corr = float(

    np.corrcoef(

        score_if_valid,

        score_ae_valid

    )[0, 1]
)


print(
    "IF-AE Correlation:",
    if_ae_corr
)


# ============================================================
# ============================================================
# PART D
# IF + AE Weight Search
#
# Validation PR-AUC 기준
# ============================================================
# ============================================================

weight_results = []


for w_if in np.arange(
    0,
    1.001,
    0.05
):


    w_ae = (
        1
        -
        w_if
    )


    score = (

        w_if
        *
        score_if_valid

        +

        w_ae
        *
        score_ae_valid
    )


    weight_results.append({

        "w_if":
            float(w_if),

        "w_ae":
            float(w_ae),

        "pr_auc":
            average_precision_score(

                y_valid,

                score
            ),

        "roc_auc":
            roc_auc_score(

                y_valid,

                score
            )
    })


weight_df = (

    pd.DataFrame(
        weight_results
    )

    .sort_values(

        "pr_auc",

        ascending=False
    )

    .reset_index(
        drop=True
    )
)


print()
print("=" * 70)
print("IF + AE WEIGHT SEARCH")
print("=" * 70)

print(
    weight_df.head(20)
)


best_w_if = float(

    weight_df.iloc[0][
        "w_if"
    ]
)


best_w_ae = float(

    weight_df.iloc[0][
        "w_ae"
    ]
)


# ============================================================
# 21. 최종 anomaly score
# ============================================================

score_anom_valid = (

    best_w_if
    *
    score_if_valid

    +

    best_w_ae
    *
    score_ae_valid
)


score_anom_test = (

    best_w_if
    *
    score_if_test

    +

    best_w_ae
    *
    score_ae_test
)


# ============================================================
# 22. Validation 기준 Threshold
#
# 참고용 단독 F1
# ============================================================

precision, recall, thresholds = (

    precision_recall_curve(

        y_valid,

        score_anom_valid
    )
)


f1_array = (

    2
    *
    precision[:-1]
    *
    recall[:-1]

    /

    (
        precision[:-1]
        +
        recall[:-1]
        +
        1e-12
    )
)


best_idx = int(

    np.argmax(
        f1_array
    )
)


best_threshold = float(

    thresholds[
        best_idx
    ]
)


valid_pred = (

    score_anom_valid
    >=
    best_threshold

).astype(int)


# ============================================================
# 23. Validation 결과
# ============================================================

valid_pr_auc = (
    average_precision_score(

        y_valid,

        score_anom_valid
    )
)


valid_roc_auc = (
    roc_auc_score(

        y_valid,

        score_anom_valid
    )
)


valid_f1 = (
    f1_score(

        y_valid,

        valid_pred
    )
)


valid_precision = (
    precision_score(

        y_valid,

        valid_pred
    )
)


valid_recall = (
    recall_score(

        y_valid,

        valid_pred
    )
)


print()
print("=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)


print(
    "Best AE:",
    best_ae_name
)

print(
    "IF weight:",
    best_w_if
)

print(
    "AE weight:",
    best_w_ae
)

print(
    "PR-AUC:",
    valid_pr_auc
)

print(
    "ROC-AUC:",
    valid_roc_auc
)

print(
    "F1:",
    valid_f1
)

print(
    "Precision:",
    valid_precision
)

print(
    "Recall:",
    valid_recall
)

print(
    "Threshold:",
    best_threshold
)


# ============================================================
# 24. Test 평가
#
# Validation에서 확정한 threshold 그대로 사용
# ============================================================

test_pred = (

    score_anom_test
    >=
    best_threshold

).astype(int)


test_metrics = {

    "pr_auc":
        average_precision_score(

            y_test,

            score_anom_test
        ),

    "roc_auc":
        roc_auc_score(

            y_test,

            score_anom_test
        ),

    "f1":
        f1_score(

            y_test,

            test_pred
        ),

    "precision":
        precision_score(

            y_test,

            test_pred
        ),

    "recall":
        recall_score(

            y_test,

            test_pred
        )
}


print()
print("=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)


for key, value in (
    test_metrics.items()
):

    print(
        key,
        ":",
        value
    )


# ============================================================
# 25. Score CSV
# ============================================================

valid_output = pd.DataFrame({

    "row_id":
        valid[ROW_ID]
        .reset_index(
            drop=True
        ),

    "y":
        y_valid,

    "score_if":
        score_if_valid,

    "score_ae":
        score_ae_valid,

    "score_anom":
        score_anom_valid
})


test_output = pd.DataFrame({

    "row_id":
        test[ROW_ID]
        .reset_index(
            drop=True
        ),

    "y":
        y_test,

    "score_if":
        score_if_test,

    "score_ae":
        score_ae_test,

    "score_anom":
        score_anom_test
})


valid_output.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "anom_scores_valid.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


test_output.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "anom_scores_test.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 26. 전처리 저장
# ============================================================

preprocessor = {

    "numeric_features":
        NUMERIC_FEATURES,

    "binary_features":
        BINARY_FEATURES,

    "feature_names":
        FEATURE_NAMES,

    "numeric_fill_values":
        numeric_fill_values,

    "binary_maps":
        binary_maps
}


joblib.dump(

    preprocessor,

    os.path.join(

        OUTPUT_DIR,

        "anomaly_preprocessor.pkl"
    )
)


# ============================================================
# 27. IF 저장
# ============================================================

joblib.dump(

    iso_model,

    os.path.join(

        OUTPUT_DIR,

        "isolation_forest.pkl"
    )
)


joblib.dump(

    if_norm,

    os.path.join(

        OUTPUT_DIR,

        "if_score_normalization.pkl"
    )
)


# ============================================================
# 28. AE scaler 저장
# ============================================================

joblib.dump(

    ae_scaler,

    os.path.join(

        OUTPUT_DIR,

        "ae_scaler.pkl"
    )
)


# ============================================================
# 29. AE normalization 저장
# ============================================================

best_ae_norm = (

    ae_norms[
        best_ae_name
    ]
)


joblib.dump(

    best_ae_norm,

    os.path.join(

        OUTPUT_DIR,

        "ae_score_normalization.pkl"
    )
)


# ============================================================
# 30. AE Model 저장
# ============================================================

torch.save(

    {

        "model_state_dict":

            ae_models[
                best_ae_latent
            ],

        "input_dim":
            INPUT_DIM,

        "latent_dim":
            best_ae_latent,

        "score_type":
            best_ae_score_type

    },

    os.path.join(

        OUTPUT_DIR,

        "denoising_autoencoder.pt"
    )
)


# ============================================================
# 31. Config 저장
# ============================================================

config = {

    "seed":
        SEED,

    "train_normal_count":
        len(train_normal),

    "feature_count":
        len(FEATURE_NAMES),

    "if_n_estimators":
        500,

    "if_max_samples":
        4096,

    "best_ae":
        best_ae_name,

    "best_ae_latent":
        best_ae_latent,

    "best_ae_score_type":
        best_ae_score_type,

    "if_weight":
        best_w_if,

    "ae_weight":
        best_w_ae,

    "if_ae_correlation":
        if_ae_corr,

    "valid_pr_auc":
        valid_pr_auc,

    "valid_roc_auc":
        valid_roc_auc,

    "valid_f1":
        valid_f1,

    "valid_threshold":
        best_threshold,

    "test_metrics":
        test_metrics,

    "test_used_for_model_selection":
        False
}


with open(

    os.path.join(

        OUTPUT_DIR,

        "anomaly_config.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        config,

        f,

        ensure_ascii=False,

        indent=4
    )


# ============================================================
# 32. 실험 결과 저장
# ============================================================

ae_result_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "ae_model_search.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


weight_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "if_ae_weight_search.csv"
    ),

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 33. ZIP
# ============================================================

ZIP_PATH = shutil.make_archive(

    "/content/anomaly_v2_model",

    "zip",

    OUTPUT_DIR
)


print()
print("=" * 70)
print("FINISHED")
print("=" * 70)

print(
    "Output:",
    OUTPUT_DIR
)

print(
    "ZIP:",
    ZIP_PATH
)