# FDS 부정수급 탐지 시스템 고도화

부정수급 의심 사례를 찾아내기 위한 FDS 프로젝트의 **코드 중심 포트폴리오 저장소**입니다. 전처리, 지도학습, 딥러닝, 이상탐지, 최종 점수 결합 코드를 함께 정리했습니다. 원본 데이터와 학습된 모델은 포함하지 않습니다.

## 기여 범위

**제가 직접 담당한 부분은 전처리와 이상탐지입니다.** `src/preprocessing/preprocess.py`에서 결측값 처리, 파생 변수 생성, 자료형 및 출력 스키마 정리를 수행했습니다. `src/anomaly/train_anomaly.py`에서는 정상 학습 데이터를 이용한 Isolation Forest와 Denoising Autoencoder 학습, 점수 결합 및 검증 흐름을 구현했습니다.

`src/ml/`, `notebooks/ml/`, `notebooks/dl/`, `src/final_model/`은 **전체 프로젝트의 흐름을 보여 주기 위해 포함한 다른 파트 코드**입니다. 이 부분을 개인 구현 성과로 주장하지 않습니다.

## 저장소 구조

| 경로 | 내용 |
| --- | --- |
| `src/preprocessing/` | 담당: Parquet 전처리 스크립트 |
| `src/anomaly/` | 담당: 이상탐지 학습·평가 스크립트 |
| `src/ml/`, `notebooks/ml/` | 지도학습·관계 피처 코드 |
| `notebooks/dl/` | 딥러닝 학습 코드 |
| `src/final_model/` | 각 모델 점수의 결합·추론 코드 |
| `configs/` | 집계된 모델 설정과 임계값 |

## 실행 준비

```bash
python -m venv .venv
python -m pip install -r requirements-train.txt
python src/preprocessing/preprocess.py --input /path/to/input.parquet --output /path/to/processed.parquet
```

이상탐지 스크립트는 상단의 `/content/train.parquet`, `/content/valid.parquet`, `/content/test.parquet` 경로를 사용합니다. 노트북도 Colab 경로를 전제로 하므로 각자의 실행 환경에 맞게 수정해야 합니다. 최종 추론 코드는 별도의 모델 가중치와 전처리 객체가 있어야 실행됩니다. 이 저장소만으로 학습 성능이나 추론 결과를 재현할 수는 없습니다.

## 평가 원칙과 데이터 보호

이상탐지 모델과 점수 정규화는 정상 학습 데이터에 맞추고, 결합 가중치·임계값은 검증 데이터로 선택하며, 테스트 데이터는 최종 평가에만 사용합니다. 데이터 누수를 피하기 위한 프로젝트의 기본 원칙입니다.

개인정보가 포함될 수 있는 원본·가공 데이터, 샘플 행, 점수 CSV, 모델 파일, 실행 결과, 보고서는 업로드하지 않았습니다. 노트북 실행 결과 셀도 비워 두었습니다. 저장소는 현재 비공개입니다.
