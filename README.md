# 🏭 설비 고장 예측 대시보드
> AI4I 2020 데이터 기반 예측 유지보수(Predictive Maintenance) 프로젝트

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📌 프로젝트 배경

제조업에서 설비 고장은 **수리비 + 생산 중단 손실**이라는 이중 비용을 발생시킵니다.
500대 기계가 가동되는 공장 기준으로 월 고장률 3.4%만 돼도 **월 수천만 원의 손실**이 발생합니다.

기존의 **사후 대응(Reactive Maintenance)** 방식에서
머신러닝 기반 **사전 탐지(Predictive Maintenance)** 방식으로 전환하면
고장의 85%를 사전에 탐지해 비용을 대폭 절감할 수 있습니다.

---

## 🔍 분석 과정

### 1️⃣ EDA (탐색적 데이터 분석)
- 데이터: AI4I 2020 Predictive Maintenance Dataset (10,000건, 고장률 3.4%)
- 고장 유형 5가지 분석: TWF / HDF / PWF / OSF / RNF
- **도메인 지식 기반 파생변수 3개 생성**
  - `Temp_diff` = Process temp − Air temp → HDF 탐지 핵심
  - `Power` = Torque × rpm × (2π/60) → PWF 탐지 핵심
  - `Torque_Wear` = Torque × Tool wear → OSF 탐지 핵심
- 파생변수 조건과 실제 고장 일치율 **100%** 확인
- VIF 기반 다중공선성 검증 후 최종 피처 5개 확정

### 2️⃣ 모델링
| 모델 | Recall | ROC-AUC | F1-score |
|------|--------|---------|---------|
| Logistic Regression | 0.88 | 0.87 | 0.16 |
| Random Forest | 0.75 | 0.94 | 0.68 |
| XGBoost | 0.79 | 0.96 | 0.71 |
| **LightGBM (최종)** | **0.85** | **0.97** | **0.69** |

- **불균형 데이터** 처리: `class_weight='balanced'`
- **Threshold 최적화**: 0.5 → 0.7로 조정해 Recall 향상
- **교차검증**: 5-Fold CV로 일반화 성능 확인
- **SHAP 분석**: 피처 중요도 및 예측 근거 설명

### 3️⃣ 비즈니스 인사이트
- 고장 탐지율(Recall) **85%** 달성
- 500대 기계 기준 **월 약 3,000만원 절감** 예상
- 고장 유형별 실무 조치 기준 도출
  - Tool wear ≥ 200분 → 교체 필요
  - Temp_diff ≤ 9K + rpm ≤ 1380 → HDF 위험
  - Power ≥ 7,000W → PWF 점검 필요

---

## 🎯 핵심 결과

| 지표 | 값 |
|------|-----|
| 최종 모델 | LightGBM |
| Recall (고장 탐지율) | **0.85** |
| ROC-AUC | **0.97** |
| F1-score | **0.69** |
| 월 비용 절감 (500대 기준) | **약 3,000만원** |
| 연간 비용 절감 | **약 3.6억원** |

---

## 📊 대시보드

| 탭 | 내용 |
|----|------|
| 📊 EDA | 수치형 변수 분포, 고장 유형 분석, 등급별 분석, 상관관계 히트맵 |
| 🔮 고장 예측 | 센서값 입력 → 실시간 고장 확률 예측 + 고장 유형 추정 |
| 🧠 SHAP 분석 | 피처 중요도, Summary Plot, Waterfall Plot |
| 💰 비용 계산기 | 파라미터 조정 → 모델 도입 ROI 실시간 계산 |

👉 **[대시보드 바로가기](https://your-app-url.streamlit.app)**

---

## 🛠️ 기술 스택
```

Python 3.x
├── 데이터 분석: pandas, numpy
├── 시각화: plotly, matplotlib, seaborn
├── 머신러닝: scikit-learn, lightgbm, xgboost
├── 모델 설명: shap
└── 대시보드: streamlit

```
---

## 📁 프로젝트 구조

\```
📦 AI4I-Predictive-Maintenance
 ┣ 📓 01_EDA.ipynb
 ┣ 📓 02_modeling.ipynb
 ┣ 📓 03_business.ipynb
 ┣ 📊 app.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┗ 📂 data/
    ┗ ai4i2020.csv
\```

---

## ⚙️ 로컬 실행 방법

\```bash
git clone https://github.com/your-username/AI4I-Predictive-Maintenance.git
cd AI4I-Predictive-Maintenance
pip install -r requirements.txt
streamlit run app.py
\```
