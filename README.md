# Credit Card Limit Prediction

A linear regression model that predicts the credit limit (`LIMIT_BAL`) assigned to a customer, using 6 months of financial behaviour data from 30,000 credit card holders.

---

## 📌 Project Overview

Banks decide how much credit to give each customer based on a range of factors — income proxies, spending patterns, repayment history, demographics. This project reverse-engineers that decision using machine learning.

**Goal:** Build a linear regression model that achieves **R² ≥ 0.45** on a held-out test set.  
**Result:** Final OLS model achieved **R² = 0.47**, with a 5-fold cross-validation score of **0.46 ± 0.01**.

---

## 📂 Repository Contents

```
├── credit_limit_final.ipynb          # Full analysis notebook (run top to bottom)
├── Credit_Limit_Prediction_Insights.pptx  # Presentation summarising findings
└── README.md
```

> **Dataset not included.** See [Data](#-data) section below for how to obtain it.

---

## 📊 Data

**Source:** The `default of credit card clients.csv` file 
 
Place it in the same directory as the notebook before running.

| Property | Detail |
|---|---|
| Rows | 30,000 customers |
| Columns | 25 features + 1 target |
| Target | `LIMIT_BAL` — credit limit in New Taiwan Dollars |
| Period | April–September 2005 |

**Key columns:**

| Column | Description |
|---|---|
| `LIMIT_BAL` | ⭐ Target — credit limit assigned (NTD) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographics |
| `PAY_0` … `PAY_6` | Repayment status each month (-2=early, 0=on time, 1+=months delayed) |
| `BILL_AMT1` … `BILL_AMT6` | Statement balance each month |
| `PAY_AMT1` … `PAY_AMT6` | Amount paid each month |
| `DEFAULT` | Whether the customer defaulted next month (not used as a predictor) |

---

## 🔬 Methodology

The notebook follows these steps in order:

### 1. Preliminary Analysis
- Shape, dtypes, missing value check
- Target variable distribution (raw and log-transformed)

### 2. Data Pre-processing
- Drop `ID` column (row identifier, no predictive value)
- Remap undocumented codes in `EDUCATION` (0, 5, 6 → 4) and `MARRIAGE` (0 → 3) to avoid spurious categories

### 3. Exploratory Data Analysis (EDA)
- Correlation heatmap across all raw features
- `LIMIT_BAL` by education level and marital status (boxplots)
- Age vs credit limit scatter (reveals nonlinear relationship)
- Payment status (`PAY_0`) vs credit limit (clear step-down with each month of delay)

### 4. Feature Engineering
No single raw feature exceeds |r| = 0.30 with `LIMIT_BAL`. Six layers of engineering were required:

| Step | Features Created | Rationale |
|---|---|---|
| Bill & payment aggregates | `AVG_BILL`, `MAX_BILL`, `TOTAL_BILL`, `AVG_PAID`, `TOTAL_PAID`, `BILL_TREND` | Compress 6 months into summary stats |
| Payment ratios | `RATIO_1`–`RATIO_6`, `AVG_UTIL`, `TOTAL_UTIL` | How much of each bill was actually paid |
| Log transforms | `LOG_PAY_AMT1`–`LOG_PAY_AMT6` | Reduce right-skew in large lump-sum payments |
| Age nonlinearity | `AGE_SQ` | Capture diminishing returns of age on credit limits |
| Calculated credit score | `CREDIT_SCORE` (0–100) | 50% repayment behaviour + 50% payment-to-bill ratio |
| Interaction term | `CS_x_BILL` | Credit score × average bill — top predictor in final model |

### 5. Encoding & Scaling
- Categorical variables (`SEX`, `EDUCATION`, `MARRIAGE`) → one-hot encoded (avoids false ordinal assumptions)
- All numerical features → `StandardScaler`
- Degree-2 polynomial expansion on 16 key engineered features (153 additional terms)

### 6. Model Building
Three models evaluated:

| Model | R² Test | Notes |
|---|---|---|
| OLS Linear Regression | **0.47** | Selected — best generalisation |
| Ridge Regression | 0.47 | Marginal difference from OLS |
| Lasso Regression | 0.46 | Some feature compression |

5-fold cross-validation confirmed no overfitting: **CV R² = 0.46 ± 0.01**

### 7. Model Evaluation
- Actual vs Predicted scatter plot (R² = 0.4664 shown on plot)
- Residual plot — reveals heteroscedasticity at high predicted limits
- Residual distribution — near-normal with slight right tail
- Top 20 OLS coefficients — polynomial terms and `AGE_SQ` dominate; `CS_x_BILL` is the strongest interpretable feature

---

## 📈 Key Findings

1. **Spending level is the strongest signal.** `CS_x_BILL` (credit score × average bill) and `TOTAL_PAID` are the top predictors. Banks appear to set limits proportional to how much a customer spends and how reliably they pay it back.

2. **Even one missed payment hurts.** Customers with `PAY_0 = 1` (one month delayed) have noticeably lower limits. The penalty compounds with each additional month of delay.

3. **Education is a real signal.** Graduate-school customers receive systematically higher limits, even after controlling for spending and payment behaviour. Banks likely use education as an income proxy.

4. **Feature engineering was the game-changer.** Raw features alone topped out at R² = 0.36. The engineered features — particularly the polynomial expansion — lifted accuracy to 0.47, a ~30% relative improvement.

5. **Residual heteroscedasticity at high limits.** The model's errors widen for customers with limits above ₹300,000. A log-transformed target or a tree-based model (e.g. XGBoost) could reduce this further.

---

## ⚙️ How to Run

**Requirements:** Python 3.8+

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

```bash
# Clone the repo and place the dataset CSV in the same folder
jupyter notebook credit_limit_final.ipynb
```

Run all cells top to bottom. The notebook is self-contained and will reproduce all plots and model outputs.

---

## 🛠 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, modelling, evaluation |

---

## 📋 Results Summary

| Metric | Value |
|---|---|
| R² (test set) | **0.47** |
| R² (5-fold CV) | **0.46 ± 0.01** |
| RMSE (test) | ~92,000 NTD |
| MAE (test) | ~67,000 NTD |
| Features (after engineering) | 155+ |
| Training set size | 24,000 |
| Test set size | 6,000 |

---

## 📎 Presentation

The `Credit_Limit_Prediction_Insights.pptx` file contains a 10-slide summary of this project, written for a non-technical audience. It covers the problem statement, EDA findings with chart interpretations, feature engineering rationale, model results, and key business takeaways.
