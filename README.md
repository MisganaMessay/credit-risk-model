# Credit Risk Probability Model for Alternative Data

## 📋 Project Overview
This project is part of a partnership between **Bati Bank** and a leading eCommerce platform to enable a **Buy-Now-Pay-Later (BNPL)** service. As an Analytics Engineer, the goal is to build a high-performance, Basel II-compliant credit scoring model that predicts the probability of default for customers using alternative transaction data.

Since the raw dataset does not contain historical default labels, this project involves engineering a **proxy risk variable** using behavioral patterns (RFM analysis) to categorize users as high-risk or low-risk.

---

## 🏛️ Business Understanding (Basel II & Regulatory Context)

In a regulated financial environment, this model adheres to the following principles:

- **Basel II Accord Alignment:** We prioritize model interpretability and thorough documentation to meet regulatory expectations for risk measurement and capital adequacy.
- **Proxy Target Variable:** Without a direct "default" label, we utilize **Recency, Frequency, and Monetary (RFM)** patterns to identify disengaged or low-value customers as a proxy for credit risk.
- **Model Trade-offs:** While complex models (like Gradient Boosting) offer higher accuracy, we compare them against simpler, interpretable models (like Logistic Regression with WoE) to ensure the bank can justify credit decisions to regulators and customers.

---

## 📊 Data Source

The dataset consists of transaction-level records from the **Xente eCommerce platform**.

- **Official Dataset:** https://www.kaggle.com/c/xente-challenge  
- **Details:** The data captures the 'who' (CustomerId), 'what' (ProductCategory), and 'how much/when' (Amount, TransactionStartTime) for over 95,000 transactions.

---

## 🛠️ Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   └── api/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

Follow these steps to set up and run the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/MisganaMessay/credit-risk-model.git
cd credit-risk-model
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Jupyter Notebook (Optional)
```bash
jupyter notebook
```

---

## 📈 Key Insights from Exploratory Data Analysis (EDA)

From the initial analysis of the transaction dataset, the following patterns were observed:

- **Extreme Skewness:** Transaction Amount and Value are highly right-skewed, with significant outliers (up to very large values). This suggests the need for robust scaling or transformation techniques (e.g., log transformation).

- **Category Dominance:** A small number of categories dominate the dataset, with over 90% of transactions concentrated in *airtime* and *financial_services*.

- **Feature Redundancy:** There is a very high correlation (≈ 0.99) between *Amount* and *Value*, suggesting that one feature may be removed to avoid multicollinearity.

- **Channel Behavior Patterns:** *ChannelId_3* accounts for the majority of transaction volume.

- **Customer Behavior Insight:** Transaction patterns show clear differences in user activity levels, useful for RFM-based segmentation and proxy risk labeling.

5. **Imputation Strategy:** Designed a robust missing-data plan using Median Imputation for skewed numerical features and Mode Imputation for categorical features to ensure production pipeline stability.

---

## ⚖️ License

## 🐳 Deployment Guide (Docker)

To build and run the API locally using Docker, follow these steps:

1. **Build the Image:**
   ```bash
   docker build -t credit-risk-api .

This project is for educational purposes as part of the 10 Academy AI Mastery program.