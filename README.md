# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end credit risk scoring system for Bati Bank using transaction data from the Xente eCommerce platform. The objective is to build a machine learning solution that predicts customer credit risk using alternative behavioral data.

Since the dataset does not contain actual loan default labels, a proxy target variable will be engineered using customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) analysis.

The final solution will include feature engineering, risk modeling, model deployment through a REST API, and automated CI/CD workflows.

---

# Credit Scoring Business Understanding

## How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord emphasizes accurate risk measurement, transparency, and regulatory compliance. Credit scoring models used by financial institutions must therefore be interpretable and properly documented. An interpretable model allows risk managers, auditors, and regulators to understand how risk predictions are generated and verify that lending decisions are fair and consistent.

Well-documented models support governance, validation, monitoring, and reproducibility. For Bati Bank, compliance with Basel II requires that model assumptions, feature engineering decisions, and evaluation procedures be clearly documented and justified.

## Without a direct default label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The Xente transaction dataset does not contain information indicating whether customers repaid or defaulted on loans. Because supervised machine learning models require labeled examples, a proxy target variable must be created.

This project will use customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) metrics to identify customer segments that may represent higher credit risk. Customers belonging to the least engaged segment will be labeled as high-risk.

However, proxy variables are assumptions rather than actual default outcomes. Some customers labeled as high-risk may never default, while some low-risk customers may eventually default. This introduces risks such as misclassification, biased decisions, and reduced predictive accuracy. Therefore, proxy-based predictions should be interpreted carefully and continuously monitored.

## What are the key trade-offs between a simple interpretable model and a high-performance model in a regulated financial context?

Logistic Regression combined with Weight of Evidence (WoE) transformation offers strong interpretability and is widely used in traditional credit scoring. The contribution of each variable can be easily explained to regulators and business stakeholders.

Gradient Boosting models often provide superior predictive performance because they can capture complex nonlinear relationships and feature interactions. However, they are more difficult to interpret and validate.

In regulated financial environments, organizations must balance predictive performance with transparency, explainability, and regulatory compliance. The final model choice should consider both business objectives and governance requirements.
=======
# credit-risk-model
Credit Risk Probability Model for Alternative Data
>>>>>>> 71b1aaf677add65ec95021db58585137729bf425
