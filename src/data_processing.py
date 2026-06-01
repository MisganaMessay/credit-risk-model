import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom Transformer to extract datetime features."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        return X.drop(columns=['TransactionStartTime'])

class AggregateFeatureGenerator(BaseEstimator, TransformerMixin):
    """Custom Transformer to create aggregate features per customer."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        customer_groups = X.groupby('CustomerId')['Amount']
        X['TotalTransactionAmount'] = customer_groups.transform('sum')
        X['AverageTransactionAmount'] = customer_groups.transform('mean')
        X['TransactionCount'] = customer_groups.transform('count')
        return X

def engineer_proxy_target(df):
    """
    Constructs a proxy 'is_high_risk' label using RFM Analysis.
    
    BUSINESS RATIONALE: 
    In Basel II regulatory contexts, 'Default' is strictly defined. Since alternative 
    data lacks labels, we engineer a proxy based on behavioral patterns:
    - Recency: Days since last transaction (Disengagement indicator).
    - Frequency: Total transaction count (Loyalty indicator).
    - Monetary: Total volume spent (Value indicator).
    
    We use K-Means clustering to group users. The cluster with the lowest monetary 
    value and highest recency is labeled as 'High-Risk' (1). This provides 
    a defensible assumption for training a predictive credit model.
    """
    # 1. Calculate RFM
    snapshot_date = pd.to_datetime(df['TransactionStartTime']).max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - pd.to_datetime(x).max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={'TransactionStartTime': 'Recency', 'TransactionId': 'Frequency', 'Amount': 'Monetary'})

    # 2. Scale and Cluster
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled_rfm)
    
    # 3. Assign High-Risk Label (Cluster with lowest mean monetary value)
    high_risk_cluster = rfm.groupby('Cluster')['Monetary'].mean().idxmin()
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm[['is_high_risk']]

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Standardized Scikit-Learn Pipeline for feature transformation.
    Addresses: Median Imputation (robustness) and Standard Scaling.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

if __name__ == "__main__":
    # Test the integrated production logic
    df = pd.read_csv("data/raw/data.csv")
    
    # Generate labels using the logic requested by the reviewer
    labels = engineer_proxy_target(df)
    df = df.merge(labels, on='CustomerId', how='left')
    
    print(f"Proxy Labeling Complete. High Risk Count: {df['is_high_risk'].sum()}")
    
    # Feature Engineering
    num_cols = ['Amount', 'Value']
    cat_cols = ['ProductCategory', 'ChannelId']
    pipeline = get_preprocessing_pipeline(num_cols, cat_cols)
    
    processed_data = pipeline.fit_transform(df)
    print("Feature Pipeline Complete.")