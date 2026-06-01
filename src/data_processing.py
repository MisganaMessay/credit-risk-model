import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X.drop(columns=['TransactionStartTime'])

class AggregateFeatureGenerator(BaseEstimator, TransformerMixin):
    """Custom Transformer to create aggregate features per customer."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Aggregating by CustomerId
        customer_groups = X.groupby('CustomerId')['Amount']
        X['TotalTransactionAmount'] = customer_groups.transform('sum')
        X['AverageTransactionAmount'] = customer_groups.transform('mean')
        X['TransactionCount'] = customer_groups.transform('count')
        X['StdTransactionAmount'] = customer_groups.transform('std').fillna(0)
        return X

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Constructs a ColumnTransformer within a Pipeline.
    Addresses: Imputation Strategy, Scaling, and Encoding.
    """
    # 1. Numerical Pipeline: Median Imputation + Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline: Mode Imputation + OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop IDs and raw timestamps after processing
    )

    # Full Pipeline
    full_pipeline = Pipeline(steps=[
        ('date_features', DateFeatureExtractor()),
        ('agg_features', AggregateFeatureGenerator()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    # Test the pipeline
    try:
        df = pd.read_csv("data/raw/data.csv")
        num_cols = ['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount']
        cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
        
        pipeline = get_preprocessing_pipeline(num_cols, cat_cols)
        processed_data = pipeline.fit_transform(df)
        
        print(f"Task 3 Complete. Processed shape: {processed_data.shape}")
        # Save processed data for Task 4
        pd.DataFrame(processed_data).to_csv("data/processed/model_ready.csv", index=False)
    except Exception as e:
        print(f"Error in pipeline: {e}")