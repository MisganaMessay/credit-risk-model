import pytest
import pandas as pd
import numpy as np
from src.data_processing import DateFeatureExtractor

def test_date_feature_extraction():
    """Test if the transformer correctly extracts the hour from a timestamp."""
    df = pd.DataFrame({
        'TransactionStartTime': ['2026-06-01T15:30:00Z']
    })
    extractor = DateFeatureExtractor()
    transformed = extractor.transform(df)
    
    # Check if 'TransactionHour' was created
    assert 'TransactionHour' in transformed.columns
    # Check if value is correct (15)
    assert transformed['TransactionHour'].iloc[0] == 15

def test_dataframe_columns():
    """Ensure the transformer drops the original timestamp column."""
    df = pd.DataFrame({
        'TransactionStartTime': ['2026-06-01T15:30:00Z']
    })
    extractor = DateFeatureExtractor()
    transformed = extractor.transform(df)
    
    assert 'TransactionStartTime' not in transformed.columns