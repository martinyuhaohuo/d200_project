import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class Create_Lag(BaseEstimator, TransformerMixin):
    def __init__(self, max_lag):
        self.max_lag = max_lag
    
    def set_output(self, *, transform):
        return self
    
    def fit(self, X, y = None):
        self.columns = X.columns
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in X.columns:
            for i in range(1, self.max_lag + 1):
                X_transformed[f"{column}_{i}L"] = X[column].shift(i)
        return X_transformed


def create_fe_pipeline():

    fe_pipeline = Pipeline(
        [
            ("missing_impute", IterativeImputer(missing_values = np.nan, random_state = 78392)),
            ("min_max_scale", MinMaxScaler(feature_range=(-1,1)))
            # ("standard_scale", StandardScaler())
        ]
    )
    
    fe_pipeline.set_output(transform="pandas")
    return fe_pipeline