import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


def x_fe_pipeline(explained_var = 0.8):

    fe_pipeline = Pipeline(
        [
            ("missing_impute", IterativeImputer(missing_values = np.nan, random_state = 78392)),
            ("standarlize", StandardScaler()),
            ("PCA", PCA(n_components=explained_var)),
            ("min_max_scale", MinMaxScaler(feature_range=(-1,1)))
            
        ]
    )
    
    fe_pipeline.set_output(transform="pandas")
    return fe_pipeline

def y_fe_pipeline():

    fe_pipeline = Pipeline(
        [
            ("missing_impute", IterativeImputer(missing_values = np.nan, random_state = 78392)),
            ("min_max_scale", MinMaxScaler(feature_range=(-1,1)))
        ]
    )
    
    fe_pipeline.set_output(transform="pandas")
    return fe_pipeline


def create_ahead_lag(y_train, x_train, y_test, x_test, target_ahead, max_lag, include_target = False, target_lag = 0):

    x_dataframe = pd.concat([x_train, x_test], axis=0)
    y_dataframe = pd.concat([y_train, y_test], axis=0)
    target = y_dataframe.columns[0]
    dataframe = pd.concat([x_dataframe, y_dataframe], axis=1)
    new_frame = dataframe.copy(deep = True)

    # create ahead prediction target
    new_frame[f"{target}_{target_ahead}H"] = new_frame[target].shift(-target_ahead)

    # if max_lag > 0, create lagged predictor
    if max_lag > 0:
        for i in range(1, max_lag + 1):
            for column in dataframe.columns:
                if column != target:
                    new_frame[f"{column}_{i}L"] = new_frame[column].shift(i)
    # if max_lag = -1, do not include any exogenous variable in X
    if max_lag == -1:
        for column in dataframe.columns:
            if column != target:
                new_frame = new_frame.drop(columns = [column])
    # if include_target, add lags of target variable
    if include_target:
        for i in range(0, target_lag + 1):
            new_frame[f"{target}_{i}L"] = new_frame[target].shift(i)
    
    # drop rows with NA values due to lagging / creating ahead
    lagged_train_set = new_frame.loc[x_train.index]
    lag = max(max_lag, target_lag)
    lagged_train_set = lagged_train_set.iloc[lag:]
    lagged_test_set = new_frame.loc[x_test.index]
    lagged_test_set = lagged_test_set.iloc[:-target_ahead]

    # split complete dataset back to train and test set
    y_train = lagged_train_set[[f"{target}_{target_ahead}H"]]
    x_train = lagged_train_set.drop(columns = [f"{target}_{target_ahead}H", target])
    y_test = lagged_test_set[[f"{target}_{target_ahead}H"]]
    x_test = lagged_test_set.drop(columns = [f"{target}_{target_ahead}H", target])

    return y_train, x_train, y_test, x_test