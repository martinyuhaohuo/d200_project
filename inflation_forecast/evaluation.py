import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from inflation_forecast.model import (
    create_MLP, 
    create_RNN, 
    create_DL, 
    create_data_loaders, 
    nn_training, 
    make_predict
    )


def compute_MSE(
        prediction: np.array, 
        target: pd.DataFrame
        ) -> tuple[float, float]:
    """
    This function computes mean squared error based on a series of prediction 
    and a series of true values
    
    Parameters
    ----------
    prediction : np.array
        prediction made by a model
    target : pd.DataFrame
        a dataframe with one column, which is a series of actual target

    Returns
    -------
    tuple[float, float]
        the mean squared error and the root mean squared error
    """
    target = target.to_numpy()
    MSE = ((prediction - target)**2).mean()
    RMSE = MSE**0.5
    return MSE, RMSE


def cross_validation_loss(
        model : object | None, 
        model_type : str, 
        x_train : pd.DataFrame, 
        y_train : pd.DataFrame, 
        n_split : int = 5, 
        batch_size : int | None = None, 
        epochs : int | None = None, 
        lr : float | None = None, 
        nodes : int | None = None, 
        hidden_size : int | None = None, 
        window_size : int | None = None, 
        individual : bool | None = None, 
        kernel_size : int | None = None
        ):
    """
    This function computes cross-validation loss based on the training set

    Parameters
    ----------
    model : object | None
        the scikit-learn model
        if use neural network, leave this parameter as None
    model_type : str
        the model type
    x_train : pd.DataFrame
        the dataframe of features in train set
    y_train : pd.DataFrame
        the dataframe of target in train set
    n_split : int, default = 5
        the number of splits
    batch_size : int | None = None
        the batch size for NN
    epochs : int | None = None
        the training epochs for NN
    lr : float | None = None
        the learning rate for NN
    nodes : int | None = None
        the hidden units number for NN
    hidden_size : int | None = None
        the size of vector of hidden state for LSTM
    window_size : int | None = None
        the size of input sequence for LSTM
    individual : bool | None = None
        whether or not take individual layer for each channel in DL
    kernel_size : int | None = None
        the number of kernels for DL
    
    Returns
    -------
    float
        the cross validation loss
    """
    time_series_cv = TimeSeriesSplit(n_splits = n_split)
    loss_list = []
    for train_index, val_index in time_series_cv.split(x_train):
        x_tra = x_train.iloc[train_index]
        y_tra = y_train.iloc[train_index]
        x_val = x_train.iloc[val_index]
        y_val = y_train.iloc[val_index]

        # for scikit-learn and LGBM models
        if model_type == "scikit":
            model.fit(x_tra, y_tra)
            prediction = model.predict(x_val)
        
        # for MLP
        elif model_type == "MLP":
            train_loader, test_loader = create_data_loaders(
                x_tra, 
                y_tra, 
                x_val, 
                y_val, 
                batch_size
                )
            MLP = create_MLP(x_tra, y_tra, nodes)
            _, _ = nn_training(
                MLP, 
                train_loader, 
                test_loader, 
                epochs, 
                lr, 
                print_error = False
                )
            prediction = make_predict(MLP, x_tra, x_val, True)
        
        # for LSTM
        elif model_type == "RNN":
            train_loader, test_loader = create_data_loaders(
                x_tra, 
                y_tra, 
                x_val, 
                y_val, 
                batch_size, 
                window_size
                )
            RNN = create_RNN(x_tra, y_tra, hidden_size, nodes)
            _, _ = nn_training(
                RNN, 
                train_loader, 
                test_loader, 
                epochs, 
                lr, 
                print_error = False
                )
            prediction = make_predict(RNN, x_tra, x_val, True, window_size)
        
        # for Dlinear
        elif model_type == "DL":
            train_loader, test_loader = create_data_loaders(
                x_tra, 
                y_tra, 
                x_val, 
                y_val, 
                batch_size, 
                window_size
                )
            DL = create_DL(x_tra, y_tra, window_size, individual, kernel_size)
            _, _ = nn_training(
                DL, 
                train_loader, 
                test_loader, 
                epochs, 
                lr, 
                print_error = False
                )
            prediction = make_predict(DL, x_tra, x_val, True, window_size)

        # compute MSE
        MSE = ((prediction - y_val.to_numpy())**2).mean()
        loss_list.append(MSE)
    
    # average MSE across folds to get CV loss
    CV_loss = np.array(loss_list).mean()

    return CV_loss





