import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from inflation_forecast.model import create_MLP, create_RNN, create_data_loaders, nn_training, make_predict

def compute_MSE(prediction, target):
    target = target.to_numpy()
    MSE = ((prediction - target)**2).mean()
    RMSE = MSE**0.5
    return MSE, RMSE

def cross_validation_loss(model, model_type, x_train, y_train, n_split = 5, batch_size = None, epochs = None, lr = None, nodes = None, hidden_size = None, window_size = None):
    time_series_cv = TimeSeriesSplit(n_splits = n_split)
    loss_list = []
    for train_index, val_index in time_series_cv.split(x_train):
        x_tra = x_train.iloc[train_index]
        y_tra = y_train.iloc[train_index]
        x_val = x_train.iloc[val_index]
        y_val = y_train.iloc[val_index]

        if model_type == "scikit":
            model.fit(x_tra, y_tra)
            prediction = model.predict(x_val)
        elif model_type == "MLP":
            train_loader, test_loader = create_data_loaders(x_tra, y_tra, x_val, y_val, batch_size)
            MLP = create_MLP(x_tra, y_tra, nodes)
            _, _ = nn_training(MLP, train_loader, test_loader, epochs, lr, print_error = False)
            prediction = make_predict(MLP, x_tra, x_val, True)
        elif model_type == "RNN":
            train_loader, test_loader = create_data_loaders(x_tra, y_tra, x_val, y_val, batch_size, window_size)
            RNN = create_RNN(x_tra, y_tra, hidden_size, nodes)
            _, _ = nn_training(RNN, train_loader, test_loader, epochs, lr, print_error = False)
            prediction = make_predict(RNN, x_tra, x_val, True, window_size)
        
        MSE = ((prediction - y_val.to_numpy())**2).mean()
        loss_list.append(MSE)
    CV_loss = np.array(loss_list).mean()
    return CV_loss





