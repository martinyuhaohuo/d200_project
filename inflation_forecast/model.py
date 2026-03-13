import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps") # Apple Silicon Metal Performance Shaders
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda") # GPU
else:
    DEVICE = torch.device("cpu") # CPU


def create_tensor(X, Y):
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    Y_tensor = torch.tensor(Y.to_numpy(), dtype=torch.float32)
    return X_tensor, Y_tensor


def create_MLP(x_train, y_train, nodes = 16):
    MLP = nn.Sequential(
    nn.Linear(x_train.shape[1], nodes), 
    nn.ReLU(), 
    nn.Dropout(0.2),
    # nn.Linear(nodes, nodes), 
    # nn.ReLU(),
    # nn.Dropout(0.2),
    nn.Linear(nodes, y_train.shape[1])
    )
    MLP = MLP.to(DEVICE)
    return MLP


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nodes):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, nodes),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(nodes, output_size)
        )
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.ffn(last_hidden)
        return output


def create_RNN(x_train, y_train, hidden_size = 16, nodes = 8):
    RNN = LSTM(x_train.shape[1], y_train.shape[1], hidden_size, nodes)
    RNN = RNN.to(DEVICE)
    return RNN


class WindowData(Dataset):
    def __init__(self, X, Y, window):
        self.X = X
        self.Y = Y
        self.window = window

    def __len__(self):
        return len(self.X) - self.window + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.window]
        Y_target = self.Y[idx + self.window - 1]
        return X_seq, Y_target


def create_data_loaders(x_train, y_train, x_test, y_test, batch_size, window_size = None):
    X_train, Y_train = create_tensor(x_train, y_train)
    X_test, Y_test = create_tensor(x_test, y_test)
    if window_size is None:
        train = TensorDataset(X_train, Y_train)
        test = TensorDataset(X_test, Y_test)
    else:
        train = WindowData(X_train, Y_train, window_size)
        test = WindowData(X_test, Y_test, window_size)
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)
    return train_loader, test_loader


def compute_accuracy(model, data_loader):
    criterion = nn.MSELoss(reduction="sum")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_element = 0
        for X, Y in data_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, Y)
            total_loss += loss.item()
            total_element += Y.numel()
        average_MSE = total_loss/total_element
    return average_MSE


def nn_training(model, train_loader, test_loader, epochs, learning_rate, print_error = False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_loss_list = []
    test_loss_list = []
    model.train()
    for epoch in range(epochs):

        MLP_loss_list = list()

        for batch, (X, Y) in enumerate(train_loader):
            outputs = model(X.to(DEVICE))
            loss = criterion(outputs, Y.to(DEVICE))
            optimizer.zero_grad()  # clear gradients before each backward (otw. they accumulate)
            loss.backward()        # fill gradients for all parameters with requires_grad=True
            optimizer.step()       # update parameters using those gradients
            MLP_loss_list.append(loss.item())

        train_loss = compute_accuracy(model, train_loader)
        train_loss_list.append(train_loss)
        test_loss = compute_accuracy(model, test_loader)
        test_loss_list.append(test_loss)
        if print_error:
            print(f'Epoch [{epoch + 1}/{epochs}], Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}', )
    
    # print(f'Final Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}', )

    return train_loss_list, test_loss_list


def make_predict(model, x_train, x_test, is_test = True, window_size = None):
    if window_size is None:
        if is_test:
            X = x_test
        else:
            X = x_train
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        model.eval()
        prediction = model(X_tensor.to(DEVICE))
        prediction = prediction.detach().to("cpu").numpy()
    else:
        if is_test:
            X = pd.concat([x_train.tail(window_size), x_test], axis=0)
            X = X.to_numpy()
            X_tensor = []
            for i in range(len(X) - window_size):
                X_tensor.append(X[i:i + window_size])
        else:
            X = x_train.to_numpy()
            X_tensor = []
            for i in range(len(X) - window_size + 1):
                X_tensor.append(X[i:i + window_size])
        X_tensor = np.array(X_tensor)
        X_tensor = torch.tensor(X_tensor, dtype=torch.float32).to(DEVICE)
        model.eval()
        prediction = model(X_tensor)
        prediction = prediction.detach().to("cpu").numpy()

    return prediction


def MLP_ensemble(nodes, lr, epochs, model_num, MLP_x_train, MLP_y_train, MLP_x_test, MLP_y_test):
    MLP_models = {}
    predictions = list()
    for i in range(model_num):
        train_loader, test_loader = create_data_loaders(MLP_x_train, MLP_y_train, MLP_x_test, MLP_y_test, 128)
        MLP_models[i] = create_MLP(MLP_x_train, MLP_y_train, nodes)
        _, _ = nn_training(MLP_models[i], train_loader, test_loader, epochs, lr, print_error = False)
        prediction = make_predict(MLP_models[i], MLP_x_train, MLP_x_test, True)
        predictions.append(prediction)
    predictions = np.array(predictions)
    ensemble_prediction = predictions.mean(axis=0)
    return ensemble_prediction, MLP_models


def RNN_ensemble(nodes, hidden_size, window_size, model_num, RNN_x_train, RNN_y_train, RNN_x_test, RNN_y_test):
    RNN_models = {}
    predictions = list()
    for i in range(model_num):
        train_loader, test_loader = create_data_loaders(RNN_x_train, RNN_y_train, RNN_x_test, RNN_y_test, 128, window_size)
        RNN_models[i] = create_RNN(RNN_x_train, RNN_y_train, hidden_size, nodes)
        _, _ = nn_training(RNN_models[i], train_loader, test_loader, 100, 0.001, print_error = False)
        prediction = make_predict(RNN_models[i], RNN_x_train, RNN_x_test, True, window_size)
        predictions.append(prediction)
    predictions = np.array(predictions)
    ensemble_prediction = predictions.mean(axis=0)
    return ensemble_prediction, RNN_models