import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


def create_data_loaders(x_train, y_train, x_test, y_test, batch_size):
    X_train, Y_train = create_tensor(x_train, y_train)
    X_test, Y_test = create_tensor(x_test, y_test)
    train = TensorDataset(X_train, Y_train)
    test = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)
    return train_loader, test_loader


def create_MLP(x_train, y_train):
    MLP = nn.Sequential(
    nn.Linear(x_train.shape[1], 32), 
    nn.ReLU(), 
    nn.Linear(32, 32), 
    nn.ReLU(),
    nn.Linear(32, 32), 
    nn.ReLU(),
    nn.Linear(32, 32), 
    nn.ReLU(),
    nn.Linear(32, y_train.shape[1])
    )
    MLP = MLP.to(DEVICE)
    return MLP


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


def MLP_training(MLP, train_loader, test_loader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(MLP.parameters(), lr=learning_rate)

    train_loss_list = []
    test_loss_list = []
    MLP.train()
    for epoch in range(epochs):

        MLP_loss_list = list()

        for batch, (X, Y) in enumerate(train_loader):
            outputs = MLP(X.to(DEVICE))
            loss = criterion(outputs, Y.to(DEVICE))
            optimizer.zero_grad()  # clear gradients before each backward (otw. they accumulate)
            loss.backward()        # fill gradients for all parameters with requires_grad=True
            optimizer.step()       # update parameters using those gradients
            MLP_loss_list.append(loss.item())

        train_loss = compute_accuracy(MLP, train_loader)
        train_loss_list.append(train_loss)
        test_loss = compute_accuracy(MLP, test_loader)
        test_loss_list.append(test_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}', )
    
    return train_loss_list, test_loss_list


def make_predict(model, x_test):
    X_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    prediction = model(X_tensor.to(DEVICE))
    prediction = pd.DataFrame(prediction.detach().to("cpu").numpy())
    return prediction
