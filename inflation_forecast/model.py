import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset


# Set Up Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps") # Apple MPS
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda") # GPU
else:
    DEVICE = torch.device("cpu") # CPU


def create_tensor(
        X : pd.DataFrame, 
        Y : pd.DataFrame
        ) -> tuple[Tensor, Tensor]:
    """
    Convert feature and target dataframes to torch.Tensor

    Parameters
    ----------
    X : pd.DataFrame
        the feature dataframe
    Y : pd.DataFrame
        the target dataframe

    Returns
    -------
    tuple[Tensor, Tensor]
        Tensors for features and target
    """
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    Y_tensor = torch.tensor(Y.to_numpy(), dtype=torch.float32)
    return X_tensor, Y_tensor


def create_MLP(
        x_train : pd.DataFrame, 
        y_train : pd.DataFrame, 
        nodes : int = 16
        ) -> nn.Sequential:
    """
    Create a MLP with one hidden layer, n nodes, and ReLU activation function

    Parameters
    ----------
    x_train : pd.DataFrame
        the feature dataframe for training sample
    y_train : pd.DataFrame
        the target dataframe for training sample
    nodes : int, default = 16
        number of hidden units in hidden layer

    Returns
    -------
    nn.Sequential
        the one layer MLP model
    """
    MLP = nn.Sequential(
    nn.Linear(x_train.shape[1], nodes), 
    nn.ReLU(), 
    nn.Dropout(0.2),
    nn.Linear(nodes, y_train.shape[1])
    )
    MLP = MLP.to(DEVICE)
    return MLP


class LSTM(nn.Module):
    """
    LSTM model with one layer LSTM cell connected with one layer FNN
    """
    hidden_size: int
    lstm: nn.LSTM
    mlp: nn.Sequential

    def __init__(
            self, 
            input_size : int, 
            output_size : int, 
            hidden_size : int, 
            nodes : int
            ) -> None:
        """
        Initalize the LSTM model

        Parameters
        ----------
        input_size : int
            the dimension of input
        output_size : int
            the dimension of output
        hidden_size : int
            the size of the hidden state vector
        nodes : int
            the number of hidden units in the FNN
        
        Returns
        -------
        None
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # The LSTM cell
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True
        )
        
        # The FNN with one hidden layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, nodes),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(nodes, output_size)
        )
    
    def forward(self, x : Tensor) -> Tensor:
        """
        Compute forward pass for the LSTM model

        Parameters
        ----------
        x : Tensor
            the Tensor of input features
        
        Returns
        -------
        Tensor
            the Tensor of target prediction
        """
        _, (h, _) = self.lstm(x)
        last_hidden = h[-1]
        output = self.mlp(last_hidden)
        return output


def create_RNN(
        x_train : pd.DataFrame, 
        y_train : pd.DataFrame, 
        hidden_size : int = 16,
        nodes : int = 8
        ) -> LSTM:
    """
    Creates a LSTM based on input dimension of X and output dimension of Y, 
    the model is moved to DEVICE
    
    Parameters
    ----------
    x_train : pd.DataFrame
        the feature dataframe for training sample
    y_train : pd.DataFrame
        the target dataframe for training sample
    hidden_size : int, default = 16
        the size of the vector of hidden state
    nodes : int, default = 8
        the number of hidden units in the FNN component
        
    Returns
    -------
    LSTM
        the LSTM model created
    """
    RNN = LSTM(x_train.shape[1], y_train.shape[1], hidden_size, nodes)
    RNN = RNN.to(DEVICE)
    return RNN


class WindowData(Dataset):
    """
    A custom Pytorch dataset object for sequential time series input
    """
    X : Tensor
    Y : Tensor
    window : int

    def __init__(self, X : Tensor, Y : Tensor, window : int) -> None:
        """
        Initalize the custom dataset

        Parameters
        ----------
        X : Tensor
            input tensor of features in a sample
        Y : Tensor
            input tensor of targets of a sample
        window : int
            the window length of the sequence
        
        Returns
        -------
        None
        """
        self.X = X
        self.Y = Y
        self.window = window

    def __len__(self) -> int:
        """
        Return the number of avaliable sequences can be drawn
        
        Returns
        -------
        int
            the number of sequences can be drawn
        """
        return len(self.X) - self.window + 1

    def __getitem__(self, idx : int) -> tuple[Tensor, Tensor]:
        """
        Return one sequence as input for the model and the target associated

        Parameters
        ----------
        idx : int
            the index for the sequence
        
        Returns
        -------
        tuple[Tensor, Tensor]
            a tuple of a sequence tensor and a target tensor
        """
        X_seq = self.X[idx:idx + self.window]
        Y_target = self.Y[idx + self.window - 1]
        return X_seq, Y_target


def create_data_loaders(
        x_train : pd.DataFrame, 
        y_train : pd.DataFrame, 
        x_test : pd.DataFrame, 
        y_test : pd.DataFrame, 
        batch_size : int, 
        window_size : int | None = None
        ) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch dataloader for train and test sets

    Parameters
    ----------
    x_train : pd.DataFrame
        dataframe of features in train set
    y_train : pd.DataFrame
        dataframe of target in train set
    x_test : pd.DataFrame
        dataframe of features in test set
    y_test : pd.DataFrame
        dataframe of target in test set
    batch_size : int
        the batch size of data loader
    window_size : int | None, default = None
        the window size of input sequence for LSTM
        
    Returns
    -------
    tuple[DataLoader, DataLoader]
        a tuple of training and testing data loaders
    """
    X_train, Y_train = create_tensor(x_train, y_train)
    X_test, Y_test = create_tensor(x_test, y_test)

    # if window_size is None, the model is FNN, use TensorDataset
    if window_size is None:
        train = TensorDataset(X_train, Y_train)
        test = TensorDataset(X_test, Y_test)
    
    # if window_size is not None, the model is LSTM, use custom dataset class
    else:
        train = WindowData(X_train, Y_train, window_size)
        test = WindowData(X_test, Y_test, window_size)
    
    # create data loader for train and test set
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader


def compute_accuracy(
        model : nn.Module, 
        data_loader : DataLoader, 
        DL : bool = False
        ) -> float:
    """
    Evaluates the MSE for a trained neural network based on a dataset
    
    Parameters
    ----------
    model : nn.Module
        trained neural network
    data_loader : DataLoader
        data loader of a dataset (train or test set)
    DL : bool, default = False
        indicates whether the model is Dlinear

    Returns
    -------
    float
        MSE over the dataset
    """
    criterion = nn.MSELoss(reduction="sum")
    # set model to eval mode to remove the effect of hidden unit drop out
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_element = 0
        for X, Y in data_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            outputs = model(X)
            if DL:
                loss = criterion(outputs, Y.unsqueeze(-1))
            else:
                loss = criterion(outputs, Y)
            total_loss += loss.item()
            total_element += Y.numel()
        average_MSE = total_loss/total_element
    return average_MSE


def nn_training(
        model : nn.Module, 
        train_loader : DataLoader, 
        test_loader : DataLoader, 
        epochs : int, 
        learning_rate : float, 
        print_error : bool = False, 
        DL : bool = False
        ) -> tuple[list[float], list[float]]:
    """
    Train the neural network and record train and test loss at each epoch

    Parameters
    ----------
    model : nn.Module
        the neural network model need to train
    train_loader : DataLoader
        the data loader for train set
    test_loader : DataLoader
        the data loader for test set
    epochs : int
        the number of epochs of training
    learning_rate : float
        the learning rate for training
    print_error : bool, default = False
        whether or not to print train and test loss for each epoch
    DL : bool, default = False
        whether or not the model is a Dlinear

    Returns
    -------
    tuple[list[float], list[float]]
        the list of loss on train set and loss on test set
    """
    # take MSE as loss function, use Adam as optimization algorithm
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr = learning_rate,
        weight_decay = 1e-4 # add weight penalty to reduce over-fitting
        )

    # set up train and test loss
    train_loss_list = []
    test_loss_list = []

    # set model to train mode
    model.train()

    # for each epoch, iterate through each batch
    for epoch in range(epochs):

        # for each batch, predicts Ys, compute MSE, evaluate gradient 
        # and update parameter
        for _, (X, Y) in enumerate(train_loader):
            outputs = model(X.to(DEVICE))
            if DL:
                loss = criterion(outputs, Y.to(DEVICE).unsqueeze(-1))
            else:
                loss = criterion(outputs, Y.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute train MSE and test MSE for each epoch
        train_loss = compute_accuracy(model, train_loader, DL)
        train_loss_list.append(train_loss)
        test_loss = compute_accuracy(model, test_loader, DL)
        test_loss_list.append(test_loss)
        if print_error:
            print(
                f"Epoch [{epoch + 1}/{epochs}], " 
                + f"Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}"
                )

    return train_loss_list, test_loss_list


def make_predict(
        model : nn.Module, 
        x_train : pd.DataFrame, 
        x_test : pd.DataFrame, 
        is_test : bool = True, 
        window_size : int | None = None
        ):
    """
    Generate prediction from a trained neural network

    Parameters
    ----------
    model : nn.Module
        the trained neural network
    x_train : pd.DataFrame
        the dataframe of feature in train set
    x_test : pd.DataFrame
        the dataframe of feature in test set
    is_test : bool, default = True
        whether the prediction is for test set
    window_size : int | None, default = None
        the window size for LSTM
    
    Returns
    -------
    np.ndarray
        the array of model predictions
    """
    # for FNN, get prediction is straight
    if window_size is None:
        if is_test:
            X = x_test
        else:
            X = x_train
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        model.eval()
        prediction = model(X_tensor.to(DEVICE))
        prediction = prediction.detach().to("cpu").numpy()
    
    # for LSTM, need to create a series of sequence as input
    else:
        if is_test:
            # for test prediction, first few obs need to include 
            # last few obs of train set in the input sequence
            X = pd.concat([x_train.tail(window_size - 1), x_test], axis=0)
            X = X.to_numpy()
            X_tensor = []
            for i in range(len(X) - window_size + 1):
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


def MLP_ensemble(
        nodes : int, 
        lr : float, 
        epochs : int, 
        model_num : int, 
        MLP_x_train : pd.DataFrame, 
        MLP_y_train : pd.DataFrame, 
        MLP_x_test : pd.DataFrame, 
        MLP_y_test : pd.DataFrame
        ) -> tuple[np.ndarray, dict[int, nn.Module]]:
    """
    Train an ensemble of MLPs and average their predictions

    Parameters
    ----------
    nodes : int
        number of hiddens units
    lr : float
        learning rate
    epochs : int
        training epochs
    model_num : int
        number of models to ensemble
    MLP_x_train : pd.DataFrame
        the dataframe of feature in train set
    MLP_y_train : pd.DataFrame
        the dataframe of target in train set
    MLP_x_test : pd.DataFrame
        the dataframe of feature in test set
    MLP_y_test : pd.DataFrame
        the dataframe of target in test set
    
    Returns
    -------
    tuple[np.ndarray, dict[int, nn.Module]]
        a tuple of an array of ensembled prediction and a dictionary of models
    """
    MLP_models = {}
    predictions = list()
    for i in range(model_num):
        train_loader, test_loader = create_data_loaders(
            MLP_x_train, 
            MLP_y_train, 
            MLP_x_test, 
            MLP_y_test, 
            128
            )
        MLP_models[i] = create_MLP(MLP_x_train, MLP_y_train, nodes)
        _, _ = nn_training(
            MLP_models[i], 
            train_loader, 
            test_loader, 
            epochs, 
            lr, 
            print_error = False
            )
        prediction = make_predict(MLP_models[i], MLP_x_train, MLP_x_test, True)
        predictions.append(prediction)
    predictions = np.array(predictions)
    ensemble_prediction = predictions.mean(axis=0)
    return ensemble_prediction, MLP_models


def RNN_ensemble(
        nodes : int, 
        hidden_size : int, 
        window_size : int, 
        model_num : int, 
        RNN_x_train : pd.DataFrame, 
        RNN_y_train : pd.DataFrame, 
        RNN_x_test : pd.DataFrame, 
        RNN_y_test : pd.DataFrame
        ) -> tuple[np.ndarray, dict[int, nn.Module]]:
    """
    Train an ensemble of LSTMs and average their predictions

    Parameters
    ----------
    nodes : int
        number of hiddens units
    hidden_size : int
        size of the vector of hidden state
    window_size : int
        the size of input sequence
    model_num : int
        number of models to ensemble
    RNN_x_train : pd.DataFrame
        the dataframe of feature in train set
    RNN_y_train : pd.DataFrame
        the dataframe of target in train set
    RNN_x_test : pd.DataFrame
        the dataframe of feature in test set
    RNN_y_test : pd.DataFrame
        the dataframe of target in test set
    
    Returns
    -------
    tuple[np.ndarray, dict[int, nn.Module]]
        a tuple of an array of ensembled prediction and a dictionary of models
    """
    RNN_models = {}
    predictions = list()
    for i in range(model_num):
        train_loader, test_loader = create_data_loaders(
            RNN_x_train, 
            RNN_y_train, 
            RNN_x_test, 
            RNN_y_test, 
            128, 
            window_size
            )
        RNN_models[i] = create_RNN(
            RNN_x_train, 
            RNN_y_train, 
            hidden_size, 
            nodes
            )
        _, _ = nn_training(
            RNN_models[i], 
            train_loader, 
            test_loader, 
            100, 
            0.001, 
            print_error = False
            )
        prediction = make_predict(
            RNN_models[i], 
            RNN_x_train, 
            RNN_x_test, 
            True, 
            window_size
            )
        predictions.append(prediction)
    predictions = np.array(predictions)
    ensemble_prediction = predictions.mean(axis=0)
    return ensemble_prediction, RNN_models


def DL_ensemble(
        kernel_size : int, 
        individual : bool, 
        window_size : int, 
        model_num : int, 
        DL_x_train : pd.DataFrame, 
        DL_y_train : pd.DataFrame, 
        DL_x_test : pd.DataFrame, 
        DL_y_test : pd.DataFrame
        ) -> tuple[np.ndarray, dict[int, nn.Module]]:
    """
    Train an ensemble of Dlinears and average their predictions

    Parameters
    ----------
    kernel_size : int
        number of kernels
    individual : bool
        whether make individual layer for each channel or not
    window_size : int
        the size of input sequence
    model_num : int
        number of models to ensemble
    DL_x_train : pd.DataFrame
        the dataframe of feature in train set
    DL_y_train : pd.DataFrame
        the dataframe of target in train set
    DL_x_test : pd.DataFrame
        the dataframe of feature in test set
    DL_y_test : pd.DataFrame
        the dataframe of target in test set
    
    Returns
    -------
    tuple[np.ndarray, dict[int, nn.Module]]
        a tuple of an array of ensembled prediction and a dictionary of models
    """
    DL_models = {}
    predictions = list()
    for i in range(model_num):
        train_loader, test_loader = create_data_loaders(
            DL_x_train, 
            DL_y_train, 
            DL_x_test, 
            DL_y_test, 
            128, 
            window_size
            )
        DL_models[i] = create_DL(
            DL_x_train, 
            DL_y_train, 
            window_size, 
            individual, 
            kernel_size
            )
        _, _ = nn_training(
            DL_models[i], 
            train_loader, 
            test_loader, 
            100, 
            0.001, 
            print_error = False
            )
        prediction = make_predict(
            DL_models[i], 
            DL_x_train, 
            DL_x_test, 
            True, 
            window_size
            )
        predictions.append(prediction)
    predictions = np.array(predictions)
    ensemble_prediction = predictions.mean(axis=0)
    return ensemble_prediction, DL_models


###############################################################################
## This part of code comes from (Zeng et al., 2023)
## It is the arhcitecture of Dlinear
## Please refer to https://github.com/vivva/DLinear
## Although included in the notebook, result from Dlinear as benchamrk is not 
## analysed in the report & presentation due to time and space constraint
###############################################################################

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0
            )

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Dlinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, enc_in, pred_len, individual, kernel_size):
        super(Dlinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(self.kernel_size)
        self.individual = individual
        self.channels = enc_in
        

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len,self.pred_len)
                    )
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len,self.pred_len)
                    )

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0,2,1)
        trend_init = trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0),seasonal_init.size(1),self.pred_len],
                dtype=seasonal_init.dtype
                ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0),trend_init.size(1),self.pred_len],
                dtype=trend_init.dtype
                ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](
                    seasonal_init[:,i,:]
                    )
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0,2,1)
        output = x
        output = output.squeeze(-1)
        return  output # to [Batch, Output length, Channel]

def create_DL(x_train, y_train, window_size, individual, kernel_size):
    DL = Dlinear(
        window_size, 
        x_train.shape[1], 
        y_train.shape[1], 
        individual, 
        kernel_size
        )
    DL = DL.to(DEVICE)
    return DL

###############################################################################
## End of the part ############################################################
###############################################################################