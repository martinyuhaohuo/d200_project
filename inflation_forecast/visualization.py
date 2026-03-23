import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# record recession periods
recession_period = [
    ["1960-05-01", "1961-02-01"], 
    ["1970-01-01", "1970-11-01"], 
    ["1973-12-01", "1975-03-01"], 
    ["1980-02-01", "1980-07-01"], 
    ["1981-08-01", "1982-11-01"], 
    ["1990-08-01", "1991-03-01"], 
    ["2001-04-01","2001-11-01"],
    ["2008-01-01","2009-06-01"],
    ["2020-03-01","2020-04-01"]
    ]
i = 0
for period in recession_period:
    recession_period[i] = [
        pd.to_datetime(period[0]), 
        pd.to_datetime(period[1])
        ]
    i += 1


def visualize_series(
        dataframe: pd.DataFrame, 
        columns: list, 
        y_label: str, 
        y_min: float, 
        y_max: float, 
        date: pd.Series | None = None, 
        recession: bool = True
        ) -> None:
    """
    Plot one or more series over time with optional recession periods

    Parameters
    ----------
    dataframe : pd.DataFrame
        the dataframe containing series to plot
    columns : list
        list of column names of series to plot
    y_label : str
        label for yaxis
    y_min : float
        lower bound of y axis
    y_max : float
        upper bound of y axis
    date : pd.Series or None, default = None
        date values used for x axis
    recession : bool, default = True
        whether or not to plot recession periods
    
    Returns
    -------
    None
        this function plot series without returning values
    """
    # figure set up
    fig = plt.figure(figsize = (10,5),dpi = 150)
    ax = plt.axes()
    color_shape = [
        ["red", "-"], 
        ["blue", "dotted"], 
        ["orange", "dotted"], 
        ["green", "dotted"]
        ]

    # plot the series
    if date is None:
        date = dataframe["sasdate"]
    else:
        date = date
    for column, color_shape in zip(columns, color_shape):
        plt.plot(
            date, 
            dataframe[column], 
            linestyle = color_shape[1], 
            markersize = 3, 
            linewidth = 1, 
            label = column, 
            color = color_shape[0]
            )
    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel("Date")
    plt.ylim(ymin = y_min, ymax = y_max)

    # plot recession periods
    if recession:
        for period in recession_period:
            plt.axvspan(period[0],period[1], color = "grey", alpha = 0.2)


def visualize_loss(
        train_loss_list: list[float], 
        test_loss_list: list[float], 
        error_metric: str = "MSE"
        ) -> None:
    """
    Plot training and testing loss over epochs

    Parameters
    ----------
    train_loss_list: list[float]
        list of train loss values
    test_loss_list: list[float]
        list of test loss values
    error_metric: str, default = "MSE"
        name of error metric used for y axis
    
    Returns
    -------
    None
        this function plot without returning values
    """
    fig = plt.figure(figsize = (10,5),dpi = 150)
    plt.plot(
        range(len(train_loss_list)), 
        train_loss_list, 
        label="Train MSE", 
        color="red"
        )
    plt.plot(
        range(len(test_loss_list)), 
        test_loss_list, 
        label="Test MSE", 
        color="blue"
        )
    plt.ylabel(f"{error_metric} Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def visualize_predict(
        prediction : np.ndarray, 
        true_value : pd.DataFrame, 
        lag : int, 
        FRED_dataframe : pd.DataFrame
        ) -> None:
    """
    Plot prediction versus actual target value
    
    Parameters
    ----------
    prediction : np.ndarray
        predicted values for y
    true_value : pd.DataFrame
        actual value for y, a pd.DataFrame with one column
    lag : int
        forecast horizon, used for labelling
    FRED_dataframe : pd.DataFrame
        original dataset with "sasdate" column

    """
    y_name = true_value.columns[0]
    Y_date = true_value.merge(
        FRED_dataframe["sasdate"], 
        left_index=True, 
        right_index=True, 
        how="left"
        )
    Y_date[f"CPIAUCSL_{lag}HP"] = prediction
    visualize_series(
        Y_date, 
        [y_name, f"CPIAUCSL_{lag}HP"], 
        "Scaled Inflation", 
        -1, 
        1, 
        recession = False
        )