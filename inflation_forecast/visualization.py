import matplotlib.pyplot as plt
import pandas as pd

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
    recession_period[i] = [pd.to_datetime(period[0]), pd.to_datetime(period[1])]
    i += 1


def visualize_series(dataframe: pd.DataFrame, columns: list, y_label: str, y_min: float, y_max: float, date = None) -> None:

    fig = plt.figure(figsize = (10,5),dpi = 150)
    ax = plt.axes()
    color_shape = [["red", "-"], ["blue", "dotted"], ["orange", "--"]]

    # plot the series
    if date is None:
        date = dataframe["sasdate"]
    else:
        date = date
    for column, color_shape in zip(columns, color_shape):
        plt.plot(date, dataframe[column], linestyle = color_shape[1], markersize = 3, linewidth = 1, label = column, color = color_shape[0])
    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel("Date")
    plt.ylim(ymin = y_min, ymax = y_max)

    # plot recession periods
    for period in recession_period:
        plt.axvspan(period[0],period[1], color = "grey", alpha = 0.2)