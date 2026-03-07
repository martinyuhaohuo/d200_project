import numpy as np
import pandas as pd

def stationary_transform(raw_data : pd.DataFrame, transform_table : pd.DataFrame) -> pd.DataFrame:
    """
    This function conduct necessary transformations needed to make a series stationary (depsite for CPI)

    Parameters:
    -----------
    raw_data : pd.DataFrame
        the raw dataframe
    transform_table : pd.DataFrame
        a table that maps column name to transform code

    Returns:
    --------
    pd.DataFrame
        a transformed dataframe with stationary time series (despite CPI and PCEPI)
    """
    transformed = raw_data.copy(deep=True)
    for column in raw_data.columns:
        if column == "CPIAUCSL" or column == "PCEPI":
            # for CPI and PCEPI, return first difference of logged value (inflation rate)
            transformed[column] = np.log(raw_data[column]).diff()
        elif transform_table.loc[0,column] == 1:
            # transform code 1: return raw value
            transformed[column] = raw_data[column]
        elif transform_table.loc[0,column] == 2:
            # transform code 2: return first difference
            transformed[column] = raw_data[column].diff()
        elif transform_table.loc[0,column] == 3:
            # transform code 3: return double difference
            transformed[column] = raw_data[column].diff().diff()
        elif transform_table.loc[0,column] == 4:
            # transform code 4: return logged value
            transformed[column] = np.log(raw_data[column])
        elif transform_table.loc[0,column] == 5:
            # transform code 5: return first difference of logged value
            transformed[column] = np.log(raw_data[column]).diff()
        elif transform_table.loc[0,column] == 6:
            # transform code 6: return double difference of logged value
            transformed[column] = np.log(raw_data[column]).diff().diff()
        elif transform_table.loc[0,column] == 7:
            # transform code 7: return first difference of percentage change
            transformed[column] = raw_data[column].pct_change().diff()
    return transformed


def report_missings(dataframe: pd.DataFrame) -> str:
    """
    Summary missing values per column

    Parameters
    ----------
    dataframe : pd.DataFrame
        the input dataframe

    Returns
    -------
    str
        the formulated output
    """
    output = ""
    i = 0
    for col_name, dtype in dataframe.dtypes.items():

        missing_count = dataframe[col_name].isna().sum()
        missing_pct = missing_count / dataframe.shape[0]
        missing_count_str = f"Mis_count: {missing_count}"
        missing_pct_str = f"Mis_pct: {missing_pct}"

        output += (
            str(i)
            + " " * (5 - len(str(i)))
            + f"{col_name}"
            + " " * (35 - len(str(col_name)))
            + "|"
            + f"{dtype}"
            + " " * (30 - len(str(dtype)))
            + "|"
            + missing_count_str
            + " " * (30 - len(missing_count_str))
            + "|"
            + f"Uni_value: {missing_pct_str}"
            + "\n"
        )
        i += 1

    return output


def remove_missing_col(dataframe, threshold = 0.1):
    new_frame = dataframe.copy(deep = True)
    remov_columns = list()
    for column in dataframe.columns:
        missing_count = dataframe[column].isna().sum()
        missing_pct = missing_count / dataframe.shape[0]
        if missing_pct > threshold:
            remov_columns.append(column)
    new_frame = new_frame.drop(columns = remov_columns)
    return new_frame


def sample_split(dataframe: pd.DataFrame, train_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset intwo a training set and a test set

    Parameters
    ----------
    dataframe : pd.DataFrame
        the input dataframe
    train_pc : float
        the percentage of training sample

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        the training set and the test set
    """
    train_obs_num = int(len(dataframe) * train_pct)
    train_frame = dataframe[:train_obs_num]
    test_frame = dataframe[train_obs_num:]
    return train_frame, test_frame


def create_ahead_lag(dataframe, target, max_ahead, max_lag):
    targets = list()
    new_frame = dataframe.copy(deep = True)
    for i in range(1, max_ahead + 1):
        new_frame[f"{target}_{i}H"] = new_frame[target].shift(-i)
        targets.append(f"{target}_{i}H")
    for i in range(1, max_lag + 1):
        for column in dataframe.columns:
            new_frame[f"{column}_{i}L"] = new_frame[column].shift(i)
    return new_frame, targets