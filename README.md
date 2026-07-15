# Forecasting Inflation in a Big Data Setting Using Neural Networks
This project evaluates whether neural networks can improve US CPI inflation forecasts in a data-rich macroeconomic setting.

The analysis compares feed-forward neural networks (FNNs) and long short-term memory networks (LSTMs) with several econometric and machine-learning benchmarks:

- Autoregressive model (AR)
- Autoregressive distributed lag model (ARDL)
- LASSO
- Random forest
- Gradient boosting machine (GBM)

Forecasts are produced for horizons of 1, 3, 6, 12, and 24 months.

## Project Overview
The project uses monthly macroeconomic indicators from the FRED-MD database. The forecasting target is the log difference of the US all-items Consumer Price Index.

The main preprocessing steps are:

1. Transform each macroeconomic series to stationarity
2. Remove series with more than 10% missing observations
3. Impute the remaining missing values using Bayesian ridge regression
4. Standardize the macroeconomic predictors
5. Extract the first 10 principal components using principal component analysis (PCA)
6. Apply min-max normalization to the predictors and inflation series
7. Fit all preprocessing transformations using only the training sample

The main training & evaluation steps are:

1. A separate model is trained for each forecasting horizon
2. Hyperparameters are selected using five-fold expanding-window time-series cross-validation
3. To reduce the sensitivity of neural networks to initialization, the FNN and LSTM forecasts are averaged across 20 independently initialized models
4. Forecasting performance is evaluated using out-of-sample root mean squared error (RMSE)

## Main Results

| Model | 1 month | 3 months | 6 months | 12 months | 24 months |
|---|---:|---:|---:|---:|---:|
| FNN | 93.09 | 97.95 | 98.62 | 106.50 | 101.03 |
| LSTM | 115.31 | 94.62 | 89.32 | 85.10 | 97.08 |
| ARDL | 98.59 | 109.61 | 105.66 | 114.73 | 107.74 |
| LASSO | 142.34 | 126.44 | 120.05 | 116.32 | 106.43 |
| Random Forest | 136.40 | 119.75 | 118.41 | 111.70 | 103.91 |
| GBM | 130.46 | 116.01 | 114.32 | 105.94 | 97.71 |

The main findings are:

- The FNN outperforms the AR benchmark at forecasting horizons of 1, 3, and 6 months.
- The LSTM outperforms AR at horizons of 3, 6, 12, and 24 months.
- The largest improvement is achieved by the LSTM at the 12-month horizon.
- The univariate AR model remains difficult to outperform using several conventional multivariate machine-learning methods.

## Installation
1. cd project-root-dir
2. conda env create -f environment.yml
3. conda activate d200_project
4. pip install -e .

## Run prediction & visualization
1. Open the file model_train_eval.ipynb in scripts folder
2. Select the virtual environment d200_project
2. Execute all code blocks