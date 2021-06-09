import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score

def plot_residuals(y, yhat):
    sns.scatterplot(x=y, y=y - yhat)
    plt.title("Residuals")
    plt.ylabel("y - yhat")
    plt.show()
    
def regression_errors(df, y, yhat):
    SSE = ((y-yhat) ** 2).sum()
    ESS = ((yhat - y.mean()) ** 2).sum()
    TSS = ((y - y.mean()) **2).sum()
    n = df.shape[0]
    MSE = mean_squared_error(y, yhat)
    RMSE =  sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(df, y):
    baseline_residuals = y - y.mean()
    sse_baseline = (baseline_residuals ** 2).sum()
    n = df.shape[0]
    mse_baseline = sse_baseline / n
    rmse_baseline = sqrt(mse_baseline)
    return sse_baseline, mse_baseline, rmse_baseline

def better_than_baseline(df,y, yhat):
    SSE, ESS, TSS, MSE, RMSE = regression_errors(df, y, yhat)
    sse_baseline, mse_baseline, rmse_baseline = baseline_mean_errors(df, y)                     
    if (SSE < sse_baseline):
        print('The model performs better than baseline')
        return True
    else:
        print('The model does not perform better than baseline')
        return False