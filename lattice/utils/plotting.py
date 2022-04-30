import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_backtest(history: pd.DataFrame, path: os.PathLike) -> plt.Figure:
    # Converting to datetimes
    datetimes = pd.to_datetime(history.time*1000000)

    # Value series
    plt.figure(figsize = (9,5))
    plt.plot(datetimes, history.total_value)
    plt.xlabel('Time')
    plt.ylabel('USD')
    sns.despine()
    plt.savefig(path/'value_series.png')

    # Allocation series
    assets = history.columns[3:]
    allocations = history[assets]
    normed_allocations = allocations #.divide(allocations.sum(axis=1), axis=0)
    plt.figure(figsize = (9,5))
    for asset in assets:
        plt.plot(datetimes, normed_allocations[asset], label = asset)
    plt.xlabel('Time')
    plt.ylabel('Portfolio %')
    plt.legend()
    sns.despine()
    plt.savefig(path/'asset_allocations.png')
    plt.close()


def plot_return_distribution(returns: list, path: os.PathLike):
    plt.figure(figsize = (9,5))
    sns.kdeplot(np.array(returns)*100, fill=True, linewidth=0, bw_method=0.1)
    plt.xlabel('Return %')
    plt.ylabel('')
    sns.despine()
    plt.savefig(path/'return_kde.png')
    plt.close()

