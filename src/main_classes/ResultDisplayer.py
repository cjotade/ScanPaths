import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fig_prediction(y_real, y_pred, x_or_y, save_folder, prediction_title, x_lim=None, y_lim=None, is_save=False):
    fig = plt.figure(figsize=(20,6))
    plt.plot(y_real[x_or_y].values, 'C0*', label='true', alpha=0.7)
    plt.plot(y_pred[x_or_y].values, 'C1*', label='predicted', alpha=0.7)
    plt.legend(loc='best', prop={'size': 13})
    plt.title(prediction_title, fontsize=18)
    plt.ylabel(f"Target {x_or_y}", fontsize=18)
    plt.xlabel("Step", fontsize=18)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    if (x_lim is not None) or (y_lim is not None):
        prediction_title += " zoom"
    plt.grid()
    if is_save:
        filename = os.path.join(save_folder, prediction_title)
        fig.savefig(f"{filename}.png")
    plt.show()

def plot_prediction(y_real, y_pred, img_type, save_folder=None, x_lim=None, y_lim=None, is_save=False):
    title = "LSTM ScanPath Prediction for {} and {} images"
    # Plot LSTM Prediction for x
    prediction_title = title.format("x", img_type)
    fig_prediction(y_real, y_pred, "x", save_folder, prediction_title, x_lim, y_lim, is_save)
    
    # Plot LSTM Prediction for y
    prediction_title = title.format("y", img_type)
    fig_prediction(y_real, y_pred, "y", save_folder, prediction_title, x_lim, y_lim, is_save)
    
def fig_histogram(error, norm_fitted, hist_title, save_folder=None, is_save=False):
    # Distribution Error Histogram
    hist_fig, ax = plt.subplots()
    ax = sns.distplot(
            error, 
            fit=norm_stat,label=r"$(\mu,\sigma)=$ ({:.2f}, {:.2f})".format(norm_fitted[0], norm_fitted[1])
    )
    plt.title(hist_title)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.legend()
    if is_save:
        filename = os.path.join(save_folder, hist_title)
        hist_fig.savefig(f'{filename}.png')
    plt.show()
    
def plot_error_histogram(y_real, y_pred, img_type, save_folder=None, is_save=False):
    from .MetricsCalculator import fit_gaussian
    norm_fitted_x, norm_fitted_y, error_x, error_y = fit_gaussian(y_real, y_pred)
    # Distribution Error Histogram for x
    title = "Distribution Error {} Fitted for {} images"
    hist_title = title.format("x", img_type)
    fig_histogram(error_x, norm_fitted_x, hist_title, save_folder, is_save)
    # Distribution Error Histogram for y
    hist_title = hist_title.format("y", img_type)
    fig_histogram(error_y, norm_fitted_y, hist_title, save_folder, is_save)
    # Print mu and sigma for x and y
    print("Error Distribution Norm x Fitted parameters (mu,sigma):", norm_fitted_x)
    print("Error Distribution Norm y Fitted parameters (mu,sigma):", norm_fitted_y)
    return norm_fitted_x, norm_fitted_y

def plot_cross_correlation_histogram(cc_hist, x_or_y, img_type="", x_lim=None, y_lim=None):
    plt.bar(x=cc_hist[0].times.magnitude,
            height=cc_hist[0][:, 0].magnitude.squeeze(),
            width=cc_hist[0].sampling_period.magnitude,
            label=x_or_y,
            alpha=0.7)
    plt.xlabel(f'time ({cc_hist[0].times.units})')
    plt.ylabel('cross-correlation histogram')
    title = "Cross-Correlation for LSTM ScanPath x and y Prediction on {} images"
    # Plot LSTM Prediction for x
    correlation_title = title.format(img_type)
    plt.title(correlation_title)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    
def plot_metric_results(results, metric, pivot, tIMG=None, predIMG=None, is_save=False, save_folder=""):
    import re
    from scipy import stats
    from .DataRetriever import DataSearcher
    searcher = DataSearcher()
    # Filtering results and cast keys
    filter_results = searcher.group_results(results, pivot)
    # Concat results
    concated_results = searcher.concat_results(results, filter_results)
    # Histogram plots
    fig, ax = plt.subplots()
    # Cross correlogram plot
    if metric in ["peak_x", "peak_y"]:
        fig.set_figheight(5)
        fig.set_figwidth(13)
        concated_results.plot.bar(x="pivot", y="peak", legend=False, ax=ax)
        plt.grid(linestyle='--', axis="y")
        plt.title(f"{metric} | tIMG={tIMG} | predIMG={predIMG}")
    # MSE, scanmatch, dtw, rec, det
    #elif metric in ["MSE"]:
    elif metric in ["MSE", "scanmatch", "dtw", "rec", "det"]:
        for i, (pivot_key, group_list) in enumerate(sorted(filter_results.items())):
            sns.distplot(results.iloc[group_list], hist=False, kde=True, ax=ax, label=f"{pivot}={pivot_key}")
        plt.grid()
        plt.legend()
        plt.title(f"{metric} | tIMG={tIMG} | predIMG={predIMG}")
    # MultiMatch
    elif metric in ["multimatch"]:
        results_multi = np.array(concated_results["peak"].values.tolist())
        for i, similarity_type in enumerate(["shape", "length", "direction", "position", "duration"]):
            concated_results[similarity_type] = results_multi[:, i]
            fig.set_figheight(5)
            fig.set_figwidth(13)
            concated_results.plot.bar(x="pivot", y=similarity_type, legend=False, ax=ax)
            plt.grid(linestyle='--', axis="y")
            plt.ylim(0, 1.01)
            plt.title(f"{metric} {similarity_type} | tIMG={tIMG} | predIMG={predIMG}")
            if i != 4:
                fig, ax = plt.subplots()
    #Gaussian plots    
    else:
        for i, (pivot_key, group_list) in enumerate(sorted(filter_results.items())):
            res = results.iloc[group_list][0].apply(pd.Series)
            mu, sigma = res[0], res[1]
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            for j in range(x.shape[-1]):
                ax.plot(x[:, j], stats.norm.pdf(x[:, j], mu.values[j], sigma.values[j]), 
                        label=f"{pivot}={pivot_key}" if j == 0 else "", color=f"C{i}")
            ax.set_ylim(0,0.025)
            ax.set_xlim(-400,400)
        plt.grid()
        plt.legend()
        plt.title(f"{metric} | tIMG={tIMG} | predIMG={predIMG}")
    if is_save:
        savepath = os.path.join(save_folder, f"general/raw/{metric}/")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        fig.savefig(os.path.join(savepath, f'{metric}-tIMG_{tIMG}-predIMG_{predIMG}.png'))
    plt.show()
    
def plot_statistical_groups(grouped_results, metric=None, tIMG=None, predIMG=None, is_save=False, save_folder=""):
    from scipy import stats
    df = pd.DataFrame(grouped_results).T
    mean = df["mean"].apply(pd.Series)
    std = df["std"].apply(pd.Series)
    if metric.startswith("norm"):
        fig, ax = plt.subplots(1, 1, figsize=(15, 3))
        mu, sigma = mean[0], mean[1]
        #mu_std, sigma_std = std[0], std[1]
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        for i in range(x.shape[-1]):
            ax.plot(x[:, i], stats.norm.pdf(x[:, i], mu.values[i], sigma.values[i]), 
                        color=f"C{i}", label=f"N={mu.index[i]}")
        ax.set_ylim(0, 0.025)
        ax.set_xlim(-400, 400)
        ax.grid()
        ax.legend()
    else:
        fig, axs = plt.subplots(1, len(mean.columns), figsize=(15, 3))
        for col in mean.columns:
            ax = axs[col] if len(mean.columns) != 1 else axs
            mean[col].plot(kind="bar", yerr=std[col], legend=False, ax=ax, zorder=3)
            ax.grid(linestyle='--', axis="y", zorder=0)
            if metric == "multimatch":
                ax.set_title(["shape", "length", "direction", "position", "duration"][col])
                ax.set_ylim(0.4, 1.01)
    if tIMG or predIMG:
        fig.suptitle(f"grouped mean {metric} | tIMG={tIMG} | predIMG={predIMG}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if is_save:
        savepath = os.path.join(save_folder, f"general/grouped/{metric}/")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        fig.savefig(os.path.join(savepath, f'{metric}-tIMG_{tIMG}-predIMG_{predIMG}.png'))
    plt.show()

def plot_pred(y_real, y_pred, x_or_y):
    plt.figure(figsize=(20,6))
    plt.plot(y_real[x_or_y], 'C0*', label='true', alpha=0.7)
    plt.plot(y_pred[x_or_y], 'C1*', label='predicted', alpha=0.7)

def plot_mcdropout(y_real, y_pred, mcdropout_mean, mcdropout_std, x_or_y):
    plot_pred(y_real, y_pred, x_or_y)
    # Plot mean std
    plt.plot(mcdropout_mean[x_or_y], 'C2--', label='MC-Dropout samples mean', alpha=0.7)
    plt.fill_between(y_real[x_or_y].index, mcdropout_mean[x_or_y] - 2*mcdropout_std[x_or_y],
                     mcdropout_mean[x_or_y] + 2*mcdropout_std[x_or_y], color='C2', alpha=0.4)
    
def plot_config_prediction(x_or_y, subject_train, subject_test, img_type="natural", N=None, x_lim=None):
    plt.legend(loc='best', prop={'size': 13})
    plt.title(f'LSTM ScanPath Prediction for {x_or_y} | images {img_type} | {N} steps ahead | train subject {subject_train} | test subject {subject_test}', fontsize=18)
    plt.grid()
    plt.ylabel(f"Target {x_or_y}", fontsize=18)
    plt.xlabel("Step", fontsize=18)
    if x_lim:
        plt.xlim(x_lim[0], x_lim[1])