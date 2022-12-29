import os
import sys
sys.path.append("..")
import json
import numpy as np
import pandas as pd

from typing import Optional, List, Union
from utils.utils import merge_dicts

def load_npz(folder_path, img_type, get_mcdropout_results=False):
    """
    Load data and transform to dataframes.
    """
    print(f"===========> Loading npz data from {img_type} images")
    npz_file = np.load(os.path.join(folder_path, f"{img_type}.npz"))
    y_pred_load, y_real_load, idx = npz_file["y_pred"], npz_file["y_real"], npz_file["idx"]
    y_pred = pd.DataFrame(y_pred_load, index=idx, columns=["x", "y"])
    y_real = pd.DataFrame(y_real_load, index=idx, columns=["x", "y"])
    if get_mcdropout_results:
        mcdropout_mean, mcdropout_std = npz_file["mcdropout_mean"], npz_file["mcdropout_std"]
        mcdropout_mean = pd.DataFrame(mcdropout_mean, index=idx, columns=["x", "y"])
        mcdropout_std = pd.DataFrame(mcdropout_std, index=idx, columns=["x", "y"])
        return y_real, y_pred, mcdropout_mean, mcdropout_std
    return y_real, y_pred

class DataGetter:
    """
    Class for group all methods which gets data
    """
    def __init__(self, results_folder):
        self.results_folder = results_folder

    def get_metrics(self, results_folder: str, img_type: str, get_mcdropout_results: bool = False, select_metrics: list = []):
        """
        Load y_real and y_pred, then get metrics.
        
        Inputs
        -------
        RESULTS_FOLDER: str
            Folder where are the results
        img_type: str
            Type of the img to be loaded
        get_mcdropout_results: bool
            Whether retrieve or not MC Dropout results.

        Outputs
        --------
        norm_fitted_x:
        norm_fitted_y:
        MSE:
        peak_x:
        peak_y:
        multimatch: 
        y_real: (steps, pos)
        y_pred: (steps, pos)

        """
        from sklearn.metrics import mean_squared_error
        from utils.process_helpers import split_trials_from_scanpath
        from .MetricsCalculator import (
            fit_gaussian, 
            get_cross_correlation_histogram, 
            calculate_multimatch,
            DTW,
            calculate_scanmatch,
            calculate_rec,
            calculate_det
            )
        # Load results
        if get_mcdropout_results:
            y_real, y_pred, mc_mean, mc_std = load_npz(results_folder, img_type)
        else:
            y_real, y_pred = load_npz(results_folder, img_type)
        
        metrics = {}
        if ("norm_x" in select_metrics) or ("norm_y" in select_metrics) or (not select_metrics):
            # Metrics Norm
            norm_fitted_x, norm_fitted_y, _, _ = fit_gaussian(y_real, y_pred)
            metrics["norm_x"] = norm_fitted_x
            metrics["norm_y"] = norm_fitted_y
        if ("MSE" in select_metrics) or (not select_metrics):
            # Metrics MSE
            MSE = mean_squared_error(y_real.values, y_pred.values)
            metrics["MSE"] = MSE
        if ("peak_x" in select_metrics) or ("peak_y" in select_metrics) or (not select_metrics):
            _, peak_x =  get_cross_correlation_histogram(y_real["x"].values, y_pred["x"].values)
            _, peak_y = get_cross_correlation_histogram(y_real["y"].values, y_pred["y"].values)
            metrics["peak_x"] = peak_x
            metrics["peak_y"] = peak_y
        if ("multimatch" in select_metrics) or (not select_metrics):
            # Multi-match
            multimatch = calculate_multimatch(y_real, y_pred)
            metrics["multimatch"] = multimatch
        if ("dtw" in select_metrics) or (not select_metrics):
            # DTW
            dtw = DTW(y_real, y_pred)
            metrics["dtw"] = dtw
        if ("scanmatch" in select_metrics) or (not select_metrics):
            #scanmatch
            scanmatch = calculate_scanmatch(y_real, y_pred)
            metrics["scanmatch"] = scanmatch 
        if ("rec" in select_metrics) or (not select_metrics):
            #scanmatch
            rec = calculate_rec(y_real, y_pred, threshold=196.9)
            metrics["rec"] = rec 
        if ("det" in select_metrics) or (not select_metrics):
            #scanmatch
            det = calculate_det(y_real, y_pred, threshold=196.9)
            metrics["det"] = det 

        scan_predictions =  {
            "y_real": y_real,
            "y_pred": y_pred
        }
        if get_mcdropout_results:
            scan_predictions["mc_mean"] = mc_mean
            scan_predictions["mc_std"] = mc_std
        return metrics, scan_predictions
    
    def calculate_metrics_results(self, only_all_vs_all: bool = False, select_metrics: list = [], pre_all_results: dict = {}) -> dict:
        """
        Group all metrics for every results found in folder. Store all results in dictionary all_results.
        """
        all_results = {}
        for res_subfolder in os.listdir(self.results_folder):
            print(res_subfolder)
            all_results[res_subfolder] = {}
            all_results[res_subfolder]["all_vs_all"] = {}

            result_folder = os.path.join(self.results_folder, res_subfolder)
            if not only_all_vs_all:
                for result_file in filter(lambda x:x.endswith(".npz"), os.listdir(result_folder)):
                    img_type = result_file.split(".")[0]
                    if pre_all_results:
                        if pre_all_results.get(res_subfolder, {}).get(img_type):
                            all_results[res_subfolder][img_type] = {
                                metric_name: metric_value for metric_name, metric_value in pre_all_results[res_subfolder][img_type].items()
                            }
                        else:
                            img_type = result_file.split(".")[0]
                            metrics, _ = self.get_metrics(result_folder, img_type, select_metrics=select_metrics)
                            all_results[res_subfolder][img_type] = {
                                metric_name: metric_value for metric_name, metric_value in metrics.items()
                            }    
                    else:
                        metrics, _ = self.get_metrics(result_folder, img_type, select_metrics=select_metrics)
                        all_results[res_subfolder][img_type] = {
                            metric_name: metric_value for metric_name, metric_value in metrics.items()
                        }
            if only_all_vs_all:
                try:
                    results_all_vs_all_folder = os.path.join(result_folder, "all_vs_all")
                    if not os.path.exists(results_all_vs_all_folder):
                        continue
                    for other_subject_file in os.listdir(results_all_vs_all_folder):
                        all_results[res_subfolder]["all_vs_all"][other_subject_file] = {}
                        other_subject_folder = os.path.join(results_all_vs_all_folder, other_subject_file)
                        for result_other in filter(lambda x:x.endswith(".npz"), os.listdir(other_subject_folder)):
                            img_type_other = result_other.split(".")[0]
                            if pre_all_results:
                                if pre_all_results.get(res_subfolder, {}).get("all_vs_all", {}).get(other_subject_file, {}).get(img_type_other):
                                    all_results[res_subfolder]["all_vs_all"][other_subject_file][img_type_other] = {
                                        metric_name: metric_value for metric_name, metric_value  in pre_all_results[res_subfolder]["all_vs_all"][other_subject_file][img_type_other].items()
                                    }
                                else:
                                    metrics_other, _ = self.get_metrics(other_subject_folder, img_type_other, select_metrics=select_metrics)
                                    all_results[res_subfolder]["all_vs_all"][other_subject_file][img_type_other] = {
                                        metric_name: metric_value for metric_name, metric_value  in metrics_other.items()
                                    }
                            else:
                                metrics_other, _ = self.get_metrics(other_subject_folder, img_type_other, select_metrics=select_metrics)
                                all_results[res_subfolder]["all_vs_all"][other_subject_file][img_type_other] = {
                                    metric_name: metric_value for metric_name, metric_value  in metrics_other.items()
                                }
                except:
                    print("EXCEPTION")
                    pass
        # Note the transpose of results
        all_results = json.loads(pd.read_json(json.dumps(all_results)).T.to_json())
        return all_results
class DataSearcher:
    """
    Class for group all methods which query over the data
    """
    def search_in_results(self, key, subject=None, tIMG=None, N=None, Nt=None, SEQ=None, INuts=None, LR=None, EPOCHS=None, PAT=None, test_subject=None) -> bool:
        """
        Whether or not the params are in key.
        """
        # Search other args in key
        query = {
            "SUBJECT": str(subject) if subject is not None else subject, 
            "tIMG": str(tIMG) if tIMG is not None else tIMG, 
            "N": str(N) if N is not None else N, 
            "Nt": str(Nt) if Nt is not None else Nt, 
            "SEQ": str(SEQ) if SEQ is not None else SEQ, 
            "INuts": str(INuts) if INuts is not None else INuts, 
            "LR": str(LR) if LR is not None else LR, 
            "EPOCHS": str(EPOCHS) if EPOCHS is not None else EPOCHS, 
            "PAT": str(PAT) if PAT is not None else PAT,
            "testSUBJECT": str(test_subject) if test_subject is not None else test_subject
        }
        for q_key, q in query.items():
            if q is None:
                continue
            elif f"{q_key}_{q}-" not in key:
                #print(f"{q_key}_{q} not in {key}")
                return False
        return True

    def get_results_by_query(self, all_results, metric,
                                            subject=None, 
                                            tIMG=None, 
                                            N=None, 
                                            Nt=None, 
                                            SEQ=None, 
                                            INuts=None, 
                                            LR=None, 
                                            EPOCHS=None, 
                                            PAT=None, 
                                            predIMG=None, 
                                            is_all_vs_all=False,
                                            test_subject=None
                                            ):    
        """
        Get the results requested by query in a DataFrame.
        It can be used for test all vs all.
        """
        results = {}
        if not is_all_vs_all:
            if not predIMG:
                # iterate over all img types in all_results if not predIMG given
                for predIMG in all_results.keys():
                    if predIMG == "all_vs_all":
                        continue
                    for key, metric_dict in all_results[predIMG].items():
                        if self.search_in_results(key, subject=subject, N=N, tIMG=tIMG,
                                            Nt=Nt, 
                                            SEQ=SEQ, 
                                            INuts=INuts, 
                                            LR=LR, 
                                            EPOCHS=EPOCHS, 
                                            PAT=PAT):
                            results[f"{key}-predIMG_{predIMG}"] = metric_dict[metric]
            else:
                # iterate over especific predIMG given
                for key, metric_dict in all_results[predIMG].items():
                    if self.search_in_results(key, subject=subject, N=N, tIMG=tIMG,
                                        Nt=Nt, 
                                        SEQ=SEQ, 
                                        INuts=INuts, 
                                        LR=LR, 
                                        EPOCHS=EPOCHS, 
                                        PAT=PAT):
                        results[f"{key}-predIMG_{predIMG}"] = metric_dict[metric]
        # Get all_vs_all results
        else:
            # iterate over all train subjects
            for key_train, test_dict in all_results["all_vs_all"].items():
                # check key_train
                # Note: Set N=None
                if self.search_in_results(key_train, subject=subject, N=None, tIMG=tIMG,
                                                Nt=Nt, 
                                                SEQ=SEQ, 
                                                INuts=INuts, 
                                                LR=LR, 
                                                EPOCHS=EPOCHS, 
                                                PAT=PAT):
                    # iterate over all test subjects for every trained subject
                    for key_test, test_img_types_dict in test_dict.items():
                        # check for key_test
                        if self.search_in_results(key_test, subject=test_subject, N=N, tIMG=tIMG,
                                                    Nt=Nt, 
                                                    SEQ=SEQ, 
                                                    INuts=INuts, 
                                                    LR=LR, 
                                                    EPOCHS=EPOCHS, 
                                                    PAT=PAT):
                            if test_img_types_dict:
                                if not predIMG:
                                    # iterate over all img types in all_results if not predIMG given
                                    for predIMG in test_img_types_dict.keys():
                                        test_metric_dict = test_img_types_dict[predIMG]
                                        testSUBJECT = key_test.split("-")[0].split("_")[-1]
                                        Nh = key_test.split("-")[2].split("_")[-1]
                                        results[f"{key_train}-testSUBJECT_{testSUBJECT}-predIMG_{predIMG}-N_{Nh}"] = test_metric_dict[metric]
                                # use especific predIMG given
                                else:
                                    test_metric_dict = test_img_types_dict[predIMG]
                                    testSUBJECT = key_test.split("-")[0].split("_")[-1]
                                    Nh = key_test.split("-")[2].split("_")[-1]
                                    results[f"{key_train}-testSUBJECT_{testSUBJECT}-predIMG_{predIMG}-N_{Nh}"] = test_metric_dict[metric] 
        return pd.DataFrame(pd.read_json(json.dumps(results), typ='series')).sort_index(ascending=True).copy()
    
    def group_results(self, results, pivot):
        """
        Returns a filtered result with new keys.
        """
        filter_results = {}
        for i, idxs in enumerate(results.index):
            for param in idxs.split("-"):
                split = param.split("_")
                # fix img types keys with "_" like white_noise
                if len(split) == 2:
                    key, value = split
                else:
                    key, value = split[0], "_".join(split[1:])
                if key == pivot:
                    filter_results.setdefault(value, []).append(i)
        return {int(k):v for k, v in filter_results.copy().items()}

    def concat_results(self, results, filter_results):
        """
        Concat results for specific plotting.
        """
        all_res = []
        for i, (pivot_key, group_list) in enumerate(filter_results.items()):
            res = results.iloc[filter_results[pivot_key]].copy()
            res["pivot"] = pivot_key
            res.rename({0: "peak"}, axis=1, inplace=True)
            res.reset_index(inplace=True, drop=True)
            res["pivot"] = res["pivot"].astype(int)
            all_res.append(res)
        return pd.concat(all_res).sort_values(by="pivot").copy()

    def get_statistical_groups(self, df_results, steps_ahead=[1, 5, 11, 20]):
        """
        Create a dictionary for every step ahead which contains the mean and std of every 
        """
        import re
        df_results = df_results[0].apply(pd.Series)
        grouped_results = {}
        for step_ahead in steps_ahead:
            df_selected_step = df_results.iloc[df_results.index.str.contains(f'N_{step_ahead}-', re.IGNORECASE)]
            if len(df_results.columns) == 1:
                mean = [df_selected_step.mean().item()]
                std = [df_selected_step.std().item()]
            else:
                mean, std = [], []
                for i_col in df_selected_step.columns:
                    mean.append(df_selected_step[i_col].mean().item())
                    std.append(df_selected_step[i_col].std().item())
            grouped_results[step_ahead] = {
                "mean": mean,
                "std": std
            }
        return grouped_results
    