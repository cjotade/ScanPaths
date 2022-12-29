import os
import sys
sys.path.append("..")

from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tqdm.notebook import tqdm
from utils.process_methods import process_features
from utils.utils import cast_to_int, create_empty_dict

from .ParamsCreator import DataParams

class DataProcessor:
    def __init__(self, subject, params, train_percent=0.5, val_percent=0.25):
        self.subject = subject
        self.params = params
        self.imgs_folder = params.data_params.IMGS_FOLDER
        self.train_percent = train_percent
        self.val_percent = val_percent

        self.scaler = None
        
        # Prediction columns
        self.pred_columns = ["x_izq", "y_izq"]
        # Functions for process data
        self.process_functions = params.model_params.PROCESS_FUNCTIONS 

        self.fit_dispatcher = {
            "fit_etr": self.fit_etr,
            "fit_pca": self.fit_pca
        }

    def run(self, img_type, return_split=True, process_functions=[], kwargs={}):
        # Pre-processing
        print("Pre-processing data...")
        pre_process_kwargs = kwargs.get("pre_process_kwargs", {})
        pre_data_blocks = self.pre_process(img_type, **pre_process_kwargs)
        # processing
        print("Processing features...")
        pre_features_blocks = self.process_features(pre_data_blocks, process_functions=process_functions)
        # Post-processing
        print("Post-processing...")
        post_process_kwargs = kwargs.get("post_process_kwargs", {})
        data_blocks, targets_blocks, features_blocks  = self.create_targets_adjust_indexes(pre_data_blocks, features_blocks=pre_features_blocks)
        data_blocks_with_features, self.feature_selectors = self.post_process(data_blocks, targets_blocks, features_blocks=features_blocks, **post_process_kwargs)
        # Structure data and return
        timeseries_kwargs = kwargs.get("timeseries_kwargs", {})
        return self.structure_data(data_blocks_with_features, targets_blocks, return_split=return_split, **timeseries_kwargs)

    def pre_process(self, img_type, select_only_saccades=False, saccades_bound=0):
        # Load data
        data = self.load_data()
        # Select data blocks
        data_blocks, img_names = self.select_data_blocks(data, img_type, select_only_saccades=select_only_saccades, saccades_bound=saccades_bound)
        self.img_names = img_names
        return data_blocks

    def process_features(self, data_blocks, process_functions=[], save_blocks_in_list=True):
        """
        Process features adding to every block the new features.
        Returns an empty list if save_blocks_in_list is False.
        """
        features_blocks = []
        img_path_names = self.get_img_names(return_split=False)
        if not process_functions:
            process_functions = self.process_functions
        for i, data_block in enumerate(data_blocks):
            processed_features_block = process_features(data_block, img_path_names[i], process_functions)
            if save_blocks_in_list and len(processed_features_block) != 0:
                features_blocks.append(processed_features_block)
        return features_blocks

    def create_targets_adjust_indexes(self, data_blocks, features_blocks=None, target_steps_ahead=None):
        """
        Create real and targets blocks for prediction
        """
        if not target_steps_ahead:
            target_steps_ahead = self.params.labels_params.TARGET_STEPS_AHEAD
        targets_blocks = []
        real_blocks = []
        for data_block in data_blocks:
            real_values = data_block[:-target_steps_ahead]
            target_values = data_block[target_steps_ahead:]
            targets_blocks.append(target_values)
            real_blocks.append(real_values)
        fts_blocks = []
        if features_blocks:
            for feature_block in features_blocks:    
                fts_values = feature_block[:-target_steps_ahead]
                fts_blocks.append(fts_values)
        return real_blocks, targets_blocks, fts_blocks

    def post_process(self, data_blocks, targets_blocks, features_blocks=[], **kwargs):
        """
        Post-process features
        """
        # Fit feature selector (eg. ExtraTreesRegressor or PCA)
        if features_blocks:
            feature_selectors = kwargs.get("feature_selectors", [])
            # Try to load features selectors
            if (not feature_selectors) and kwargs.get("load_feature_selectors", False):
                feature_selectors_filepath = kwargs.get("feature_selectors_filepath")
                if feature_selectors_filepath:
                    try:
                        feature_selectors = joblib.load(feature_selectors_filepath)
                        print(f"feature_selectors {feature_selectors_filepath} successfully loaded!")
                    except:
                        print(f"Couldn't load {feature_selectors_filepath}")
            # Use passed features selectors
            if feature_selectors:
                ft_selectors = feature_selectors 
                fts_sel = True
            # Fit feature selectors
            else: 
                ft_selectors = kwargs.get("feature_selectors_map", [])
                fts_sel = False
            for feature_selector_or_kwargs in ft_selectors:
                # Use feature selector directly
                if fts_sel:
                    print("Using feature selector directly")
                    feature_selector = feature_selector_or_kwargs
                # Fit feature selector by using function dispatcher and kwargs
                else:
                    print("Fitting feature selector...")
                    fs_kwargs = feature_selector_or_kwargs.copy()
                    fit_fn_name = fs_kwargs.pop("fit_fn_name")
                    fit_fn = self.fit_dispatcher[fit_fn_name]
                    # Fit feature selector
                    feature_selector = fit_fn(features_blocks, targets_blocks, **fs_kwargs)
                    feature_selectors.append(feature_selector) # case when feature_selectors is empty
                # Transform every feature_block with feature_selector.transform
                features_blocks_selected = []
                for features_block in features_blocks:
                    fts_block_selected = feature_selector.transform(features_block)
                    features_blocks_selected.append(fts_block_selected)
                features_blocks = features_blocks_selected
        else:
            feature_selectors = []
        if kwargs.get("save_feature_selectors", False):
            feature_selectors_filepath = kwargs["feature_selectors_filepath"]
            if not os.path.exists(os.path.dirname(feature_selectors_filepath)):
                os.makedirs(os.path.dirname(feature_selectors_filepath))
            if not os.path.isfile(feature_selectors_filepath):
                joblib.dump(feature_selectors, feature_selectors_filepath)
            else:
                print(f"feature_selectors {feature_selectors_filepath} already exists")
        # Stack data_blocks and features_blocks
        data_blocks_with_features = []
        if features_blocks:
            for data_block, feature_block in zip(data_blocks, features_blocks):
                data_block_with_features = np.hstack([data_block, feature_block])
                data_blocks_with_features.append(data_block_with_features)
        else:
            data_blocks_with_features = data_blocks
        return data_blocks_with_features, feature_selectors

    def structure_data(self, data_blocks_with_features, targets_blocks, return_split=True, **kwargs):
        # Fit scaler
        self.scaler = kwargs.get("scaler", None)
        if not self.scaler: #and (len(data_blocks_with_features[0].shape) <= 2):
            kwargs["scaler"] = self.fit_data_and_set_scaler(data_blocks_with_features)
        # Create timeseries data
        X_gens, y_gens = self.create_timeseries(data_blocks_with_features, targets_blocks, **kwargs)
        # Return in train, test and val data
        if return_split:
            return self.split_data(X_gens, y_gens)
        else:
            return X_gens, y_gens
    
    def get_train_data_for_fit(self, blocks):
        X_fitter = []
        for i in range(len(blocks)):
            # Note we are using only train for fit (set to train_percent of data)
            if i < int(len(blocks)*self.train_percent): 
                X_fitter.append(blocks[i])
            else:
                break
        X_fitter = np.concatenate(X_fitter)
        return X_fitter
    
    def fit_etr(self, data_blocks_with_features, targets_blocks, **kwargs):
        """
        Method for fit ExtraTreesRegressor to train blocks and return feature selector
        """
        threshold = kwargs.pop("threshold", None)
        # Create etr
        etr_reg = ExtraTreesRegressor(random_state=0, **kwargs)
        # Get train features and targets
        features_train = self.get_train_data_for_fit(data_blocks_with_features)
        targets_train = self.get_train_data_for_fit(targets_blocks)
        # Fit etr
        etr_reg.fit(features_train, targets_train)
        # return feature selector
        feature_selector = SelectFromModel(etr_reg, threshold=threshold, prefit=True)
        return feature_selector

    def fit_pca(self, data_blocks_with_features, targets_blocks=None, pca_percent=0.95, **kwargs):
        pca = PCA(pca_percent, random_state=0) #0.85
        features_train = self.get_train_data_for_fit(data_blocks_with_features)
        pca.fit(features_train)
        return pca

    def fit_scaler(self, data_blocks_with_features, targets_blocks=None):
        """
        Fit standard scaler to train blocks and return scaler
        """
        X_fitter = self.get_train_data_for_fit(data_blocks_with_features)
        # Scale data
        scaler = StandardScaler()
        # return scaler
        scaler = scaler.fit(X_fitter)
        return scaler
    
    def fit_data_and_set_scaler(self, data_blocks_with_features):
        self.scaler = self.fit_scaler(data_blocks_with_features)
        return self.scaler

    def create_timeseries(self, data_blocks_with_features, targets_blocks, seq_length=None, batch_size=None, scaler=None, transient_response=None, target_steps_ahead=None, **kwargs):
        if not seq_length:
            seq_length = self.params.timeseries_params.SEQ_LENGTH
        X_gens, y_gens = [], []
        for data_block, target_block in zip(data_blocks_with_features, targets_blocks):
            # Separate data in features and targets
            X = data_block[1:] # Missing first for TimeseriesGenerator's bug
            y = target_block[:-1] # Missing last TimeseriesGenerator's bug
            if transient_response:
                X = X[transient_response:]
                y = y[transient_response:]
            if target_steps_ahead:
                X = X[:-target_steps_ahead]
                y = y[target_steps_ahead:]
            # Scale data
            if scaler:
                X = scaler.transform(X)
            # Time Sequences
            tsg_batch_size = X.shape[0] - seq_length if not batch_size else batch_size
            data_gen = TimeseriesGenerator(X, y, length=seq_length, batch_size=tsg_batch_size)
            # Unpack Time series to numpy arrays
            for data in data_gen:
                X_gen, y_gen = data
                # Storing data
                if X_gen.shape[:-1] == (tsg_batch_size, seq_length) or tsg_batch_size is None:
                    X_gens.append(X_gen)
                    y_gens.append(y_gen)
        return X_gens, y_gens

    def _split_data(self, data):
        # Split Train and Test
        # Note we are fixing train size and val size
        train_size = int(len(data)*self.train_percent) 
        val_size = int(len(data)*self.val_percent)
        data_train = data[:train_size]
        data_val = data[train_size:train_size+val_size]
        data_test = data[train_size+val_size:]
        return data_train, data_val, data_test

    def split_data(self, X_gens, y_gens):
        # Split Train and Test
        X_train, X_val, X_test = self._split_data(X_gens)
        y_train, y_val, y_test = self._split_data(y_gens)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def fill_img_names(self, data_blocks):
        self.img_names = [block.iloc[0]["img_id"] for block in data_blocks]

    def get_img_names(self, return_split=True, return_with_folder=True):
        if return_with_folder:
            imgs_path = list(map(lambda img: os.path.join(self.imgs_folder, img), self.img_names))
        else:
            imgs_path = self.img_names
        if len(np.unique(imgs_path)) == 1:
            imgs_path = [f"{imgs_path[i].split('.png')[0]}_{i}.png" for i in range(len(imgs_path))]
        if return_split:
            train_size = int(len(imgs_path)*self.train_percent) 
            val_size = int(len(imgs_path)*self.val_percent)
            train_imgs = imgs_path[:train_size]
            val_imgs = imgs_path[train_size:train_size+val_size]
            test_imgs = imgs_path[train_size+val_size:]
            return train_imgs, val_imgs, test_imgs
        else:
            return imgs_path

    def get_scaler(self):
        return self.scaler

    def get_process_functions(self):
        return self.process_functions

    def get_feature_selectors(self):
        if self.feature_selectors:
            return self.feature_selectors
        else:
            return None

    def _retrieve_img_indexes(self, data_blocks):
        img_indexes = {"white": [], "black": [], "grey": [], "natural": [], "inverted": [], "pink_noise": [], "white_noise": [], "all": []}
        for i, data_block in enumerate(data_blocks):
            if "blanca" in data_block.iloc[0]["img_name"]:
                img_indexes["white"].append(i)
            if "negra" in data_block.iloc[0]["img_name"]:
                img_indexes["black"].append(i)
            if "_gr" in data_block.iloc[0]["img_name"]:
                img_indexes["grey"].append(i)
            if "_pi" in data_block.iloc[0]["img_name"]:
                img_indexes["pink_noise"].append(i)
            if "_up" in data_block.iloc[0]["img_name"]:
                img_indexes["inverted"].append(i)
            if "_li" in data_block.iloc[0]["img_name"]:
                img_indexes["natural"].append(i)
            if "_pa" in data_block.iloc[0]["img_name"]:
                img_indexes["white_noise"].append(i)
        return img_indexes

    def _image_matcher(self, imgs_folder):
        # Read image list
        img_list = pd.read_csv(os.path.join(imgs_folder, "imagelst.dat"), header=None, names=['names'], sep='\t', engine='python')
        img_list = img_list['names'].str.split(' ', expand=True)
        img_list.columns = ['jpg', 'id']
        return img_list

    def _match_image(self, img_name, match_list):
        return match_list[match_list['jpg'].str.contains(img_name)]['id'].values[0]

    def load_data(self):
        """
        Load data using subject and params
        """
        data_params = self.params.data_params
        data_folder = os.path.join(data_params.DATA_FOLDER, self.subject)
        width_orig = data_params.WIDTH_ORIG
        height_orig = data_params.HEIGHT_ORIG
        width = data_params.WIDTH
        height = data_params.HEIGHT
        # Columns names
        cols = ['ms', 'x_izq', 'y_izq', 'p_izq', 'x_der', 'y_der', 'p_der', 'vx_izq', 'vy_izq', 'vx_der', 'vy_der', 'a1', 'a2', 'a3', 'a4']
        selected_cols = ['ms', 'x_izq', 'y_izq', 'x_der', 'y_der']
        # Read data
        df_data = pd.read_csv(os.path.join(data_folder, f"{data_folder.split('/')[-1]}.asc"), header=None, names=cols, sep='\t', engine='python')
        df_data = df_data[selected_cols]
        # Images
        img_names = df_data[df_data['x_izq'].str.contains(".jpg",na=False)]
        img_names = img_names['x_izq'].str.split(' ', expand=True)[2].reset_index()
        # Indexation
        timer_idx = df_data[df_data['x_izq'].str.contains("Timer_TTL_Imagen",na=False)] 
        timer_idx = timer_idx['x_izq'].reset_index()
        fix_idx = df_data[df_data['ms'].str.contains('SSACC R', na=False)]     
        fix_idx = fix_idx['ms'].reset_index()
        initial_idx = [fix_idx[fix_idx['index'] > thresh]['index'].values[0] for thresh in timer_idx['index'].values]
        initial_idx = pd.DataFrame(initial_idx, columns=['index'])
        end_idx = df_data[df_data['ms'].str.contains("END",na=False)] 
        end_idx = end_idx['ms'].reset_index()
        img_names['initial_idx'] = initial_idx['index'] + 1
        img_names['end_idx'] = end_idx['index'] - 1
        img_names.rename({2:'img_name'}, axis=1, inplace=True)
        # Clean data
        df_data["ms"] = df_data["ms"].apply(lambda x: cast_to_int(x))
        df_data.dropna(inplace=True)
        # Parse data from string to float
        for col in selected_cols:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce') 
        df_data.dropna(inplace=True)
        # Normalizing data
        df_data['x_izq'] -= int((width_orig - width)/2)
        df_data['y_izq'] -= int((height_orig - height)/2)
        df_data['x_der'] -= int((width_orig - width)/2)
        df_data['y_der'] -= int((height_orig - height)/2)
        # Matching images names to id
        match_list = self._image_matcher(self.imgs_folder)
        data_blocks = []
        for i, row in img_names[['initial_idx', 'end_idx', 'img_name']].iterrows():
            init = row['initial_idx']
            end = row['end_idx']
            img_name = row['img_name']
            img_id = self._match_image(img_name, match_list)
            data = df_data[((df_data.index >= init) & (df_data.index <= end))].copy()
            data['img_name'] = img_name
            data['img_id'] = f"{img_id}.png"
            data.set_index('ms', inplace=True)
            if not data.empty:
                data_blocks.append(data)
        return data_blocks

    def select_data_blocks(self, data, img_type, select_only_saccades=False, saccades_bound=0):
        """
        Select data_blocks for img_type given and store img_names.
        """
        from main_classes.MetricsCalculator import detect_sac_fix_from_scanpath
        if (img_type is None) or (img_type == "all"):
            return data.copy()
        img_type_indexes = self._retrieve_img_indexes(data)[img_type]
        data_list, img_names = [], []
        for idx in img_type_indexes:
            block = data[idx].copy()
            if len(block) < 100:
                continue
            # Fill img_names
            img_names.append(block.iloc[0]["img_id"])
            # Store blocks using pred_columns
            if select_only_saccades:
                sacc_info, _ = detect_sac_fix_from_scanpath(block[["x_izq", "y_izq"]])
                t_i, t_f = sacc_info["t_i"].astype(int), sacc_info["t_f"].astype(int)
                block_arr = []
                for idx_i, idx_f in zip(t_i, t_f):
                    block_arr.append(block[self.pred_columns].iloc[idx_i-saccades_bound:idx_f+saccades_bound])
                block_arr = pd.concat(block_arr).values
            else:
                block_arr = block[self.pred_columns].values
            data_list.append(block_arr)
        return data_list.copy(), img_names
