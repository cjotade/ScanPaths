import os
import re
import cv2
import numpy as np

from .process_helpers import insert_multiple_features_to_data
from .process_helpers import (
    img_to_power_spectrum,
    power_spectrum_to_lin, 
    power_spectrum_to_powerlaw,
)
from .process_helpers import foveat_img

from .utils import split_chunks

def process_features(data_block, img_path_i="", process_functions=[]):
    """
    Add features to data_block
    process_functions: List[Dict]
    """
    data_block = np.array(data_block)
    if (not img_path_i) or (not process_functions):
        return []
    processed_features_block = []
    for process_dict in process_functions:
        # Get kwargs
        kwargs = process_dict.get("kwargs", {})
        feature_selectors = process_dict.get("feature_selectors", [])
        # Process function
        process_fn = process_dict["process_fn"]
        # Check if load_features
        if kwargs.get("load_features", False):
            save_folder = kwargs["save_folder"]
            img_name = f'{img_path_i.split("/")[-1].split(".")[0]}.npz'
            features_filepath = os.path.join(save_folder, img_name)
            features = np.load(features_filepath)["features"]
        else:
            # Load if save_folder (for not process if calculated before)
            save_folder = kwargs.get("save_folder", None)
            if save_folder:
                img_name = f'{img_path_i.split("/")[-1].split(".")[0]}.npz'
                features_filepath = os.path.join(save_folder, img_name)
                if os.path.isfile(features_filepath):
                    print("Found features in:", features_filepath)
                    features = np.load(features_filepath)["features"]
                else:
                    print(f"Calculating features for image: {img_path_i}")
                    features = process_fn(data_block, img_path_i, **kwargs)     
            # Process feature
            else:
                print(f"Working on img: {img_path_i}")
                features = process_fn(data_block, img_path_i, **kwargs) 
        for feature_selector in feature_selectors:
            features = feature_selector.transform(features)
        processed_features_block.append(features)
    processed_features_block = np.hstack(processed_features_block)
    return processed_features_block

def process_foveatedImg(data, img_path_i, IMG_SIZE=None, pretrained_model_fn=None, pretrained_model_kwargs={}, save_folder="", **kwargs):
    """
    Process image by adding a foveated area given a scanpath (data)
    data: scanpath (x, y) positions
    img_path_i: str
    save_folder: ../../data/foveated_images/subject_name/
    pretrained_model_fn: function which load the pretrained_model
    """
    if (not os.path.exists(save_folder)) and save_folder:
        os.makedirs(save_folder)
    data = np.array(data)
    img = cv2.imread(img_path_i)
    # Any negative value will be set to 1
    data[:, 0][data[:, 0] < 0] = 1
    data[:, 1][data[:, 1] < 0] = 1
    # Fixations outside the screen resolution will be set to the screen resolution
    data[:, 0][data[:, 0] > img.shape[1]] = img.shape[1] - 1
    data[:, 1][data[:, 1] > img.shape[0]] = img.shape[0] - 1
    # Load SALICON pretrained model
    pretrained_model = pretrained_model_fn(**pretrained_model_kwargs)
    # Allocate space for saliency foveated images
    features = np.zeros(
        (len(data), np.prod(pretrained_model.output.shape[1:])) 
    )
    for k, data_k in enumerate(data):
        j, i = data_k.astype(int)
        fov_img = foveat_img(img, fixs=[(j, i)]) #(x,y)
        if IMG_SIZE:
            fov_img = cv2.resize(fov_img, IMG_SIZE)
        features[k] = pretrained_model(np.expand_dims(fov_img, 0))
    # Save features in folder
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_filename = f"{img_path_i.split('/')[-1].split('.')[0]}.npz"
        save_filepath = os.path.join(save_folder, save_filename)
        np.savez_compressed(save_filepath, features=features)
    return features

def process_LR(data, img_path_i, add_pos_coef=True, **kwargs):
    """
    Process and fit linear regression on log Power Spectrum from image
    """
    data = np.array(data)
    # Calculate power spectrum
    img = cv2.imread(img_path_i, 0)
    magnitude_spectrum = img_to_power_spectrum(img)
    # Select two spectrum directions
    spectrum1 = magnitude_spectrum[magnitude_spectrum.shape[0]//2:,  magnitude_spectrum.shape[1]//2]
    spectrum2 = magnitude_spectrum[magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2:]
    # Get lin reg from power spectrum
    slope1, pos_coef1 = power_spectrum_to_lin(spectrum1)
    slope2, pos_coef2 = power_spectrum_to_lin(spectrum2)
    # Add features to prediction
    if add_pos_coef:
        features = np.zeros((len(data), 4))
        features[:] = [slope1, pos_coef1, slope2, pos_coef2]
    else:
        features = np.zeros((len(data), 2))
        features[:] = [slope1, slope2]
    return features

def process_S(data, img_path_i, **kwargs):
    """
    Same as process_LR but use only slopes.
    """
    return process_LR(data, img_path_i, add_pos_coef=False)

def process_PL(data, img_path_i, **kwargs):
    """
    Process and fit Power Law on Power Spectrum from image
    """
    data = np.array(data)
    # Calculate power spectrum
    img = cv2.imread(img_path_i, 0)
    magnitude_spectrum = img_to_power_spectrum(img)
    # Select two spectrum directions
    spectrum1 = magnitude_spectrum[magnitude_spectrum.shape[0]//2:,  magnitude_spectrum.shape[1]//2]
    spectrum2 = magnitude_spectrum[magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2:]
    # Get powerlaw coefs from power spectrum
    alpha1, beta1 = power_spectrum_to_powerlaw(spectrum1)
    alpha2, beta2 = power_spectrum_to_powerlaw(spectrum2)
    # Add features to prediction
    features = np.zeros((len(data), 4))
    features[:] = [alpha1, beta1, alpha2, beta2]
    return features

def process_local_LR(data, img_path_i, add_pos_coef=True, patch_size=40, **kwargs):
    """
    Process and fit linear regression on log Power Spectrum from a patch of image
    """
    data = np.array(data)
    img = cv2.imread(img_path_i, 0)
    # Any negative value will be set to 1
    data[:, 0][data[:, 0] < 0] = 1
    data[:, 1][data[:, 1] < 0] = 1
    # Fixations outside the screen resolution will be set to the screen resolution
    #! Revisar ya que me habia equivocado en restar estas condiciones de borde
    data[:, 0][data[:, 0] > img.shape[1]] = img.shape[1] - 1
    data[:, 1][data[:, 1] > img.shape[0]] = img.shape[0] - 1
    upper_bound = int(abs(np.array(data.max(axis=0)[::-1])-img.shape).max()) #[::-1] for reversed
    lower_bound = abs(int(np.array(data.min(axis=0)[::-1]).min()))
    len_pad = max(upper_bound, lower_bound) + patch_size//2
    img_pad = cv2.copyMakeBorder(img, len_pad, len_pad, len_pad, len_pad, borderType=cv2.BORDER_CONSTANT, value=127)
    features = []
    for data_i in data:
        j, i = data_i.astype(int) + len_pad
        img_patch = img_pad[i-patch_size//2:i+patch_size//2, j-patch_size//2:j+patch_size//2]
        if img_patch.shape != (patch_size, patch_size):
            #! JUST FOR CHECK ERRORS
            print(img_patch.shape)
            print(i, j)
            print(img_pad.shape)
            print(img.shape)
            print()
        magnitude_spectrum = img_to_power_spectrum(img_patch)  
        spectrum1 = magnitude_spectrum[magnitude_spectrum.shape[0]//2:,  magnitude_spectrum.shape[1]//2]
        spectrum2 = magnitude_spectrum[magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2:]
        # Get lin reg from power spectrum
        slope1, pos_coef1 = power_spectrum_to_lin(spectrum1)
        slope2, pos_coef2 = power_spectrum_to_lin(spectrum2)
        if add_pos_coef:
            features.append([slope1, pos_coef1, slope2, pos_coef2])
        else:
            features.append([slope1, slope2])
    #return np.hstack([data, features])
    return features

def process_local_S(data, img_path_i, patch_size=40, **kwargs):
    return process_local_LR(data, img_path_i, add_pos_coef=False, patch_size=patch_size)

def process_LC(data, img_path_i, **kwargs):
    """
    Process and select Luminance Contrast given image and data
    """
    data = np.array(data)
    splitted_path = img_path_i.split("/")
    assert splitted_path[-1].endswith(".png")
    splitted_path[-2] = "LC_images"
    img = cv2.imread(os.path.join(*splitted_path), 0)
    # Any negative value will be set to 1
    data[:, 0][data[:, 0] < 0] = 1
    data[:, 1][data[:, 1] < 0] = 1
    # Fixations outside the screen resolution will be set to the screen resolution
    data[:, 0][data[:, 0] > img.shape[1]] = img.shape[1] - 1
    data[:, 1][data[:, 1] > img.shape[0]] = img.shape[0] - 1
    features = []
    for data_i in data:
        j, i = data_i.astype(int) - 1
        LC = img[i, j]
        features.append([LC])        
    #return np.hstack([data, features])
    return features