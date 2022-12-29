import os
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import cv2

#######################################
### Helpers data processing methods ###
#######################################

def create_real_dict(y_real, n_steps_ahead: Optional[List] = [1]):
    """
    Auxiliar method which creates a dict with many same y_real.
    """
    if not isinstance(n_steps_ahead, list):
        n_steps_ahead = [n_steps_ahead]
    y_real_dict = {
        i_step: [
            pd.DataFrame(y_real_i, 
                columns=["x", "y"], 
                index=np.arange(1, len(y_real_i)+1)) for y_real_i in y_real
        ] for i_step in n_steps_ahead
    }
    return y_real_dict

def adjust_indexes(y_real_dict, y_pred_dict, n_steps_ahead, mcdropout_dicts=None):
    """
    Calculates the intersection between y_real and y_pred and returns them
    """
    from .utils import create_empty_dict
    if not isinstance(n_steps_ahead, list):
        n_steps_ahead = [n_steps_ahead]
    y_real_dict_new = create_empty_dict(y_real_dict.keys())
    y_pred_dict_new = create_empty_dict(y_pred_dict.keys())
    if mcdropout_dicts:
        mcdropout_mean_dict, mcdropout_std_dict = mcdropout_dicts
        mcdropout_mean_dict_new = create_empty_dict(mcdropout_mean_dict.keys())
        mcdropout_std_dict_new = create_empty_dict(mcdropout_std_dict.keys())
    for n_step in n_steps_ahead:
        for i_img, (y_r, y_p) in enumerate(zip(y_real_dict[n_step], y_pred_dict[n_step])):
            # For calculate metrics in intersection
            idx_intersection = y_r.index.intersection(y_p.index)
            y_real_dict_new[n_step].append(y_r.loc[idx_intersection])
            y_pred_dict_new[n_step].append(y_p.loc[idx_intersection])
            if mcdropout_dicts:
                mcdropout_mean_dict_new[n_step].append(mcdropout_mean_dict[n_step][i_img].loc[idx_intersection])
                mcdropout_std_dict_new[n_step].append(mcdropout_std_dict[n_step][i_img].loc[idx_intersection])
    if mcdropout_dicts:
        return y_real_dict_new, y_pred_dict_new, mcdropout_mean_dict_new, mcdropout_std_dict_new
    else:
        return y_real_dict_new, y_pred_dict_new

def split_trials_from_scanpath(y):    
    idxs = np.where(y.index[0] == y.index)[0]
    y_arr = []
    for i in range(len(idxs)-1):
        df_trial = y.iloc[idxs[i]:idxs[i+1]] 
        y_arr.append(df_trial)
    y_arr.append(y.iloc[idxs[i+1]:])
    return y_arr

################################################
### Features data processing helper methods  ###
###         (for add features to data)       ###
################################################

def power_spectrum_to_lin(spectrum):
    """
    Get linear regression from power spectrums
    """
    def lin_reg(x, a, b):
        return a * x + b
    def fit_linreg(spectrum):
        from scipy.optimize import curve_fit
        X = np.log(np.arange(1, len(spectrum)+1))
        popt, pcov = curve_fit(lin_reg, X, spectrum)
        return popt[0], popt[1]
    slope, pos_coef = fit_linreg(spectrum)
    return slope, pos_coef

def power_spectrum_to_powerlaw(spectrum):
    """
    Get linear regression from power spectrums
    """
    def powerlaw(x, alpha, beta):
        return alpha * np.power(x, beta)
    def fit_powerlaw(spectrum):
        from scipy.optimize import curve_fit
        X = np.arange(1, len(spectrum)+1)
        popt, pcov = curve_fit(powerlaw, X, spectrum)
        return popt[0], popt[1]
    alpha, beta = fit_powerlaw(spectrum)
    return alpha, beta

def img_to_power_spectrum(img):
    """
    Get power spectrum from image
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift+1e-5+1e-5j)) # Avoiding log(0)
    return magnitude_spectrum

def insert_multiple_features_to_data(data, features):
    """
    Insert a list of features to data
    """
    # Iterate over list of features and insert them
    for feature in features:
        data = np.insert(data, data.shape[-1], feature, axis=1)
    return data

"""
Perry, Jeffrey S., and Wilson S. Geisler. "Gaze-contingent real-time simulation of arbitrary visual fields." Human vision and electronic imaging VII. Vol. 4662. International Society for Optics and Photonics, 2002.
Jiang, Ming, et al. "Salicon: Saliency in context." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
"""
def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    # gaussian blur
    Gauss_kernel2D = genGaussiankernel(5, sigma)
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gauss_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    # upsample
    for i in range(1, prNum):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gauss_kernel2D)
        pyramids[i] = curr_im
    return pyramids

def foveat_img(im, fixs):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    sigma = 0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    # compute coef
    p = 7.5
    k = 3
    alpha = 2.5
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)
    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))
    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma
    omega[omega>1] = 1
    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i
    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))
    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))
    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]
        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])
    im_fov = im_fov.astype(np.uint8)
    return im_fov