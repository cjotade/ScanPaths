import os
import sys
sys.path.append("..")
import time
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from typing import Optional, List, Union

from utils.utils import create_empty_dict

def train_model(X_train, y_train, X_val, y_val, model, epochs=500, patience=15):
    # Train
    earlies = np.ones(patience) * 1e10
    stop = patience
    t_train = time.process_time() 
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        t = time.process_time() 
        t_losses, v_losses = [], []
        print(f"Epoch {epoch}/{epochs}:")
        for i in range(len(X_train)):
            t_loss = model.train_on_batch(X_train[i], y_train[i])
            t_losses.append(t_loss)
            if i < len(X_val):
                v_loss = model.test_on_batch(X_val[i], y_val[i])
                v_losses.append(v_loss)
        avg_loss_train = np.array(t_losses).mean()
        avg_loss_val = np.array(v_losses).mean()
        train_loss.append(avg_loss_train)
        val_loss.append(avg_loss_val)
        print("{:.2f}s | loss: {:.3f}, val_loss: {:.4f}".format((time.process_time() - t), avg_loss_train, avg_loss_val))
        # Early Stopping
        earlies[epoch % patience] = avg_loss_val
        if earlies.min() != avg_loss_val:
            stop -= 1
            if stop == 0:
                print("=== Early stopping ===")
                break
        else:
            stop = patience  
    print("Training Finished! Elapsed Time: {:.2f} minutes".format((time.process_time() - t_train) / 60))
    return train_loss, val_loss

def forward_n_steps_ahead(model, X_test_i, scaler, n_steps_ahead = [1], img_test_i="", process_functions=[]):
    """
    Use for prediction for the ith image in X_test.
    
    Inputs
    ---------
    X_test_i: (step, seq_len, pos)
        The data of the ith image in X_test. 
    
    Outputs
    ----------
    results_forward: [{
                        i_step_ahead: int,
                        model_pred: (steps, pos),
                        X_test_i: (steps, seq_len, pos),
                        index: (steps)
                    }]
    """
    from .DataProcessor import process_features
    # Check n_steps_ahead is list
    if not isinstance(n_steps_ahead, list):
        n_steps_ahead = [n_steps_ahead]
    results_forward = []
    # Model Prediction
    model_pred = model.predict(X_test_i)
    # Copy X_test_i
    X_test_ii = X_test_i.copy()
    # Store pred if in n_steps_ahead
    if 1 in n_steps_ahead:
        results_forward.append({
                "i_step_ahead": 1,
                "model_pred": model_pred,
                "X_test_i": X_test_ii,
                "index": np.arange(1, len(X_test_ii)+1)
            })
    # Forward pass
    for i_step_ahead in range(2, max(n_steps_ahead)+1):
        features = process_features(model_pred, img_test_i, process_functions=process_functions)
        if len(features) != 0:
            model_pred_features = np.hstack([model_pred, features])  
        else: 
            model_pred_features = model_pred
        model_pred_scaled = scaler.transform(model_pred_features)
        # Update every step in X_test_ii for predict ahead
        X_test_ii = np.concatenate([
            X_test_ii[:, 1:],
            np.expand_dims(model_pred_scaled, axis=1)
        ], axis=1)
        # Predict ahead
        model_pred = model.predict(X_test_ii)
        # Store pred if in n_steps_ahead
        if i_step_ahead in n_steps_ahead:
            results_forward.append({
                "i_step_ahead": i_step_ahead,
                "model_pred": model_pred,
                "X_test_i": X_test_ii,
                "index": np.arange(i_step_ahead, len(X_test_ii)+i_step_ahead)
            })
    return results_forward

def prediction_n_step(model, X_test, scaler, n_steps_ahead = [1], imgs_test=[], process_functions=[]):
    from tqdm import tqdm
    """
    Use prediction on every image in X_test
    
    Outputs
    ------------
    y_preds_dict: {i_step_ahead: (i_img, df(step, pos))}
        Dictionary with values as a list of dataframes.
    
    """
    if not isinstance(n_steps_ahead, list):
        n_steps_ahead = [n_steps_ahead]
    if imgs_test:
        if len(X_test) != len(imgs_test):
            print(len(X_test), len(imgs_test))
        assert len(X_test) == len(imgs_test)
    else:
        imgs_test = ["" for _ in range(len(X_test))]
    # Model Prediction and Padding sequence
    y_preds_dict = create_empty_dict(n_steps_ahead)
    range_tqdm = tqdm(range(len(X_test)))
    # Iterate over every image in X_test list
    for i in range_tqdm:
        range_tqdm.set_description("Making prediction of image %s" % i)
        # Forward n_steps_ahead, returns a list of dicts
        forward_results_list = forward_n_steps_ahead(model, X_test[i], scaler, n_steps_ahead, imgs_test[i], process_functions)
        for forward_dict in forward_results_list:
            i_step_ahead, model_pred, index = forward_dict["i_step_ahead"], forward_dict["model_pred"], forward_dict["index"]
            # Prediction dataframes
            df_y_pred = pd.DataFrame(model_pred, index=index, columns=["x", "y"]) 
            # Store Data
            y_preds_dict[i_step_ahead].append(df_y_pred)
    return y_preds_dict

def sample_mc_dropout(model, X_test, scaler, n_steps_ahead = [1], n_mc_samples=30, imgs_test=[], process_functions=[]):
    """
    Sample from network model.
    
    Outputs
    ----------
    mmcd_pred_dict: {i_step_ahead: (n_images, n_mc_samples, t_steps, pos)}
    """
    from tensorflow.keras import Model
    if imgs_test:
        assert len(X_test) == len(imgs_test)
    else:
        imgs_test = ["" for _ in len(X_test)]
    model_mcdropout = Model(model.inputs, model.outputs)
    # MC-Dropout prediction
    # Store for every sample
    mcd_preds_dict, mcd_idxs_dict = create_empty_dict(n_steps_ahead), create_empty_dict(n_steps_ahead)
    range_X_test = tqdm(range(len(X_test)))
    for i in range_X_test:
        range_X_test.set_description(f"Forward of image {i+1}")
        forward_list = forward_n_steps_ahead(model, X_test[i], scaler, n_steps_ahead, imgs_test[i], process_functions)
        # Store for every image
        mcd_pred, mcd_idxs = create_empty_dict(n_steps_ahead), create_empty_dict(n_steps_ahead)
        for _ in tqdm(range(n_mc_samples)):
            # For every step ahead
            for forward_dict in forward_list:
                i_step_ahead, X_test_i, index = forward_dict["i_step_ahead"], forward_dict["X_test_i"], forward_dict["index"]
                mcd_pred_i = model_mcdropout(X_test_i, training=True)
                mcd_pred[i_step_ahead].append(mcd_pred_i.numpy())
                mcd_idxs[i_step_ahead].append(index)
        for i_step_ahead in mcd_pred.keys():
            mcd_preds_dict[i_step_ahead].append(mcd_pred[i_step_ahead])
            mcd_idxs_dict[i_step_ahead].append(mcd_idxs[i_step_ahead])
    return mcd_preds_dict, mcd_idxs_dict

def model_mc_dropout(model, X_test, scaler, n_steps_ahead = [1], n_mc_samples=30, imgs_test=[], process_functions=[]):
    """
    Model Monte Carlo Dropout (Gal, 2016)
    
    Outputs
    ----------
    mcd_pred_dict: {i_step_ahead: mcd_pred}
    """
    if not isinstance(n_steps_ahead, list):
        n_steps_ahead = [n_steps_ahead]  
    # Get samples by forward in network
    mcd_pred_dict, mcd_idxs_dict = sample_mc_dropout(model, X_test, scaler, n_steps_ahead, n_mc_samples, imgs_test, process_functions=[])
    # Get mean and std over samples and store them in a dict
    mcd_mean_dict, mcd_std_dict = create_empty_dict(n_steps_ahead), create_empty_dict(n_steps_ahead)
    # For every step ahead
    for i_step, mcd_preds in mcd_pred_dict.items():
        # For every i_img
        for i_img, mcd_pred in enumerate(mcd_preds):
            pred_stack = np.stack(mcd_pred)
            mc_mean = pred_stack.mean(axis=0)
            mc_std = pred_stack.std(axis=0)
            mc_idx = np.array(mcd_idxs_dict[i_step])[i_img][0] # Select 0 cause all index of samples are the same 
        
            mcd_mean_dict[i_step].append(
                pd.DataFrame(mc_mean, index=mc_idx, columns=["x", "y"])
            )
            mcd_std_dict[i_step].append(
                pd.DataFrame(mc_std, index=mc_idx, columns=["x", "y"])
            )
    return mcd_mean_dict, mcd_std_dict