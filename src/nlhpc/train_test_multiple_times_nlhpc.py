import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import time
import numpy as np 
import pandas as pd 
import tensorflow as tf

import sys
sys.path.append("..")

from main_classes.ParamsCreator import (
    DataParams,
    LabelsParams,
    TimeseriesParams,
    ModelParams,
    ParamsParser
)

from main_classes.ModelCreator import (
    create_model_lstm,
    create_model_attention,
    SALICONtf
)

from main_classes.ModelProcessor import (
    train_model,
    prediction_n_step, 
    model_mc_dropout
)

from main_classes.DataProcessor import DataProcessor

from utils.process_methods import process_foveatedImg
from utils.process_helpers import create_real_dict, adjust_indexes


def set_params(SUBJECT, TRAIN_IMG_TYPE="natural", TARGET_STEPS_AHEAD=1, is_train=True, load_model=False):
    # Parameters for load data
    data_params = DataParams(
        DATA_FOLDER="../../data/",
        IMGS_FOLDER="../../data/images/", 
        HEIGHT_ORIG=1080, 
        WIDTH_ORIG=1920, 
        HEIGHT=768, 
        WIDTH=1024
    )

    # Parameters for create labels
    labels_params = LabelsParams(
        TRAIN_IMG_TYPE=TRAIN_IMG_TYPE, #natural, grey, black, inverted, white_noise, pink_noise, white
        TARGET_STEPS_AHEAD=TARGET_STEPS_AHEAD
    )

    # Parameters to create time series
    timeseries_params = TimeseriesParams(
        SEQ_LENGTH=10
    )

    # Parameters for learning
    model_params = ModelParams(
        INPUT_UNITS=30, 
        LEARNING_RATE=0.0001, 
        EPOCHS=5000, #500,
        PATIENCE=150, #15
        CREATE_MODEL_FN=create_model_attention
    )

    params = ParamsParser(
        data_params,
        labels_params,
        timeseries_params,
        model_params,
        RESULTS_FOLDER="../results/FovSOS-FS/", #"../results/MCD/", #"../results/MCDropout/", 
        CHECKPOINTS_FOLDER="../checkpoints/FovSOS-FS/", #"../checkpoints/MCD/", #"../checkpoints/MCDropout/", 
        N_MC_SAMPLES=None,
        is_train=is_train,
        is_save=True,
        is_save_figs=False,
        load_model=load_model
    )

    process_functions = [
        {
            "process_fn": process_foveatedImg,
            "kwargs": {
                "save_folder": f"../saliency_features/{SUBJECT}/",
                "load_features": True,
                "pretrained_model_fn": SALICONtf,
                "pretrained_model_kwargs": {
                    "salicon_weights": "../checkpoints/SALICON/model_lr0.01_loss_crossentropy.h5",
                    "vgg16_weights": "../checkpoints/SALICON/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
                }
            }
        }
    ]

    process_kwargs = {
        "post_process_kwargs": {
            "feature_selectors_map": [
                {"fit_fn_name": "fit_etr", "threshold": None},
                {"fit_fn_name": "fit_pca", "pca_percent": 0.95}
            ],
            "load_feature_selectors": True,
            "save_feature_selectors": True,
            "feature_selectors_filepath": f"../feature_selectors/{SUBJECT}/feature_selectors_Nt-{TARGET_STEPS_AHEAD}_tIMG-{TRAIN_IMG_TYPE}.joblib",
        },
        "timeseries_kwargs": {
            #"seq_length": 10,
            "transient_response": None,
            "batch_size": None,
            #"target_steps_ahead": params.labels_params.TARGET_STEPS_AHEAD,
        }
    }

    return params, process_functions, process_kwargs

def main(
    SUBJECTS=["s605", "s609", "s611", "s613", "s616", "s617", "s619", "s620", "s622"], 
    N_STEPS_AHEAD=[1, 5, 11, 20], 
    TRAIN_IMG_TYPE="natural",
    TARGETS_STEPS_AHEAD=[1],
    is_train=True,
    load_model=False,
    test_on_train_images=True,
    test_on_rest_images=True,
    enable_continue_if_trained=True,
    return_split_test=True
    ):
    start_time = time.time()
    print("Starting process...")
    for SUBJECT in SUBJECTS: 
        print(f"Working on subject: {SUBJECT}")
        for TARGET_STEPS_AHEAD in TARGETS_STEPS_AHEAD:
            print("USING TARGET_STEPS_AHEAD:", TARGET_STEPS_AHEAD)
            params, process_functions, process_kwargs = set_params(SUBJECT, TRAIN_IMG_TYPE=TRAIN_IMG_TYPE, TARGET_STEPS_AHEAD=TARGET_STEPS_AHEAD, is_train=is_train, load_model=load_model)
            # DataProcessor
            data_processor = DataProcessor(SUBJECT, params)
            print("Loading and split data...")
            # Load data
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.run(
                    params.labels_params.TRAIN_IMG_TYPE,
                    process_functions=process_functions,
                    kwargs=process_kwargs
            )
            print("Creating model...")
            # Creating Model
            model = params.model_params.CREATE_MODEL_FN(
                                input_shape=(X_train[0].shape[1], X_train[0].shape[2]), 
                                input_units=params.model_params.INPUT_UNITS, 
                                learning_rate=params.model_params.LEARNING_RATE
            )
                
            if params.load_model:
                # We use this just for testing purposes
                params_name = params.get_params_name(SUBJECT, N_STEPS_AHEAD[0]) # Load model with N=1
                CHECKPOINT_FOLDER, _, CHECKPOINT_PATH = params.get_folders(params_name)
                if CHECKPOINT_PATH.split('/')[-1] in os.listdir(CHECKPOINT_FOLDER):
                    model.load_weights(CHECKPOINT_PATH).expect_partial()
                    #tf.train.Checkpoint.restore(...).expect_partial()
                    print("Successfully load weights")
                
            # Train
            if params.is_train:
                if params.check_folders(SUBJECT, N_STEPS_AHEAD) and enable_continue_if_trained:
                    print("Continue due to model is already trained")
                    continue
                    
                train_loss, val_loss = train_model(X_train, y_train, X_val, y_val, 
                                                model=model, 
                                                epochs=params.model_params.EPOCHS, 
                                                patience=params.model_params.PATIENCE
                                                )
                
            for n_step in N_STEPS_AHEAD:
                # Update params
                params.get_params_name(SUBJECT, n_step, update=True)
                if params.is_train:
                    # Store model weights
                    model.save_weights(params.CHECKPOINT_PATH)
                    print(f"Successfully save weights in {params.CHECKPOINT_PATH}")

            if test_on_train_images:
                process_functions_prediction = [
                    {
                        "process_fn": process_foveatedImg,
                        "kwargs": {
                            "load_features": False,
                            "pretrained_model_fn": SALICONtf,
                            "pretrained_model_kwargs": {
                                "salicon_weights": "../checkpoints/SALICON/model_lr0.01_loss_crossentropy.h5",
                                "vgg16_weights": "../checkpoints/SALICON/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
                            },
                            "feature_selectors": data_processor.get_feature_selectors()
                        }
                    }
                ]
                _, _, imgs_test = data_processor.get_img_names()
                scaler = data_processor.get_scaler()
                # Prediction train images
                y_pred_dict = prediction_n_step(
                                        model=model, 
                                        X_test=X_test,
                                        scaler=scaler, 
                                        n_steps_ahead=N_STEPS_AHEAD,
                                        imgs_test=imgs_test,
                                        process_functions=process_functions_prediction
                )
                y_real_dict = create_real_dict(y_real=y_test, n_steps_ahead=N_STEPS_AHEAD)
                
                if params.N_MC_SAMPLES:
                    mcdropout_mean_dict, mcdropout_std_dict = model_mc_dropout(
                                                                    model=model, 
                                                                    X_test=X_test, 
                                                                    scaler=scaler, 
                                                                    n_steps_ahead=N_STEPS_AHEAD, 
                                                                    n_mc_samples=params.N_MC_SAMPLES,
                                                                    imgs_test=imgs_test,
                                                                    process_functions=process_functions_prediction
                                                            )
                if params.N_MC_SAMPLES:
                    y_real_dict, y_pred_dict, mcdropout_mean_dict, mcdropout_std_dict = adjust_indexes(
                        y_real_dict=y_real_dict, 
                        y_pred_dict=y_pred_dict, 
                        n_steps_ahead=N_STEPS_AHEAD, 
                        mcdropout_dicts=[mcdropout_mean_dict, mcdropout_std_dict])
                else:
                    y_real_dict, y_pred_dict = adjust_indexes(
                        y_real_dict=y_real_dict, 
                        y_pred_dict=y_pred_dict, 
                        n_steps_ahead=N_STEPS_AHEAD) 
                            
                for n_step in N_STEPS_AHEAD:
                    # Update params
                    params.get_params_name(SUBJECT, n_step, update=True)

                    # Store results
                    y_real_concat = pd.concat(y_real_dict[n_step])
                    save_kwargs = {
                        "y_pred": pd.concat(y_pred_dict[n_step]),
                        "y_real": y_real_concat,
                        "idx": y_real_concat.index
                    }
                    if params.is_train:
                        save_kwargs = {
                            **save_kwargs,
                            **{
                                "train_loss": train_loss,
                                "val_loss": val_loss
                            }
                        }
                    if params.N_MC_SAMPLES:
                        save_kwargs = {
                            **save_kwargs, 
                        **{
                            "mcdropout_mean": pd.concat(mcdropout_mean_dict[n_step]),
                            "mcdropout_std": pd.concat(mcdropout_std_dict[n_step]),
                        }
                        }
                    if params.is_save:
                        np.savez(os.path.join(params.SAVE_FOLDER, f"{params.labels_params.TRAIN_IMG_TYPE}.npz"), 
                            **save_kwargs)
                        print("SAVED", params.labels_params.TRAIN_IMG_TYPE)

            # All rest img_types prediction
            if test_on_rest_images:
                for img_type in params.get_rest_img_types(): 
                    print(f"Testing on {img_type} images:")
                    if f"{img_type}.npz" in os.listdir(params.SAVE_FOLDER):
                        print(f"{img_type} predictions already calculated and stored, continue..")
                        continue
                    process_kwargs["post_process_kwargs"]["feature_selectors"] = data_processor.get_feature_selectors()
                    # Load data
                    if return_split_test:
                        (_, _), (_, _), (X_gens_test, y_gens_test) = data_processor.run(
                            img_type, 
                            return_split=return_split_test,
                            process_functions=process_functions,
                            kwargs=process_kwargs
                        )
                    else:
                        X_gens_test, y_gens_test = data_processor.run(
                            img_type, 
                            return_split=False,
                            process_functions=process_functions,
                            kwargs=process_kwargs
                        )
                    process_functions_prediction_test = [
                        {
                            "process_fn": process_foveatedImg,
                            "kwargs": {
                                "load_features": False,
                                "pretrained_model_fn": SALICONtf,
                                "pretrained_model_kwargs": {
                                    "salicon_weights": "../checkpoints/SALICON/model_lr0.01_loss_crossentropy.h5",
                                    "vgg16_weights": "../checkpoints/SALICON/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
                                },
                                "feature_selectors": data_processor.get_feature_selectors()
                            }
                        }
                    ]
                    if return_split_test:
                        _, _, imgs = data_processor.get_img_names(return_split=return_split_test)
                    else:
                        imgs = data_processor.get_img_names(return_split=return_split_test)
                    
                    scaler = data_processor.get_scaler()
                    # Prediction
                    y_pred_dict_test = prediction_n_step(model=model, 
                                                        X_test=X_gens_test,
                                                        scaler=scaler, 
                                                        n_steps_ahead=N_STEPS_AHEAD,
                                                        imgs_test=imgs,
                                                        process_functions=process_functions_prediction_test
                                                        )
                    y_real_dict_test = create_real_dict(y_real=y_gens_test, n_steps_ahead=N_STEPS_AHEAD)

                    if params.N_MC_SAMPLES:
                        # MC Dropout Prediction
                        mcdropout_mean_dict_test, mcdropout_std_dict_test = model_mc_dropout(
                            model=model, 
                            X_test=X_gens_test, 
                            scaler=scaler_x, 
                            n_steps_ahead=N_STEPS_AHEAD, 
                            n_mc_samples=params.N_MC_SAMPLES,
                            imgs_test=imgs,
                            process_functions=process_functions_prediction_test
                        )
                    if params.N_MC_SAMPLES:
                        y_real_dict_test, y_pred_dict_test, mcdropout_mean_dict_test, mcdropout_std_dict_test = adjust_indexes(
                            y_real_dict=y_real_dict_test, 
                            y_pred_dict=y_pred_dict_test, 
                            n_steps_ahead=N_STEPS_AHEAD, 
                            mcdropout_dicts=[mcdropout_mean_dict_test, mcdropout_std_dict_test])
                    else:
                        y_real_dict_test, y_pred_dict_test = adjust_indexes(
                            y_real_dict=y_real_dict_test, 
                            y_pred_dict=y_pred_dict_test, 
                            n_steps_ahead=N_STEPS_AHEAD) 
                    for n_step in N_STEPS_AHEAD:
                        # Update params
                        params.get_params_name(SUBJECT, n_step, update=True)
                        y_real_concat_test = pd.concat(y_real_dict_test[n_step])
                        save_kwargs = {
                            "y_pred": pd.concat(y_pred_dict_test[n_step]),
                            "y_real": y_real_concat_test,
                            "idx": y_real_concat_test.index
                        }
                        if params.N_MC_SAMPLES:
                            save_kwargs = {**save_kwargs, 
                                        **{
                                            "mcdropout_mean": pd.concat(mcdropout_mean_dict_test[n_step]),
                                            "mcdropout_std": pd.concat(mcdropout_std_dict_test[n_step]),
                                        }}
                        if params.is_save:
                            np.savez(os.path.join(params.SAVE_FOLDER, f"{img_type}.npz"), 
                                **save_kwargs)
                            print("SAVED", img_type)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    print("GPU:", tf.config.list_physical_devices('GPU'))

    # Recursive Forecast
    #SUBJECTS = ["s605", "s609", "s611", "s613", "s616", "s617", "s619", "s620", "s622"]
    SUBJECTS = ["s616", "s617"]
    TRAIN_IMG_TYPE = "natural"
    N_STEPS_AHEAD = [1, 5, 11, 20]
    TARGETS_STEPS_AHEAD = [1]
    is_train = False
    load_model = True
    test_on_train_images = False
    test_on_rest_images = True
    enable_continue_if_trained = True
    return_split_test = True

    # Direct Forecast
    #SUBJECTS = ["s605", "s609", "s611", "s613", "s616", "s617", "s619", "s620", "s622"]
    #SUBJECTS = ["s616", "s617", "s619", "s620", "s622"]
    #TRAIN_IMG_TYPE = "natural"
    #N_STEPS_AHEAD = [1]
    #TARGETS_STEPS_AHEAD = [5, 11, 20]
    #is_train = True
    #load_model = False
    #test_on_train_images = True
    #test_on_rest_images = True
    #enable_continue_if_trained = True
    #return_split_test = True

    main(
        SUBJECTS=SUBJECTS, 
        TRAIN_IMG_TYPE=TRAIN_IMG_TYPE,
        N_STEPS_AHEAD=N_STEPS_AHEAD, 
        TARGETS_STEPS_AHEAD=TARGETS_STEPS_AHEAD, 
        is_train=is_train,
        load_model=load_model, 
        test_on_train_images=test_on_train_images, 
        test_on_rest_images=test_on_rest_images,
        enable_continue_if_trained=enable_continue_if_trained,
        return_split_test=return_split_test
    )