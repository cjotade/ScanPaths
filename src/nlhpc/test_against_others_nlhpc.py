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

HORIZONS = {
    1: {
        "MCD": [1, 1], #[N, Nt]
        "PosScan": [1, 1], #[N, Nt]
        "FovSOS-FS": [1, 1],
        "FovSOS-FS_DirectPred": [1, 1],
    },
    5: {
        "MCD": [5, 1],
        "PosScan": [5, 1],
        "FovSOS-FS": [5, 1],
        "FovSOS-FS_DirectPred": [1, 5],
    },
    11: {
        "MCD": [11, 1],
        "PosScan": [11, 1],
        "FovSOS-FS": [11, 1],
        "FovSOS-FS_DirectPred": [1, 11],
    },
    20: {
        "MCD": [20, 1],
        "PosScan": [20, 1],
        "FovSOS-FS": [20, 1],
        "FovSOS-FS_DirectPred": [1, 20],
    }
    
}

def set_params(SUBJECT, 
        TRAIN_IMG_TYPE="natural", 
        TARGET_STEPS_AHEAD=1, 
        results_folder="../results/FovSOS-FS_DirectPred/", 
        checkpoints_folder="../checkpoints/FovSOS-FS_DirectPred/", 
        is_train=True, 
        load_model=False
    ):
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
        RESULTS_FOLDER=results_folder, #"../results/MCD/", #"../results/MCDropout/", 
        CHECKPOINTS_FOLDER=checkpoints_folder, #"../checkpoints/MCD/", #"../checkpoints/MCDropout/", 
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

def split_params_name(PARAMS_NAME):
    PARAMS_SPLIT = PARAMS_NAME.split("-")
    SUBJECT = PARAMS_SPLIT[0].split("_")[-1]
    TRAIN_IMG_TYPE = "_".join(PARAMS_SPLIT[1].split("_")[1:])
    N_STEPS_AHEAD = int(PARAMS_SPLIT[2].split("_")[-1])
    TARGET_STEPS_AHEAD = int(PARAMS_SPLIT[3].split("_")[-1])
    SEQ_LENGTH = int(PARAMS_SPLIT[4].split("_")[-1])
    INPUT_UNITS = int(PARAMS_SPLIT[5].split("_")[-1])
    LEARNING_RATE = float(PARAMS_SPLIT[6].split("_")[-1])
    EPOCHS = int(PARAMS_SPLIT[7].split("_")[-1])
    PATIENCE = int(PARAMS_SPLIT[8].split("_")[-1])
    return SUBJECT, TRAIN_IMG_TYPE, N_STEPS_AHEAD, TARGET_STEPS_AHEAD, SEQ_LENGTH, INPUT_UNITS, LEARNING_RATE, EPOCHS, PATIENCE

def main(
    SUBJECTS_TEST=["s605", "s609", "s611", "s613", "s616", "s617", "s619", "s620", "s622"], 
    HORIZONS_TEST=[1, 5, 11, 20],
    TRAIN_IMG_TYPE="natural",
    TEST_IMG_TYPE="natural",
    is_train=False,
    load_model=True,
    model_name="FovSOS-FS_DirectPred",
    ):
    start_time = time.time()
    print("Starting process...")
    for horizon_test in HORIZONS_TEST:
        print(f"Horizon: {horizon_test}")
        N, Nt = HORIZONS[horizon_test][model_name]
        checkpoints_folder = f"../checkpoints/{model_name}/"
        filtered_checkpoints_folder = list(
            filter(
                lambda path: path if (TRAIN_IMG_TYPE in path) and (f"N_{N}-" in path) and (f"Nt_{Nt}-" in path) else None, 
                os.listdir(checkpoints_folder)
            )
        )
        print(filtered_checkpoints_folder)
        for PARAMS_NAME in filtered_checkpoints_folder:
            # Create parameters for load trained model
            SUBJECT, TRAIN_IMG_TYPE, N_STEPS_AHEAD, TARGET_STEPS_AHEAD, SEQ_LENGTH, INPUT_UNITS, LEARNING_RATE, EPOCHS, PATIENCE = split_params_name(PARAMS_NAME)
            params, process_functions, process_kwargs = set_params(SUBJECT, 
                TRAIN_IMG_TYPE=TRAIN_IMG_TYPE, 
                TARGET_STEPS_AHEAD=TARGET_STEPS_AHEAD, 
                results_folder=f"../results/{model_name}/", 
                checkpoints_folder=checkpoints_folder, 
                is_train=is_train, 
                load_model=load_model
            )
            data_processor = DataProcessor(SUBJECT, params)
            print("Loading and split data...")
            # Load data
            (X_train_st, _), (_, _), (_, _) = data_processor.run(
                    params.labels_params.TRAIN_IMG_TYPE,
                    process_functions=process_functions,
                    kwargs=process_kwargs
            )
            print("Creating model...")
            # Creating Model
            model = params.model_params.CREATE_MODEL_FN(
                                input_shape=(X_train_st[0].shape[1], X_train_st[0].shape[2]), 
                                input_units=params.model_params.INPUT_UNITS, 
                                learning_rate=params.model_params.LEARNING_RATE
            )
            if params.load_model:
                # We use this just for testing purposes
                CHECKPOINT_FOLDER, _, CHECKPOINT_PATH = params.get_folders(PARAMS_NAME)
                if CHECKPOINT_PATH.split('/')[-1] in os.listdir(CHECKPOINT_FOLDER):
                    model.load_weights(CHECKPOINT_PATH).expect_partial()
                    print("Successfully load weights")
            # Process saliency features since testing purposes
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
            # Scaler from pre-trained model
            scaler = data_processor.get_scaler()

            for SUBJECT_TEST in SUBJECTS_TEST:
                # Folders for test subject
                TEST_FOLDER = os.path.join(params.RESULTS_FOLDER, PARAMS_NAME, "all_vs_all")
                PARAMS_NAME_TEST = f"SUBJECT_{SUBJECT_TEST}-tIMG_{TEST_IMG_TYPE}-N_{N_STEPS_AHEAD}-Nt_{TARGET_STEPS_AHEAD}-SEQ_{SEQ_LENGTH}-InUts_{INPUT_UNITS}-LR_{LEARNING_RATE}-EPOCHS_{EPOCHS}-PAT_{PATIENCE}"
                SAVE_FOLDER = os.path.join(TEST_FOLDER, PARAMS_NAME_TEST) 

                if not os.path.exists(TEST_FOLDER):
                    os.makedirs(TEST_FOLDER)

                if not os.path.exists(SAVE_FOLDER):
                    os.makedirs(SAVE_FOLDER)
                
                if (SUBJECT == SUBJECT_TEST) or (f"{TEST_IMG_TYPE}.npz" in os.listdir(SAVE_FOLDER)):
                    print(f"Same subjects or already processed, {SUBJECT} and {SUBJECT_TEST}")
                    continue

                print(f"Testing on {SUBJECT_TEST}")
                # DataProcessor
                data_processor_test = DataProcessor(SUBJECT_TEST, params)
                print("Loading and split data...")
                # Load data
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor_test.run(
                        TEST_IMG_TYPE,
                        process_functions=process_functions_prediction,
                        kwargs=process_kwargs
                )
                print("Creating model...")

                # Images from test subject
                _, _, imgs_test = data_processor_test.get_img_names()
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
                
                N_STEPS_AHEAD_list = [N_STEPS_AHEAD] if not isinstance(N_STEPS_AHEAD, list) else N_STEPS_AHEAD
                for n_step in N_STEPS_AHEAD_list:
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
                        PARAMS_NAME_TEST_STEPS = f"SUBJECT_{SUBJECT_TEST}-tIMG_{TEST_IMG_TYPE}-N_{n_step}-Nt_{TARGET_STEPS_AHEAD}-SEQ_{SEQ_LENGTH}-InUts_{INPUT_UNITS}-LR_{LEARNING_RATE}-EPOCHS_{EPOCHS}-PAT_{PATIENCE}"
                        SAVE_FOLDER_STEPS = os.path.join(TEST_FOLDER, PARAMS_NAME_TEST_STEPS) 
                        if not os.path.exists(SAVE_FOLDER_STEPS):
                            os.makedirs(SAVE_FOLDER_STEPS)
                        np.savez(os.path.join(SAVE_FOLDER_STEPS, f"{TEST_IMG_TYPE}.npz"), **save_kwargs)
                        print("SAVED", TEST_IMG_TYPE)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    """
    MODIFICAR PARA NATURAL VS OTRAS IMAGENES
    """
    print("GPU:", tf.config.list_physical_devices('GPU'))

    # General parameters
    SUBJECTS = ["s605", "s609", "s611", "s613", "s616", "s617", "s619", "s620", "s622"]
    #TRAIN_IMG_TYPES = ["natural", "grey", "pink_noise", "white", "white_noise", "inverted", "black"]
    TRAIN_IMG_TYPES = ["natural"]
    TEST_IMG_TYPES = ["natural"]
    HORIZONS_TEST = [1, 5, 11, 20]
    model_name = "FovSOS-FS_DirectPred"
    is_train = False
    load_model = True

    for TRAIN_IMG_TYPE in TRAIN_IMG_TYPES:
        print("TRAIN_IMG_TYPE:", TRAIN_IMG_TYPE)
        for TEST_IMG_TYPE in TEST_IMG_TYPES:
            main(
                SUBJECTS_TEST=SUBJECTS, 
                TRAIN_IMG_TYPE=TRAIN_IMG_TYPE,
                TEST_IMG_TYPE=TEST_IMG_TYPE,
                HORIZONS_TEST=HORIZONS_TEST, 
                is_train=is_train,
                load_model=load_model, 
                model_name=model_name,  
            )