import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
sys.path.append("..")

from main_classes.ParamsCreator import (
    DataParams,
    LabelsParams,
    TimeseriesParams,
    ModelParams,
    ParamsParser
)
from main_classes.ModelCreator import SALICONtf
from main_classes.DataProcessor import DataProcessor
from utils.process_methods import process_foveatedImg


def set_params(SUBJECT, TRAIN_IMG_TYPE="natural", TARGET_STEPS_AHEAD=1, load_features=True):
    # Number of Steps Ahead
    N_STEPS_AHEAD = [1, 5, 11, 20]  #1, 5, 11, 20

    # Subject
    #SUBJECT = "s619" # s605, s609, s611, s613, s616, s617, s619, s620, s622

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
        SEQ_LENGTH=10 #10, 30
    )


    # Parameters for learning
    model_params = ModelParams(
        INPUT_UNITS=30, 
        LEARNING_RATE=0.0001, #0.0001, 0.0002
        EPOCHS=500, #500, 1000
        PATIENCE=15 #15, 30
    )

    params = ParamsParser(
        data_params,
        labels_params,
        timeseries_params,
        model_params,
        RESULTS_FOLDER="../results/foveated_pretrained_salicon/", #"../results/MCDropout/", #"../results/simple/"
        CHECKPOINTS_FOLDER="../checkpoints/foveated_pretrained_salicon/", #"../checkpoints/MCDropout/", #"../checkpoints/simple/"
        N_MC_SAMPLES=None,
        is_train=True,
        is_save=True,
        is_save_figs=False,
        load_model=False
    )

    process_functions = [
        {
            "process_fn": process_foveatedImg,
            "kwargs": {
                "save_folder": f"../saliency_features/{SUBJECT}/",
                "load_features": load_features,
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
            "transient_response": None,
            "batch_size": None,
        }
    }

    return params, process_functions, process_kwargs

def main(TRAIN_IMG_TYPE="natural", TARGETS_STEPS_AHEAD=[1], load_features=True):
    SUBJECTS = ["s605", "s609", "s611", "s613", "s616", "s617", "s619", "s620", "s622"]
    #SUBJECTS = ["s605"]
    for SUBJECT in SUBJECTS:
        print("Working on subject:", SUBJECT)
        for TARGET_STEPS_AHEAD in TARGETS_STEPS_AHEAD:
            print("TARGET_STEPS_AHEAD:", TARGET_STEPS_AHEAD)
            params, process_functions, process_kwargs = set_params(SUBJECT, TRAIN_IMG_TYPE=TRAIN_IMG_TYPE, TARGET_STEPS_AHEAD=TARGET_STEPS_AHEAD, load_features=load_features)

            # DataProcessor
            data_processor = DataProcessor(SUBJECT, params) 
            
            data_processor.run(
                img_type=params.labels_params.TRAIN_IMG_TYPE,
                return_split=False,
                process_functions=process_functions,
                kwargs=process_kwargs
                )

if __name__ == "__main__":
    # Params
    TARGETS_STEPS_AHEAD = [1, 5, 11, 20]
    load_features = True


    main(
        TARGETS_STEPS_AHEAD=TARGETS_STEPS_AHEAD,
        load_features=load_features
    )