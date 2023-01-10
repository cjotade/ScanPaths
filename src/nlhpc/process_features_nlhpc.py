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


def set_params(SUBJECT):
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
        TRAIN_IMG_TYPE="natural", #natural, grey, black, inverted, white_noise, pink_noise, white
        TARGET_STEPS_AHEAD=1
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
                "load_features": False,
                "pretrained_model_fn": SALICONtf,
                "pretrained_model_kwargs": {
                    "salicon_weights": "../checkpoints/SALICON/model_lr0.01_loss_crossentropy.h5",
                    "vgg16_weights": "../checkpoints/SALICON/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
                }
            }
        }
    ]

    return params, process_functions

def main():
    SUBJECTS = ["s605", "s609", "s611", "s613", "s616", "s617", "s619", "s620", "s622"]
    #SUBJECTS = ["s605"]
    for SUBJECT in SUBJECTS:
        print(SUBJECT)
        params, process_functions = set_params(SUBJECT)
        #img_types = [params.labels_params.TRAIN_IMG_TYPE]
        img_types = [params.labels_params.TRAIN_IMG_TYPE] + params.get_rest_img_types()
        #img_types = params.get_rest_img_types()
        # DataProcessor
        data_processor = DataProcessor(SUBJECT, params) 
        # Preprocess data
        for img_type in img_types:
            print(f"Working on images {img_type}")
            # Pre-processing
            print("Pre-processing data...")
            pre_data_blocks = data_processor.pre_process(img_type)
            # Processing
            print("Processing features...")
            pre_features_blocks = data_processor.process_features(pre_data_blocks, process_functions=process_functions)

if __name__ == "__main__":
    main()