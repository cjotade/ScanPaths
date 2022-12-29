import os 
import numpy as np

class DataParams:
    def __init__(self, DATA_FOLDER, IMGS_FOLDER, HEIGHT_ORIG=1080, WIDTH_ORIG=1920, HEIGHT=768, WIDTH=1024):
        self.DATA_FOLDER = DATA_FOLDER
        self.IMGS_FOLDER = IMGS_FOLDER
        self.HEIGHT_ORIG = HEIGHT_ORIG
        self.WIDTH_ORIG = WIDTH_ORIG
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

class LabelsParams:
    def __init__(self, TRAIN_IMG_TYPE, TARGET_STEPS_AHEAD):
        self.TRAIN_IMG_TYPE = TRAIN_IMG_TYPE
        self.TARGET_STEPS_AHEAD = TARGET_STEPS_AHEAD

class TimeseriesParams:
    def __init__(self, SEQ_LENGTH):
        self.SEQ_LENGTH = SEQ_LENGTH

class ModelParams:
    def __init__(self, INPUT_UNITS, LEARNING_RATE, EPOCHS, PATIENCE, CREATE_MODEL_FN=None, PROCESS_FUNCTIONS=[]):
        self.INPUT_UNITS = INPUT_UNITS
        self.LEARNING_RATE = LEARNING_RATE
        self.EPOCHS = EPOCHS
        self.PATIENCE = PATIENCE
        self.CREATE_MODEL_FN = CREATE_MODEL_FN
        self.PROCESS_FUNCTIONS = PROCESS_FUNCTIONS
    
class ParamsParser:
    def __init__(
        self, 
        data_params: DataParams, 
        labels_params: LabelsParams,
        timeseries_params: TimeseriesParams,
        model_params: ModelParams,
        RESULTS_FOLDER: str, 
        CHECKPOINTS_FOLDER: str,
        N_MC_SAMPLES: int,
        is_train: bool,
        is_save: bool,
        is_save_figs: bool,
        load_model: bool
        ):
        # Params
        self.data_params = data_params
        self.labels_params = labels_params
        self.timeseries_params = timeseries_params
        self.model_params = model_params
        # Folders
        self.RESULTS_FOLDER = RESULTS_FOLDER
        self.CHECKPOINTS_FOLDER = CHECKPOINTS_FOLDER
        # Train
        self.N_MC_SAMPLES = N_MC_SAMPLES
        self.is_train = is_train
        # Model
        self.load_model = load_model
        # Storage
        self.is_save = is_save
        self.is_save_figs = is_save_figs

        self.IMG_TYPES = [
            "natural", 
            "white", 
            "black", 
            "grey", 
            "inverted", 
            "pink_noise", 
            "white_noise"
        ]
        # Img types except the one used for training
        assert self.labels_params.TRAIN_IMG_TYPE in self.IMG_TYPES

    def get_params_name(self, subject, n_steps_ahead: int, update=False):
        params_name = f"SUBJECT_{subject}-tIMG_{self.labels_params.TRAIN_IMG_TYPE}-N_{n_steps_ahead}-Nt_{self.labels_params.TARGET_STEPS_AHEAD}-SEQ_{self.timeseries_params.SEQ_LENGTH}-InUts_{self.model_params.INPUT_UNITS}-LR_{self.model_params.LEARNING_RATE}-EPOCHS_{self.model_params.EPOCHS}-PAT_{self.model_params.PATIENCE}"
        if update:
            self.update_params(params_name)
        return params_name

    def get_folders(self, params_name):
        CHECKPOINT_FOLDER = os.path.join(self.CHECKPOINTS_FOLDER, params_name) 
        SAVE_FOLDER = os.path.join(self.RESULTS_FOLDER, params_name)
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, "checkpoint")
        return CHECKPOINT_FOLDER, SAVE_FOLDER, CHECKPOINT_PATH

    def update_params(self, params_name):
        self.CHECKPOINT_FOLDER, self.SAVE_FOLDER, self.CHECKPOINT_PATH = self.get_folders(params_name)
        if not os.path.exists(self.CHECKPOINT_FOLDER):
            os.makedirs(self.CHECKPOINT_FOLDER)
        if not os.path.exists(self.SAVE_FOLDER):
            os.makedirs(self.SAVE_FOLDER)

    def check_folders(self, subject, n_steps_ahead_arr):
        if not isinstance(n_steps_ahead_arr, list):
            n_steps_ahead_arr = [n_steps_ahead_arr]
        for n_steps_ahead in n_steps_ahead_arr:
            params_name = self.get_params_name(subject, n_steps_ahead)
            CHECKPOINT_FOLDER, _, CHECKPOINT_PATH = self.get_folders(params_name)
            if os.path.exists(CHECKPOINT_FOLDER):
                check_bool = CHECKPOINT_PATH.split('/')[-1] in os.listdir(CHECKPOINT_FOLDER)
            else:
                return False
            if not check_bool:
                return False
        return True
    
    def get_rest_img_types(self):
        img_types = self.IMG_TYPES.copy()
        img_types.pop(np.where(np.array(img_types) == self.labels_params.TRAIN_IMG_TYPE)[0][0])
        return img_types