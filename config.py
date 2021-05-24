# -------------- import libraries --------------
from pathlib import Path
import torch
# -------------- import libraries --------------

class config():
    # ------------ Config ------------
    NUM_CLASSES = 397
    SR = 32_000
    DURATION = 7
    MAX_READ_SAMPLES: int = 3
    TRAIN_BATCH_SIZE = 100
    TRAIN_NUM_WORKERS = 2
    VAL_BATCH_SIZE = 128
    VAL_NUM_WORKERS = 2
    EPOCH_NUM = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_ROOT = Path(".")

    # Root_PATH = "../datasets/Kkiller"
    # LABEL_IDS_PATH = "Kkiller BirdCLEF Mels Computer D7 Part?/LABEL_IDS.json"
    # TRAIN_METADATA_PATH = "Kkiller BirdCLEF Mels Computer D7 Part?/rich_train_metadata.csv"

    Root_PATH = "../input"
    TRAIN_METADATA_PATH = "kkiller-birdclef-mels-computer-d7-part?/rich_train_metadata.csv"
    LABEL_IDS_PATH = "kkiller-birdclef-mels-computer-d7-part?/LABEL_IDS.json"

    neptune = True
    # ---- Config ------------