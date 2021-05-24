from pathlib import Path
import torch

class config():
    # ------------ Config ------------
    NUM_CLASSES = 397
    SR = 32_000
    DURATION = 7
    MAX_READ_SAMPLES: int = 5
    TRAIN_BATCH_SIZE = 100
    TRAIN_NUM_WORKERS = 0
    VAL_BATCH_SIZE = 128
    VAL_NUM_WORKERS = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_ROOT = Path(".")
    Root_PATH = "../datasets/Kkiller"
    LABEL_IDS_PATH = "Kkiller BirdCLEF Mels Computer D7 Part?/LABEL_IDS.json"
    TRAIN_METADATA_PATH = "Kkiller BirdCLEF Mels Computer D7 Part?/rich_train_metadata.csv"
    # ---- Config ------------