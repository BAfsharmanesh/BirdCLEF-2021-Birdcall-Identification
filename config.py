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
    MODEL_ROOT = Path(".")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---- Config ------------