# -------------- import libraries --------------
from icecream import ic
import pandas as pd
from pathlib import Path
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_prep import get_df, BirdClefDataset
import train_model
from train_model import train
from config import config
# -------------- import libraries --------------

# ------------ Config ------------
NUM_CLASSES = config.NUM_CLASSES
SR = config.SR
DURATION = config.DURATION
MAX_READ_SAMPLES: int = config.MAX_READ_SAMPLES
TRAIN_BATCH_SIZE = config.TRAIN_BATCH_SIZE
TRAIN_NUM_WORKERS = config.TRAIN_NUM_WORKERS
VAL_BATCH_SIZE = config.VAL_BATCH_SIZE
VAL_NUM_WORKERS = config.VAL_NUM_WORKERS
MODEL_ROOT = config.MODEL_ROOT
DEVICE = config.DEVICE
# ------------ Config ------------


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


MEL_PATHS = sorted(Path("../datasets/Kkiller").glob("Kkiller BirdCLEF Mels Computer D7 Part?/rich_train_metadata.csv"))
TRAIN_LABEL_PATHS = sorted(Path("../datasets/Kkiller").glob("Kkiller BirdCLEF Mels Computer D7 Part?/LABEL_IDS.json"))

LABEL_IDS, df = get_df(mel_paths=MEL_PATHS, train_label_paths=TRAIN_LABEL_PATHS)

ogg_paths = Path("../datasets/Kkiller").glob("**/*.npy")
ogg_name = [str(x).split('\\')[-1].split('.')[0]+'.ogg' for x in ogg_paths]
df = df[df.filename.isin(ogg_name)]

# ic(df["primary_label"].value_counts())
# ic([df["label_id"].min(), df["label_id"].max()])
# ic(len(df))
#
# for i in range(len(df)):
#     ic(np.load(str(df.impath.iloc[i])).shape)
#     ic(df.duration.iloc[i]/5)
#     # plt.show()

# ds = BirdClefDataset(meta=df, sr=SR, num_classes=NUM_CLASSES, duration=DURATION, is_train=True)
# ic(len(ds))
#
# x, y = ds[np.random.choice(len(ds))]
# ic([x.shape, y.shape, np.where(y >= 0.5)])
# lbd.specshow(x[0])
# plt.title(f"bird name: {[key for key,value in LABEL_IDS.items() if value==np.where(y >= 0.5)[0]]}")
# plt.show()


MODEL_NAMES = ["resnest50"]

for model_name in MODEL_NAMES:
    print("\n\n###########################################", model_name.upper())

    train(
        model_name,
        df,
        device=DEVICE,
        epochs=2,
        suffix=f"_sr{SR}_d{DURATION}_v1_v1",
        folds=[0]
    )