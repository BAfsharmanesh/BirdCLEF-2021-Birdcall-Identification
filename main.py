# -------------- import libraries --------------
from icecream import ic
import pandas as pd
from pathlib import Path
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy as np

from data_prep import get_df, BirdClefDataset, load_data
import train_model
from train_model import train
from config import config
from Seed_everything import seed_everything
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
DEVICE = config.DEVICE

MODEL_ROOT = config.MODEL_ROOT
Root_PATH =  config.Root_PATH
LABEL_IDS_PATH = config.LABEL_IDS_PATH
TRAIN_METADATA_PATH = config.TRAIN_METADATA_PATH

ic(DEVICE)
# ------------ Config ------------
seed_everything()


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


MEL_PATHS = sorted(Path(Root_PATH).glob(TRAIN_METADATA_PATH))
TRAIN_LABEL_PATHS = sorted(Path(Root_PATH).glob(LABEL_IDS_PATH))


LABEL_IDS, df = get_df(mel_paths=MEL_PATHS, train_label_paths=TRAIN_LABEL_PATHS)
ic(df.shape)
ic(df.head())

ogg_paths = Path("../datasets/Kkiller").glob("**/*.npy")
ogg_name = [str(x).split('\\')[-1].split('.')[0]+'.ogg' for x in ogg_paths]
df = df[df.filename.isin(ogg_name)]

ic(df["primary_label"].value_counts())
ic([df["label_id"].min(), df["label_id"].max()])


audio_image_store = load_data(df)
len(audio_image_store)
print("shape:", next(iter(audio_image_store.values())).shape)
lbd.specshow(next(iter(audio_image_store.values()))[0])
plt.show()

ic(pd.Series([len(x) for x in audio_image_store.values()]).value_counts())

ds = BirdClefDataset(audio_image_store, meta=df, sr=SR, duration=DURATION, is_train=True)
ic(len(ds))

x, y = ds[np.random.choice(len(ds))]
ic([x.shape, y.shape, np.where(y >= 0.5)])
lbd.specshow(x[0])
plt.show()


MODEL_NAMES = ["resnest50"]

for model_name in MODEL_NAMES:
    print("\n\n###########################################", model_name.upper())

    train(
        model_name,
        df,
        audio_image_store,
        device=DEVICE,
        epochs=2,
        suffix=f"_sr{SR}_d{DURATION}_v1_v1",
        folds=[0]
    )