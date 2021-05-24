# -------------- import libraries --------------
import pandas as pd
from ast import literal_eval
import json
import numpy as np
from torch.utils.data import Dataset
import joblib
from tqdm.auto import tqdm
from config import config
from Seed_everything import seed_everything
# -------------- import libraries --------------

# ------------ Config ------------
MAX_READ_SAMPLES: int = config.MAX_READ_SAMPLES
SR = config.SR
DURATION = config.DURATION
NUM_CLASSES = config.NUM_CLASSES
# ------------ Config ------------
seed_everything()


def get_df(mel_paths, train_label_paths):
    df = None
    LABEL_IDS = {}

    for file_path in mel_paths:
        temp = pd.read_csv(str(file_path), index_col=0)
        temp["impath"] = temp.apply(
            lambda row: file_path.parent / "audio_images/{}/{}.npy".format(row.primary_label, row.filename), axis=1)
        df = temp if df is None else df.append(temp)

    df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)

    for file_path in train_label_paths:
        with open(str(file_path)) as f:
            LABEL_IDS.update(json.load(f))

    return LABEL_IDS, df


def load_data(df):
    def load_row(row):
        # impath = TRAIN_IMAGES_ROOT/f"{row.primary_label}/{row.filename}.npy"
        return row.filename, np.load(str(row.impath))[:MAX_READ_SAMPLES]
    pool = joblib.Parallel(4)
    mapper = joblib.delayed(load_row)
    tasks = [mapper(row) for row in df.itertuples(False)]
    res = pool(tqdm(tasks))
    res = dict(res)
    return res


class BirdClefDataset(Dataset):

    def __init__(self, audio_image_store, meta, sr=SR, is_train=True, num_classes=NUM_CLASSES, duration=DURATION):
        self.audio_image_store = audio_image_store
        self.meta = meta.copy().reset_index(drop=True)
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration * self.sr

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        image = self.audio_image_store[row.filename]

        image = image[np.random.choice(len(image))]
        image = self.normalize(image)

        t = np.zeros(self.num_classes, dtype=np.float32) + 0.0025  # Label smoothing
        t[row.label_id] = 0.995

        return image, t