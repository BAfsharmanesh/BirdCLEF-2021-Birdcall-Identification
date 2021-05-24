# -------------- import libraries --------------
import json
import re
import time
from pathlib import Path
import gc
import numpy as np
import torch
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import resnest.torch as resnest_torch
import timm
from  torch.utils.data import DataLoader

from sklearn.metrics import label_ranking_average_precision_score
from tqdm.auto import tqdm

from data_prep import BirdClefDataset
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



# ---------------- get_model ----------------
def get_model(name, num_classes):
    """
    Loads a pretrained model.
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    if "resnest" in name:
        model = getattr(resnest_torch, name)(pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name.startswith("resnext") or  name.startswith("resnet"):
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif name.startswith("tf_efficientnet_b"):
        model = getattr(timm.models.efficientnet, name)(pretrained=True)
    elif "efficientnet-b" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        model = pretrainedmodels.__dict__[name](pretrained='imagenet')

    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)

    return model
# ---------------- get_model ----------------



# ---------------- one_step ----------------
def one_step(xb, yb, net, criterion, optimizer, scheduler=None, device='cpu'):
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    o = net(xb)
    loss = criterion(o, yb)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        l = loss.item()

        o = o.sigmoid()
        yb = (yb > 0.5) * 1.0
        lrap = label_ranking_average_precision_score(yb.cpu().numpy(), o.cpu().numpy())

        o = (o > 0.5) * 1.0

        prec = (o * yb).sum() / (1e-6 + o.sum())
        rec = (o * yb).sum() / (1e-6 + yb.sum())
        f1 = 2 * prec * rec / (1e-6 + prec + rec)

    if scheduler is not None:
        scheduler.step()

    return l, lrap, f1.item(), rec.item(), prec.item()
# ---------------- one_step ----------------



# ---------------- evaluate ----------------
@torch.no_grad()
def evaluate(net, criterion, val_laoder, device='cpu'):
    net.eval()

    os, y = [], []
    val_laoder = tqdm(val_laoder, leave=False, total=len(val_laoder))

    for icount, (xb, yb) in enumerate(val_laoder):
        y.append(yb.to(device))

        xb = xb.to(device)
        o = net(xb)

        os.append(o)

    y = torch.cat(y)
    o = torch.cat(os)

    l = criterion(o, y).item()

    o = o.sigmoid()
    y = (y > 0.5) * 1.0

    lrap = label_ranking_average_precision_score(y.cpu().numpy(), o.cpu().numpy())

    o = (o > 0.5) * 1.0

    prec = ((o * y).sum() / (1e-6 + o.sum())).item()
    rec = ((o * y).sum() / (1e-6 + y.sum())).item()
    f1 = 2 * prec * rec / (1e-6 + prec + rec)

    return l, lrap, f1, rec, prec,
# ---------------- evaluate ----------------



# ---------------- one_epoch ----------------
def one_epoch(net, criterion, optimizer, scheduler, train_laoder, val_laoder, device='cpu'):
    net.train()
    l, lrap, prec, rec, f1, icount = 0., 0., 0., 0., 0., 0
    train_laoder = tqdm(train_laoder, leave=False)
    epoch_bar = train_laoder
    for (xb, yb) in epoch_bar:
        epoch_bar.set_description("----|----|----|----|---->")
        _l, _lrap, _f1, _rec, _prec = one_step(xb, yb, net, criterion, optimizer, device=device)
        l += _l
        lrap += _lrap
        f1 += _f1
        rec += _rec
        prec += _prec

        icount += 1

        if hasattr(epoch_bar, "set_postfix") and not icount % 10:
            epoch_bar.set_postfix(
                loss="{:.6f}".format(l / icount),
                lrap="{:.3f}".format(lrap / icount),
                prec="{:.3f}".format(prec / icount),
                rec="{:.3f}".format(rec / icount),
                f1="{:.3f}".format(f1 / icount),
            )

    scheduler.step()

    l /= icount
    lrap /= icount
    f1 /= icount
    rec /= icount
    prec /= icount

    l_val, lrap_val, f1_val, rec_val, prec_val = evaluate(net, criterion, val_laoder, device=device)

    return (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val)
# ---------------- one_epoch ----------------



# ---------------- AutoSave ----------------
class AutoSave:
    def __init__(self, model_root, top_k=2, metric="f1", mode="min", root=None, name="ckpt"):
        self.top_k = top_k
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.root = Path(root or model_root)
        assert self.root.exists()
        self.name = name

        self.top_models = []
        self.top_metrics = []

    def log(self, model, metrics):
        metric = metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank + 1, metric)
        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop(0)

        self.logs.append(metrics)
        self.save(model, metric, rank, metrics["epoch"])

    def save(self, model, metric, rank, epoch):
        t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(self.name, epoch, self.metric, metric, t)
        name = re.sub(r"[^\w_-]", "", name) + ".pth"
        path = self.root.joinpath(name)

        old_model = None
        self.top_models.insert(rank + 1, name)
        if len(self.top_models) > self.top_k:
            old_model = self.root.joinpath(self.top_models[0])
            self.top_models.pop(0)

        torch.save(model.state_dict(), path.as_posix())

        if old_model is not None:
            old_model.unlink()

        self.to_json()

    def rank(self, val):
        r = -1
        for top_val in self.top_metrics:
            if val <= top_val:
                return r
            r += 1

        return r

    def to_json(self):
        # t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_logs".format(self.name)
        name = re.sub(r"[^\w_-]", "", name) + ".json"
        path = self.root.joinpath(name)

        with path.open("w") as f:
            json.dump(self.logs, f, indent=2)
# ---------------- AutoSave ----------------



# ---------------- one_fold ----------------
def one_fold(model_name,df , fold, train_set, val_set, model_root, epochs=20, save=True, save_root=None, device='cpu'):

  save_root = Path(save_root) or model_root

  saver = AutoSave(model_root=MODEL_ROOT, root=save_root, name=f"birdclef_{model_name}_fold{fold}", metric="f1_val")

  net = get_model(model_name, num_classes=NUM_CLASSES).to(device)

  criterion = nn.BCEWithLogitsLoss()

  optimizer = optim.Adam(net.parameters(), lr=8e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=epochs)

  train_data = BirdClefDataset(meta=df.loc[train_set].reset_index(drop=True),
                           sr=SR, num_classes=NUM_CLASSES, duration=DURATION, is_train=True)
  train_laoder = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True)
  val_data = BirdClefDataset(meta=df.loc[val_set].reset_index(drop=True),  sr=SR, num_classes=NUM_CLASSES, duration=DURATION, is_train=False)
  val_laoder = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS, shuffle=False)

  epochs_bar = tqdm(list(range(epochs)), leave=False)
  for epoch  in epochs_bar:
    epochs_bar.set_description(f"--> [EPOCH {epoch:02d}]")
    net.train()

    (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val) = one_epoch(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_laoder=train_laoder,
        val_laoder=val_laoder,
        device=device
      )

    epochs_bar.set_postfix(
    loss="({:.6f}, {:.6f})".format(l, l_val),
    prec="({:.3f}, {:.3f})".format(prec, prec_val),
    rec="({:.3f}, {:.3f})".format(rec, rec_val),
    f1="({:.3f}, {:.3f})".format(f1, f1_val),
    lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
    )

    print(
        "[{epoch:02d}] loss: {loss} lrap: {lrap} f1: {f1} rec: {rec} prec: {prec}".format(
            epoch=epoch,
            loss="({:.6f}, {:.6f})".format(l, l_val),
            prec="({:.3f}, {:.3f})".format(prec, prec_val),
            rec="({:.3f}, {:.3f})".format(rec, rec_val),
            f1="({:.3f}, {:.3f})".format(f1, f1_val),
            lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
        )
    )

    if save:
      metrics = {
          "loss": l, "lrap": lrap, "f1": f1, "rec": rec, "prec": prec,
          "loss_val": l_val, "lrap_val": lrap_val, "f1_val": f1_val, "rec_val": rec_val, "prec_val": prec_val,
          "epoch": epoch,
      }

      saver.log(net, metrics)
# ---------------- one_fold ----------------



# ---------------- train ----------------
def train(model_name,df, device='cpu', epochs=20, save=True, n_splits=5, seed=177, save_root=None, suffix="", folds=None):
    gc.collect()
    torch.cuda.empty_cache()

    save_root = save_root or MODEL_ROOT / f"{model_name}{suffix}"
    save_root.mkdir(exist_ok=True, parents=True)

    fold_bar = tqdm(df.reset_index().groupby("fold").index.apply(list).items(), total=df.fold.max() + 1)

    for fold, val_set in fold_bar:
        if folds and not fold in folds:
            continue

        print(f"\n############################### [FOLD {fold}]")
        fold_bar.set_description(f"[FOLD {fold}]")
        train_set = np.setdiff1d(df.index, val_set)

        one_fold(model_name, df, fold=fold, train_set=train_set, val_set=val_set, epochs=epochs, save=save,
                 save_root=save_root, device=device, model_root=MODEL_ROOT)

        gc.collect()
        torch.cuda.empty_cache()
# ---------------- train ----------------

