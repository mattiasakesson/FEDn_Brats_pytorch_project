import os
import numpy as np
import sys
import fire
import yaml
import json
#from fedn.utils.helpers import get_helper
from fedn.utils.pytorchhelper import PytorchHelper

import collections
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism

import torch
from collections import OrderedDict

#HELPER_MODULE = 'pytorchhelper'

def init_seed(out_path='seed.npz', device=None):
    # Init and save
    model = _compile_model(device)
    _save_model(model, out_path)


def _save_model(model, out_path):
    weights = model.state_dict()
    weights_np = collections.OrderedDict()
    print("save model")
    for w in weights:
        print("layer: ", w)
        weights_np[w] = weights[w].cpu().detach().numpy()
    #helper = get_helper(HELPER_MODULE)
    helper = PytorchHelper()
    helper.save_model(weights_np, out_path)


def _load_model(model_path, device=None):
    #import fedn
    #helper = get_helper(HELPER_MODULE)
    #helper = fedn.utils.helpers.get_helper('pytorchhelper')
    helper = PytorchHelper()




    weights_np = helper.load_model(model_path)
    weights = collections.OrderedDict()
    print("model weights list: ")
    for w in weights_np:
        print("layer: ", w)
        weights[w] = torch.tensor(weights_np[w])
    model = _compile_model(device)
    model.load_state_dict(weights)
    model.eval()
    return model

def _compile_model(device=None):
    VAL_AMP = True

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=8,#16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)


    # define inference method
    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)


    return model


def train(in_model_path, out_model_path, data_path='/var/data'):

    #b = np.load(in_model_path)
    #weights_np = OrderedDict()
    #for i in b.files:
    #    print("layer: ", i)
    #    weights_np[i] = b[i]

    #print("listdir var")
    #for f in os.listdir('/var'):
        #nf = os.path.join('/var',f)
        #print(nf, " - ", os.path.exists(nf), os.path.isfile(nf), os.path.isdir(nf))
        #print(f, " - ", os.path.exists(f), os.path.isfile(f), os.path.isdir(f))

        #print(f, " - ", nf, " - ", os.path.isfile(nf), os.path.isfile(f))
    '''
    with open('/var/client_settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise
    '''

    device = torch.device("cuda:0")
    print("Load data")



    # Load data
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    print("is data_path a directory: ", os.path.isdir(data_path))
    print("data_path: ", data_path)
    train_ds = DecathlonDataset(
        root_dir=data_path,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

    print("Load model")
    # Load model
    model= _load_model(in_model_path, device)
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Train
    print("Train")
    epochs = 1
    val_interval = 1


    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    print("Placeholder for training")
    # UNCOMMENT THIS LINES FOR REAL TRAINING

    total_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
        #lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start
    


    # Save
    _save_model(model, out_model_path)



def validate(in_model_path, out_json_path, data_path='/var/data'):

    with open('/var/client_settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise



    # Load data

    helper = PytorchHelper()
    weights = helper.load_model(in_model_path)

    results = {'testspace': 0.5}



    # Save JSON
    with open(out_json_path, "w") as fh:
        fh.write(json.dumps(results))

    return 0


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d

if __name__ == '__main__':
    fire.Fire({
        'init_seed': init_seed,
        'train': train,
        'validate': validate,
    })