import os
import numpy as np
import sys
import fire
import yaml
import json

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)
ROI_SIZE = [256, 256, 120] #[160, 160, 80]
import collections
import os
import shutil
import tempfile
import time
from torch.utils.data import Dataset
#from data_clients_brats2020 import get_clients

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
    RandRotated,
    Rand3DElasticd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism

import torch
from collections import OrderedDict


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
            result.append(d[key] == 0)
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0).float()
        return d

def get_train_transform(bratsdatatest=False):

    print("Using Innovia Augmentation settings with roi size: ", ROI_SIZE)
    if bratsdatatest:
        pixdim = (1.0, 1.0, 1.0)
    else:
        pixdim = (1.0, 1.0, 2.0)

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
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                ),
                RandSpatialCropd(keys=["image", "label"], roi_size=ROI_SIZE, random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

            ]
        )



    return train_transform

def get_val_transform(bratsdatatest=False):

    if bratsdatatest:
        pixdim = (1.0, 1.0, 1.0)
    else:
        pixdim = (1.0, 1.0, 2.0)

    val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=pixdim,
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
    return val_transform


def init_seed(out_path='seed.npz', device=None):
    # Init and save
    model = _compile_model(device)
    _save_model(model, out_path)


def _save_model(model, out_path):

    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def _load_model(model_path, device=None):

    parameters_np = helper.load(model_path)
    model = _compile_model(device)
    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict)



    return model

# define inference method
def inference(input, model):

    VAL_AMP = True
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=ROI_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

def _compile_model(device=None):

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
    ).to(device)

    return model


def train(in_model_path, out_model_path, data_path='/var/data', client_settings_path='/var/client_settings.yaml'):


    print("Training entrypoint starts")
    with open(client_settings_path, 'r') as fh: # Used by CJG for local training

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise
    batch_size = client_settings['batch_size']
    local_epochs = client_settings['local_epochs']
    data_path = client_settings['data_path']

    device = torch.device("cuda:0")

    bratsdatatest = client_settings['bratsdatatest']

    num_workers = client_settings['num_workers']

    print("")
    print("client settings")
    for s in client_settings:
        print(s, ": ", client_settings[s])

    # Load data
    image_files = [os.path.join('train', 'images', i) for i in os.listdir(os.path.join(data_path, 'train', 'images'))] # Changed by CJG to local data
    label_files = [os.path.join('train', 'labels', i) for i in os.listdir(os.path.join(data_path, 'train', 'labels'))] # Changed by CJG to local data

    print("image files")
    for im in image_files:
        print(os.path.join(data_path,im), " - ", os.path.isfile(os.path.join(data_path,im)))
    print("label files")
    for im in label_files:
        print(os.path.join(data_path,im), " - ", os.path.isfile(os.path.join(data_path,im)))

    train_ds = BratsDataset(root_dir=data_path,
                            transform=get_train_transform(bratsdatatest),
                            image_files=image_files,
                            label_files=label_files)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("num workers: ", num_workers)
    print("batch size: ", batch_size)


    print("Load model")
    # Load model
    model = _load_model(in_model_path, device)
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True


    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

    # if optimizer parameters exists then load them
    if os.path.isfile(client_settings['checkpoint_path']):
        checkpoint = torch.load(client_settings['checkpoint_path'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("optimizer parameters loaded")

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.state_dict()

    total_start = time.time()
    for epoch in range(local_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{local_epochs}")
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
                print("inputs shape: ", inputs.shape)
                print("labels shape: ", labels.shape)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            print("model shape: ", model.state_dict()['convInit.conv.weight'].shape)
            print("optimizer shape: ", optimizer.param_groups[0]['params'][0].shape)
            print("device: ", device)
            print("loss: ", loss)

            #for i, (name, param) in enumerate(model.named_parameters()):
            #    print(name, param.size(), " - ", optimizer.param_groups[0]['params'][i].shape)


            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )

    total_time = time.time() - total_start

    # Metadata needed for aggregation server side. DUMMY VALUES!
    metadata = {
        'num_examples': len(train_ds),
        'batch_size': 1,
        'epochs': 1,
        'lr': 0.001
    }

    # Save JSON metadata file
    save_metadata(metadata, out_model_path)

    # Save optimizer parameters locally
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
    }, client_settings['checkpoint_path'])

    # Save
    _save_model(model, out_model_path)
    print("Model training done!")


def validate(in_model_path, out_json_path, data_path='/var/data', client_settings_path='/var/client_settings.yaml'):

    with open(client_settings_path, 'r') as fh:
        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise
    batch_size = client_settings['batch_size']

    bratsdatatest = client_settings['bratsdatatest']
    if bratsdatatest:
        num_workers = 4
    else:
        num_workers = 12
    device = torch.device("cuda:0")
    data_path = client_settings['data_path']

    # Load data
    image_files = [os.path.join('val', 'images', i) for i in os.listdir(os.path.join(data_path, 'val', 'images'))] # Changed by CJG to local data
    label_files = [os.path.join('val', 'labels', i) for i in os.listdir(os.path.join(data_path, 'val', 'labels'))] # Changed by CJG to local data


    val_ds = BratsDataset(root_dir=data_path, transform=get_val_transform(bratsdatatest), image_files=image_files,
                          label_files=label_files)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=num_workers)

    model = _load_model(in_model_path, device)
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    model.eval()
    with torch.no_grad():

        for val_data in val_loader:

            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = inference(val_inputs, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)


        metric = dice_metric.aggregate().item()
        metric_batch = dice_metric_batch.aggregate()
        metric_bg = metric_batch[0].item()
        metric_GTV = metric_batch[1].item()
        metric_CTV = metric_batch[2].item()
        metric_Brainstem = metric_batch[3].item()


    results = {'meandice': metric, 'diceBackground': metric_bg, 'diceGTV': metric_GTV, 'diceCTV': metric_CTV, 'diceBrainstem': metric_Brainstem}
    print("val results: ")
    for k in results:
        print(k, ": ", results[k])

    # Save JSON
    save_metrics(results, out_json_path)



class ConvertToMultiChannelBasedOnBratsClassesd_old(MapTransform):
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




class BratsDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_files=None, label_files=None):
        self.root_dir = root_dir
        self.transform = transform
        #self.image_dir = os.path.join(self.root_dir, "images")
        #self.label_dir = os.path.join(self.root_dir, "labels")
        self.image_dir = os.path.join(self.root_dir) #CJG
        self.label_dir = os.path.join(self.root_dir)#CJG
        if image_files and label_files:

            #print("if")
            self.image_files = sorted(image_files)
            self.label_files = sorted(label_files)
        else:

            self.image_files = sorted(os.listdir(self.image_dir))
            self.label_files = sorted(os.listdir(self.label_dir))

        assert len(self.image_files) == len(self.label_files), "Number of image files and label files do not match."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        label_file = os.path.join(self.label_dir, self.label_files[idx])

        sample = {"image": image_file, "label": label_file}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    fire.Fire({
        'init_seed': init_seed,
        'train': train,
        'validate': validate,
    })