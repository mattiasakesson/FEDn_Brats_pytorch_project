import os
import numpy as np
import sys
import fire
import yaml
import json

import collections
import os
import shutil
import tempfile
import time

import sys
sys.path.append('client')


from client.entrypoint import ConvertToMultiChannelBasedOnBratsClassesd, BratsDataset, _compile_model, inference
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


def train(data_path='/home/mattias/Documents/projects/brats_datasets/hospitaldata/train'):

    with open('client_settings.yaml', 'r') as fh:

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise
    batch_size = client_settings['batch_size']
    epochs = client_settings['epochs']

    device = torch.device("cuda:0")

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

    # image_files = get_clients([client_settings['training_dataset']], os.path.join(data_path, 'images'))
    # label_files = get_clients([client_settings['training_dataset']], os.path.join(data_path, 'labels'))
    image_files = [os.path.join('images', i) for i in
                   os.listdir(os.path.join(data_path, 'images'))]  # Changed by CJG to local data
    label_files = [os.path.join('labels', i) for i in
                   os.listdir(os.path.join(data_path, 'labels'))]  # Changed by CJG to local data

    #print("image_files:")
    #for r in image_files:
     #   print(r, os.path.isfile(r))

   # print("label_files:")
   # for r in label_files:
    #    print(r, os.path.isfile(r))
    print("-- -- -- --")

    print("data path: ", data_path)
    train_ds = BratsDataset(root_dir=data_path,
                            transform=train_transform,
                            image_files=image_files,
                            label_files=label_files)

    #train_ds.__getitem__(1)
    #print("train_ds: ", train_ds)
    #for i in train_ds:
     #   print(i)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)




    # Train
    print("Train")



    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)



    # use amp to accelerate training


    experiment_name = 'testexperiment'
    if os.path.isdir(experiment_name):

        if len(os.listdir(experiment_name)) > 0:

            start_epoch = np.max(np.array([int(mf.split(".")[0]) for mf in os.listdir(experiment_name)]))
            model = _load_model(os.path.join(experiment_name,str(start_epoch)+'.npz'), device)
        else:
            start_epoch = 0
            model = _compile_model(device)


    else:
        os.makedirs(experiment_name)
        start_epoch = 0
        model = _compile_model(device)

    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

    total_start = time.time()
    for epoch in range(start_epoch, start_epoch+epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{start_epoch+epochs}")
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
        _save_model(model, os.path.join(experiment_name,str(epoch)))


    print("Model training done!")


def validate(model_path, data_path='/home/mattias/Documents/projects/brats_datasets/hospitaldata'):



    with open('client_settings.yaml', 'r') as fh: # CJG change

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise
    batch_size = client_settings['batch_size']

    device = torch.device("cuda:0")

    # Load data

    val_transform = Compose(
        [
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
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    # image_files = get_clients([client_settings['validation_dataset']], os.path.join(data_path, 'images'))
    # label_files = get_clients([client_settings['validation_dataset']], os.path.join(data_path, 'labels'))
    image_files = [os.path.join('val', 'images', i) for i in
                   os.listdir(os.path.join(data_path, 'val', 'images'))]  # Changed by CJG to local data
    label_files = [os.path.join('val', 'labels', i) for i in
                   os.listdir(os.path.join(data_path, 'val', 'labels'))]  # Changed by CJG to local data

    #print("val files:")
    #for r in image_files:
    #    print(r)

    val_ds = BratsDataset(root_dir=data_path, transform=val_transform, image_files=image_files,
                          label_files=label_files)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)

    #model = _compile_model(device)
    model = _load_model(model_path, device)
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean")
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
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()

    metric = dice_metric.aggregate().item()
    metric_batch = dice_metric_batch.aggregate()
    metric_bg = metric_batch[0].item()
    metric_GTV = metric_batch[1].item()
    metric_CTV = metric_batch[2].item()
    metric_Brainstem = metric_batch[3].item()


    results = {'meandice': metric, 'diceBackground': metric_bg, 'diceGTV': metric_GTV, 'diceCTV': metric_CTV,
           'diceBrainstem': metric_Brainstem}

    print("results")
    for k in results:
        print(k, ": ", results[k])



    return results


    # Save JSON

def validate_all(modelname, data_path):


    path = 'validations'
    if not os.path.isdir(path):
        os.makedirs(path)
    results = {}
    for model in os.listdir(modelname):
        results[model.split(".")[0]] = validate(os.path.join(modelname,model))

def _save_model(model, out_path):

    weights = model.state_dict()
    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy()
    np.savez_compressed(out_path, **weights_np)


def _load_model(path, device):


    a = np.load(path)
    weights = collections.OrderedDict()
    for i in a.files:

        weights[i] = torch.tensor(a[i])

    model = _compile_model(device)
    model.load_state_dict(weights)
    return model




if __name__ == '__main__':
    fire.Fire({

        'train': train,
        'validate': validate,
        'validateall': validate_all
    })
