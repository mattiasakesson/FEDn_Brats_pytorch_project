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


from client.entrypoint import (ConvertToMultiChannelBasedOnBratsClassesd, BratsDataset, _compile_model, inference,
                               get_train_transform, get_val_transform)
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


def train(data_path):

    with open('client_settings.yaml', 'r') as fh:

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise

    batch_size = client_settings['batch_size']
    epochs = client_settings['epochs']
    experiment_name = client_settings['experiment_name']
    device = torch.device("cuda:0")
    bratsdatatest = client_settings['bratsdatatest']

    if bratsdatatest:
        num_workers = 4
    else:
        num_workers = 12

    # Train data
    image_files = [os.path.join('images', i) for i in
                   os.listdir(os.path.join(data_path, 'images'))]  # Changed by CJG to local data
    label_files = [os.path.join('labels', i) for i in
                   os.listdir(os.path.join(data_path, 'labels'))]  # Changed by CJG to local data

    train_ds = BratsDataset(root_dir=data_path,
                            transform=get_train_transform(bratsdatatest),
                            image_files=image_files,
                            label_files=label_files)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Load data
    if client_settings['validate_during_training']:

        if client_settings['validation_data'] != '':
            validation_data = client_settings['validation_data']
        else:
            validation_data = os.path.join("/".join(data_path.split("/")[:-1]),'val')

        print("validation data: ", validation_data)

        image_files = [os.path.join('images', i) for i in
                       os.listdir(os.path.join(validation_data, 'images'))]
        label_files = [os.path.join('labels', i) for i in
                       os.listdir(os.path.join(validation_data, 'labels'))]

        val_ds = BratsDataset(root_dir=validation_data, transform=get_val_transform(bratsdatatest), image_files=image_files,
                              label_files=label_files)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=num_workers)

        if 'validation_name' in client_settings and client_settings['validation_name'] != '':
            validation_name = client_settings['validation_name']
        else:
            validation_name = "_".join(validation_data.split("/"))
        print("validation name: ", validation_name)
        results = {}
        if os.path.isfile(os.path.join(experiment_name, 'validations', validation_name)):
            print("Found previous validation results.")
            results = json.load(open(os.path.join(experiment_name, 'validations', validation_name)))


    # If new experiment creates an experiment directory
    if os.path.isdir(experiment_name):

        if not os.path.isdir(os.path.join(experiment_name,'weights')):
            print("experiment dir exist without without weights folder. STRANGE!")
            os.makedirs(os.path.join(experiment_name,'weights'))


        if len(os.listdir(os.path.join(experiment_name,'weights'))) > 0:

            latest_model = np.max(np.array([int(mf.split(".")[0]) for mf in os.listdir(os.path.join(experiment_name,'weights'))]))
            model = _load_model(os.path.join(experiment_name,'weights',str(latest_model)+'.npz'), device)
            start_epoch = (latest_model+1)
        else:
            start_epoch = 0
            model = _compile_model(device)


    else:
        os.makedirs(experiment_name)
        os.makedirs(os.path.join(experiment_name, 'weights'))
        os.makedirs(os.path.join(experiment_name, 'validations'))
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
        _save_model(model, os.path.join(experiment_name,'weights',str(epoch)))
        if validation_data:

            results[str(epoch)] = validate_model(model, val_loader, device)

            # Save validation results after each epoch to prevent scenarios when user interrupt training before end.
            with open(os.path.join(experiment_name, 'validations', validation_name), 'w') as outfile:
                json.dump(results, outfile)


    print("Model training done!")


def validate_model(model, val_loader, device):



    print("Validation model")
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

    print("--")
    return results



def validate(data_path):

    with open('client_settings.yaml', 'r') as fh: # CJG change

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise

    device = torch.device("cuda:0")
    experiment_name = client_settings['experiment_name']
    bratsdatatest = client_settings['bratsdatatest']

    if bratsdatatest:
        num_workers=4
    else:
        num_workers=12

    # Load data
    image_files = [os.path.join('images', i) for i in
                   os.listdir(os.path.join(data_path, 'images'))]
    label_files = [os.path.join('labels', i) for i in
                   os.listdir(os.path.join(data_path, 'labels'))]

    val_ds = BratsDataset(root_dir=data_path, transform=get_val_transform(bratsdatatest), image_files=image_files,
                          label_files=label_files)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=num_workers)

    if 'validation_name' in client_settings and client_settings['validation_name'] != '':
        validation_name = client_settings['validation_name']
    else:
        validation_name = "_".join(data_path.split("/"))

    results = {}
    tic = time.time()
    if os.path.isfile(os.path.join(experiment_name, 'validations',validation_name)):
        print("validation on experiment and data exists.")
        print("Checking for experiment updates (new training)")

        results = json.load(open(os.path.join(experiment_name, 'validations',validation_name)))
        prev_rounds = [r + ".npz" for r in list(results.keys())]
        print("prev rounds ", prev_rounds)
        current_rounds = os.listdir(os.path.join(experiment_name, 'weights'))
        print("current rounds: ", current_rounds)
        model_states_to_validate = list(set(current_rounds) - set(prev_rounds))
        print("modelstate updates: ", model_states_to_validate)
    else:
        model_states_to_validate = os.listdir(os.path.join(experiment_name, 'weights'))


    for model_path in model_states_to_validate:
        model = _load_model(os.path.join(experiment_name, 'weights', model_path), device)

        results[model_path.split(".")[0]] = validate_model(model, val_loader, device)
        print("validation time: ", time.time()-tic)
        tic = time.time()
    with open(os.path.join(experiment_name, 'validations',validation_name), 'w') as outfile:
        json.dump(results, outfile)


def _save_model(model, out_path):

    weights = model.state_dict()
    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy()
    print("saving: ", out_path)
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
        'validatemodel': validate_model
    })
