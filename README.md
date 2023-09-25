# FEDn Brats pytorch project

## Pre-process data
YOU NEED TO HAVE THE BRATS_2020 DATASET ON YOUR MACHINE! ELSE CONTACT ANDERS. \
Change: <PATH/TO/DATA> to the location you have your data and: <NEW/DATA/PATH>  to the location you wish to store the transformed data.


```console
bin/init_venv.sh
.assist-pytorch-venv/bin/python bin/dataPrep.py transform <PATH/TO/DATA>/BRATS_2020/MICCAI_BraTS2020_TrainingData <NEW/DATA/PATH>
```


## Add client to federation

make sure you have a correct client certificate for current federation (usully named client.yaml)

### Using docker image

Change <NEW/DATA/PATH> to the data path of your local data. \
Make sure your client yaml file is named: client.yaml


```console
docker run --gpus all --shm-size=32gb \
-v $PWD/client.yaml:/app/client.yaml \
-v <NEW/DATA/PATH>:/var/data \
-v $PWD/client_settings.yaml:/var/client_settings.yaml \
-e ENTRYPOINT_OPTS=--data_path=/var/data/ \
mattiasakessons/pytorchtest run client --secure=True --force-ssl -in client.yaml
```

### Using singularity

Transform the docker image into singularity

```console
singularity build bratspytorch.sif mattiasakessons/pytorchtest:latest
```


# Transform to locally collected hospital data

## Data storing structure

- <DATA/PATH>
  - train
    - images
      - trainsample1image.nii.gz \
      ...
      - trainsample1image.nii.gz \
    - labels
      - trainsample1label.nii.gz \
      ...
      - trainsample1label.nii.gz \
  - validate
    - images
      - valsample1image.nii.gz \
      ...
      - valsample1image.nii.gz \
    - labels
      - valsample1label.nii.gz \
      ...
      - valsample1label.nii.gz \



Name convention of sample pair should start with identifator in name. \
Example: \
../images/example23_image.nii.gz \
../labels/example23_label.nii.gz

```console
docker run --gpus all --shm-size=32gb \
-v $PWD/client.yaml:/app/client.yaml \
-v <NEW/DATA/PATH>:/var/data \
-v $PWD/client_settings.yaml:/var/client_settings.yaml \
-e ENTRYPOINT_OPTS=--data_path=/var/data/ \
mattiasakessons/pytorchtest run client --secure=True --force-ssl -in client.yaml
```


