# FEDn Brats pytorch project

## Pre-process data
YOU NEED TO HAVE THE BRATS_2020 DATASET ON YOUR MACHINE! ELSE CONTACT ANDERS.


```console
bin/init_venv.sh
.assist-pytorch-venv/bin/python bin/dataPrep.py transform <PATH/TO/DATA>/BRATS_2020/MICCAI_BraTS2020_TrainingData <NEW/DATA/PATH>
```


## Add client to federation

make sure you have a correct client certificate for current federation (usully named client.yaml)

### Using docker image

change <PATH_TO_YOUR_DATA> to the data path of your local data

docker run --gpus '"device=0"' 
-v $PWD/client.yaml:/app/client.yaml 
-v <NEW/DATA/PATH>:/var/data 
-v $PWD/client_settings.yaml:/var/client_settings.yaml 
-e ENTRYPOINT_OPTS=--data_path=/var/data/ 
mattiasakesson/assist_dockerimage /venv/bin/fedn run client --secure=True --force-ssl -in client.yaml

### Using singularity

Transform the docker image into singularity

```console
singularity build bratspytorch.sif docker://mattiasakessons/bratspytorch:latest
```


## Customize docker image
