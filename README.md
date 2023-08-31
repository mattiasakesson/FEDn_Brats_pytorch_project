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
docker run --gpus '"device=0"' \
-v $PWD/client.yaml:/app/client.yaml \
-v <NEW/DATA/PATH>:/var/data \
-v $PWD/client_settings.yaml:/var/client_settings.yaml \
-e ENTRYPOINT_OPTS=--data_path=/var/data/ \
mattiasakesson/assist_dockerimage /venv/bin/fedn run client --secure=True --force-ssl -in client.yaml \
```
The docker image: mattiasakesson/assist_dockerimage was recently updated so if you have run this client-server before you need to remove the docker image or add the --build flag to the script.
### Using singularity

Transform the docker image into singularity

```console
singularity build bratspytorch.sif docker://mattiasakessons/bratspytorch:latest
```


