# FEDn Brain Tumor Segmentation



## Connect client to federation
 - Clone this repo (or create a workspace where you have the client-settings.yaml file) 
 - Download a client cert from studio (client.yaml)
 - Set the absolute path to the client_settings.yaml file as an environment variable named FEDN_CLIENT_SETTINGS:
   ```console
     export FEDN_CLIENT_SETTINGS=<ABSOLUTE-PATH-TO-THIS-REPO-FOLDER>/client_settings.yaml
    ```
- install fedn (recommending using a virtual environment):
  ```console
  pip install fedn==0.15.0
  ```

- connect client to studio:
  ```console
  fedn run client -in client.yaml --secure=True --force-ssl
  ```
  


-------------------------------------------------
## Using Brats 2000 dataset
### Pre-process data
YOU NEED TO HAVE THE BRATS_2020 DATASET ON YOUR MACHINE!  \
Change: <PATH/TO/DATA> to the location you have your data and: <NEW/DATA/PATH>  to the location you wish to store the transformed data.


```console
bin/init_venv.sh
.assist-pytorch-venv/bin/python bin/dataPrep.py transform <PATH/TO/DATA>/BRATS_2020/MICCAI_BraTS2020_TrainingData <NEW/DATA/PATH>
```





# Transform to locally collected hospital data

## Data structure

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
  - val
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


# Local Training and Validation
NEW!


## Train local
docker run --gpus all --shm-size=32gb \
-v <NEW/DATA/PATH>:/var/data \
-v $PWD/client_settings.yaml:/var/client_settings.yaml \
-v $PWD/client:/var/client \
-v $PWD/local_script.py:/var/local_script.py \
-v $PWD/experiments:/experiments \
-v $PWD/mainseed.npz:/experiments/mainseed.npz \
-e ENTRYPOINT_OPTS=--data_path=/var/data \
mattiasakessons/bratspytorch /venv/bin/python  /var/local_script.py  train

## Validate local
docker run --gpus all --shm-size=32gb \
-v <NEW/DATA/PATH>:/var/data \
-v $PWD/client_settings.yaml:/var/client_settings.yaml \
-v $PWD/client:/var/client \
-v $PWD/local_script.py:/var/local_script.py \
-v $PWD/experiments:/experiments \
-v $PWD/mainseed.npz:/experiments/mainseed.npz \
-e ENTRYPOINT_OPTS=--data_path=/var/data \
mattiasakessons/bratspytorch /venv/bin/python  /var/local_script.py validate




