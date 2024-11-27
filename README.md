
# FEDn Brain Tumor Segmentation


 
-------------------------------------------------
Use real MRI data or play around with the Brats data set
## Using Brats 2000 dataset (Optional play example)
### Pre-process data
YOU NEED TO HAVE THE BRATS_2020 DATASET ON YOUR MACHINE!  \
Change: <PATH/TO/DATA> to the location you have your data and: <NEW/DATA/PATH>  to the location you wish to store the transformed data.



## Conect client to federation

- Start by cloning this repo.


-  Download a client cert from your studio project (client.yaml)

- Install fedn (recommending using a venv) and add this environment variables: 
- Add your data path in the file: client_settings.yaml
  
```console
pip install fedn
export FEDN_CLIENT_SETTINGS=<ABSOLUTE-PATH-TO-THIS-REPO-FOLDER>/client_settings.yaml
```

- Join the federation with this command:
```console
fedn client start -in client.yaml --secure=True --force-ssl
```


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



Name convention of sample pair should start with identifier in name. \
Example: \
../images/example23_image.nii.gz \
../labels/example23_label.nii.gz



