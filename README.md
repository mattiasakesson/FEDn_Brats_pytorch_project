# FEDn Brats pytorch project



## Conect client to federation

- Start by cloning this repo


- Make sure you have a correct client certificate for current federation (usually named client.yaml) and put it in this 
repo's basefolder.

- Install fedn (recommending using a venv) and add this environment variables: 
- Add your data path in the file: client_settings.yaml
  
```console
pip install fedn
export FEDN_PACKAGE_EXTRACT_DIR=package
export FEDN_AUTH_SCHEME=Token
export FEDN_CLIENT_SETTINGS=<ABSOLUTE-PATH-TO-THIS-REPO-FOLDER>/client_settings.yaml
```

- Join the federation with this command:
```console
fedn client start -in client.yaml --secure=True --force-ssl
```

The running scripts will be running in a separate virtual environment located in a created folder named package with name: assist-venv

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



