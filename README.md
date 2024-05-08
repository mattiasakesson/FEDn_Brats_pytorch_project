# FEDn Brats pytorch project



## Conect client to federation

- Start by cloning this repo


- Make sure you have a correct client certificate for current federation (usually named client.yaml) and put it in this 
repo's basefolder.

- Install fedn (recommending using a venv )

Add this environment variables:
```console
export FEDN_PACKAGE_EXTRACT_DIR=package
export FEDN_AUTH_SCHEME=Token
export FEDN_DATA_PATH=<DATA/PATH>
```


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



