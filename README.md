# FEDn Brats pytorch project

## 
## Add client to federation

make sure you have a correct client certificate for current federation (usully named client.yaml)

### using docker image

change <PATH_TO_YOUR_DATA> to the data path of your local data

docker run --gpus '"device=0"' -v $PWD/client.yaml:/app/client.yaml -v <PATH_TO_YOUR_DATA>:/var/data -v $PWD/client_settings.yaml:/var/client_settings.yaml -e ENTRYPOINT_OPTS=--data_path=/var/data/ mattiasakesson/assist_dockerimage /venv/bin/fedn run client --secure=True --force-ssl -in client.yaml

### using singularity

Transform the docker image into singularity

```console
singularity build bratspytorch.sif docker://mattiasakessons/bratspytorch:latest
```


## Customize docker image