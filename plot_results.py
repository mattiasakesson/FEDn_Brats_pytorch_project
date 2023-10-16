import numpy as np
from matplotlib import pylab as plt
import json
import yaml
import os
import fire



def main():
    with open('client_settings.yaml', 'r') as fh:

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise

    experiment_name = client_settings['experiment_name']

    if not os.path.isdir(os.path.join(experiment_name,'plots')):
        os.makedirs(os.path.join(experiment_name,'plots'))
    for validation in os.listdir(os.path.join(experiment_name,'validations')):
        print("validation file: ", validation)
        path = os.path.join(experiment_name,'validations',validation)
        results = json.load(open(path))

        plotname = os.path.join(experiment_name,'plots',validation)
        create_plot(results,name=plotname)

def create_plot(results,name):

    metrics = 'meandice', 'diceGTV', 'diceCTV', 'diceBrainstem'
    epochs = [str(e) for e in np.sort(np.int32(list(results.keys())))]
    epochs_int = np.int32(np.array(epochs))
    print("epochs: ", epochs_int)
    values = {m: [results[e][m] for e in epochs] for m in metrics}
    plt.rcParams.update({'font.size': 30})
    f, ax = plt.subplots(2, 2, figsize=(40, 30))
    i = 0
    for x in range(2):
        for y in range(2):
            ax[x, y].plot(epochs_int, values[metrics[i]])
            ax[x, y].set_title(metrics[i],fontsize=50)
            i += 1
    plt.savefig(name)

if __name__ == '__main__':
    fire.Fire({

        'main': main,

    })
