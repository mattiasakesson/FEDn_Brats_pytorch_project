from client.entrypoint import train, validate, init_seed
import os
import fire
import tarfile
import datetime
import shutil
model_folder = '/experiments/modelfolder'
result_folder = '/experiments/resultfolder'

def validate_local():

    model_versions = [f for f in os.listdir(model_folder) if f.endswith('.npz')]
    print("model versions: ", model_versions)
    for model_name in model_versions:
        print()
        print()
        epoch = model_name[:-4]



        model_path = os.path.join(model_folder,model_name)
        result_path = os.path.join(result_folder,epoch)
        print("model_path: ", model_path)
        print("epoch: ", epoch)
        validate(model_path, result_path, data_path='/var/data', client_settings_path='/var/client_settings.yaml')


    tar_filename = '/experiments/experiment_results_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.tar'
    with tarfile.open(tar_filename, 'w') as tar:
        # Add files to the archive
        for file in os.listdir(result_folder):

            tar.add(os.path.join(result_folder,file))




def cross_validation(model_folder2='/experiments/importedmodels', result_folder='/experiments/importedmodelsresults'):

    if not os.listdir(result_folder):
        os.mkdir(result_folder)
    for m in os.listdir(model_folder2):
        model_path = os.path.join(model_folder2,m)
        result_path = os.path.join(result_folder,m+'_result')
        validate(model_path, result_path, data_path='/var/data', client_settings_path='/var/client_settings.yaml')


def train_local(epochs=1000):

    in_model_path = '/experiments/mainseed.npz'

    print("init experiment")
    init_experiments()

    for epoch in range(epochs):
        print()
        print()
        print("epoch: ", epoch)

        out_model_path = os.path.join(model_folder,str(epoch) + '.npz')
        train(in_model_path, out_model_path, data_path='/var/data', client_settings_path='/var/client_settings.yaml')
        in_model_path = out_model_path

def init_experiments():

    if os.path.isdir(model_folder):
        shutil.rmtree(model_folder)

    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)

    os.mkdir(model_folder)
    os.mkdir(result_folder)
    #init_seed(out_path=os.path.join(model_folder, 'seed.npz'))




if __name__ == '__main__':
    fire.Fire({
        'train': train_local,
        'validate': validate_local,
        'init': init_experiments,
        'cross_validation': cross_validation
    })