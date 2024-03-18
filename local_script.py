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

    print("result_folder")
    for file in os.listdir(result_folder):
        print("file: ", file)
    tar_filename = '/experiments/experiment_results_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.tar'
    with tarfile.open(tar_filename, 'w') as tar:
        # Add files to the archive
        for file in os.listdir(result_folder):
            print("file: ", file)
            tar.add(os.path.join(result_folder,file))



def train_local():

    in_model_path = '/experiments/mainseed.npz'

    print("init experiment")
    init_experiments()

    for epoch in range(3):
        print()
        print()
        print("epoch: ", epoch)
        print("list model_folder: ", os.listdir(model_folder))
        out_model_path = os.path.join(model_folder,str(epoch) + '.npz')
        train(in_model_path, out_model_path, data_path='/var/data', client_settings_path='/var/client_settings.yaml')
        in_model_path = out_model_path

def init_experiments():

    print("list: ", os.listdir('/experiments'))
    if os.path.isdir(model_folder):
        print("remove")
        shutil.rmtree(model_folder)
    if os.path.isdir(result_folder):
        print("remove")
        shutil.rmtree(result_folder)
    print("list: ", os.listdir('/experiments'))

    os.mkdir(model_folder)
    #if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
    #init_seed(out_path=os.path.join(model_folder, 'seed.npz'))




if __name__ == '__main__':
    fire.Fire({
        'train': train_local,
        'validate': validate_local,
        'init': init_experiments
    })