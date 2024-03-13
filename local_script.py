from client.entrypoint import train, validate, init_seed
import os
import fire


def validate_local():

    model_folder = '/experiments/modelfolder'
    result_folder = '/experiments/resultfolder'

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


def train_local():

    model_folder = '/experiments/modelfolder'

    in_model_path = '/experiments/modelfolder/seed.npz'
    for epoch in range(100):
        print()
        print()
        print("epoch: ", epoch)
        print("list model_folder: ", os.listdir(model_folder))
        out_model_path = os.path.join(model_folder,str(epoch) + '.npz')
        train(in_model_path, out_model_path, data_path='/var/data', client_settings_path='/var/client_settings.yaml')
        in_model_path = out_model_path

def init_experiments():


    os.mkdir('/experiments/modelfolder')
    os.mkdir('/experiments/resultfolder')
    init_seed(out_path='/experiments/modelfolder/seed.npz')


if __name__ == '__main__':
    fire.Fire({
        'train': train_local,
        'validate': validate_local,
        'init': init_experiments
    })