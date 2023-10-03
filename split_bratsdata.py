import fire
from client.data_clients_brats2020 import get_data_list, get_clients
import os
import shutil
#get_clients([client_settings['training_dataset']], os.path.join(data_path, 'images'))


TRAIN_CLIENTS = ['CBICA', 'noname_1', 'TMC', 'TCGA_02', 'TCGA_06', 'TCGA_08', 'TCGA_12', 'TCGA_14', 'TCGA_19', 'TCGA_76', 'noname_2', 'TCGA_CS']

VALIDATION_CLIENTS = ['TCGA_DU', 'TCGA_FG', 'TCGA_HT', 'noname_3']


# Copy data


def main(data_path='/home/mattias/Documents/projects/brats_datasets/BRATS_2020_pytorchformat/train'):





    # Build new data dirs
    new_data_path = '/home/mattias/Documents/projects/brats_datasets/hospitaldata'

    if not os.path.isdir(new_data_path):
        os.makedirs(new_data_path)

    if not os.path.isdir(os.path.join(new_data_path,'train')):
        os.makedirs(os.path.join(new_data_path,'train'))

    if not os.path.isdir(os.path.join(new_data_path,'train','images')):
        os.makedirs(os.path.join(new_data_path,'train', 'images'))

    if not os.path.isdir(os.path.join(new_data_path,'train','labels')):
        os.makedirs(os.path.join(new_data_path,'train', 'labels'))

    if not os.path.isdir(os.path.join(new_data_path,'val')):
        os.makedirs(os.path.join(new_data_path,'val'))

    if not os.path.isdir(os.path.join(new_data_path,'val','images')):
        os.makedirs(os.path.join(new_data_path,'val','images'))

    if not os.path.isdir(os.path.join(new_data_path,'val','labels')):
        os.makedirs(os.path.join(new_data_path,'val','labels'))

    image_files = get_clients(TRAIN_CLIENTS, os.path.join(data_path, 'images'))
    label_files = get_clients(TRAIN_CLIENTS, os.path.join(data_path, 'labels'))


    # Copy train images

    for r_ in image_files:
        image_file = os.path.join(data_path,'images',r_)
        print("exi image path: ", image_file, os.path.isfile(image_file))
        new_image_path = os.path.join(new_data_path,'train','images',r_)
        print("new image path: ", new_image_path)
        if not os.path.isfile(new_image_path):
            shutil.copy(image_file, new_image_path)

    # Copy train labels

    for r_ in label_files:
        label_file = os.path.join(data_path,'labels',r_)
        print("exi label path: ", label_file, os.path.isfile(label_file))
        new_label_path = os.path.join(new_data_path,'train','labels',r_)
        print("new label path: ", new_label_path)
        if not os.path.isfile(new_label_path):
            shutil.copy(label_file, new_label_path)

    image_files = get_clients(VALIDATION_CLIENTS, os.path.join(data_path, 'images'))
    label_files = get_clients(VALIDATION_CLIENTS, os.path.join(data_path, 'labels'))

    # Copy val images

    for r_ in image_files:
        image_file = os.path.join(data_path,'images',r_)
        print("exi image path: ", image_file, os.path.isfile(image_file))
        new_image_path = os.path.join(new_data_path,'val','images',r_)
        print("new image path: ", new_image_path)
        if not os.path.isfile(new_image_path):
            shutil.copy(image_file, new_image_path)

    # Copy train labels

    for r_ in label_files:
        label_file = os.path.join(data_path,'labels',r_)
        print("exi label path: ", label_file, os.path.isfile(label_file))
        new_label_path = os.path.join(new_data_path,'val','labels',r_)
        print("new label path: ", new_label_path)
        if not os.path.isfile(new_label_path):
            shutil.copy(label_file, new_label_path)








if __name__ == '__main__':
    fire.Fire({

       'main':main
    })
