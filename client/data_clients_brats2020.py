import os

#dataset_folder =  "../brats_datasets/BRATS_2020/MICCAI_BraTS2020_TrainingData"
dataset_folder =  "/var/data/BRATS_2020/MICCAI_BraTS2020_TrainingData"


def get_data_list(dataset_folder):
    bratsClients = {}


    bratsClients['all_subjects'] = [record for record in os.listdir(dataset_folder) if
                         record.startswith("BraTS20_Training")]

    bratsClients['CBICA'] = [record for record in os.listdir(dataset_folder) if
                       (record.startswith("BraTS20_Training") and int(record.split("_")[2]) <= 129)]

    bratsClients['noname_1'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 130 <= int(record.split("_")[2]) <= 149)]

    bratsClients['TMC'] = [record for record in os.listdir(dataset_folder) if
                     (record.startswith("BraTS20_Training") and (
                             150 <= int(record.split("_")[2]) <= 157 or int(record.split("_")[2]) == 270))]

    bratsClients['TCGA_02'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 158 <= int(record.split("_")[2]) <= 179)]

    bratsClients['TCGA_06'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 180 <= int(record.split("_")[2]) <= 213)]

    bratsClients['TCGA_08'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 214 <= int(record.split("_")[2]) <= 225)]

    bratsClients['TCGA_12'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 226 <= int(record.split("_")[2]) <= 233)]

    bratsClients['TCGA_14'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 234 <= int(record.split("_")[2]) <= 237)]

    bratsClients['TCGA_19'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 238 <= int(record.split("_")[2]) <= 245)]

    bratsClients['TCGA_76'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 246 <= int(record.split("_")[2]) <= 259)]

    bratsClients['noname_2'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 260 <= int(record.split("_")[2]) <= 269)]

    bratsClients['TCGA_CS'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 271 <= int(record.split("_")[2]) <= 281)]

    bratsClients['TCGA_DU'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 282 <= int(record.split("_")[2]) <= 316)]

    bratsClients['TCGA_FG'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 317 <= int(record.split("_")[2]) <= 322)]

    bratsClients['TCGA_HT'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 323 <= int(record.split("_")[2]) <= 335)]

    bratsClients['noname_3'] = [record for record in os.listdir(dataset_folder) if
                        (record.startswith("BraTS20_Training") and 336 <= int(record.split("_")[2]) <= 369)]

    return bratsClients


def get_clients(client_names,datafolder):

    brats_clients = get_data_list(datafolder)
    data_list = []
    for c in client_names:
        data_list += brats_clients[c]
        
    return data_list
