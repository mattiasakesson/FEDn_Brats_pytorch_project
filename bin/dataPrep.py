import os
import nibabel as nib
import numpy as np
import fire

def load_modalities(modality_filenames):
    volumes = []
    for filename in modality_filenames:
        volume = nib.load(filename)
        volumes.append(volume.get_fdata())
    return volumes

def save_4_channel_volume(output_filename, volumes):
    stacked_volume = np.stack(volumes, axis=-1)
    new_image = nib.Nifti1Image(stacked_volume, np.eye(4))
    nib.save(new_image, output_filename)

def sort_modalities(files):
    sorted_modalities = sorted(files, key=lambda x: ('_flair' in x, '_t1' in x, '_t1ce' in x, '_t2' in x))
    return sorted_modalities

def process_brats_data(input_folder, output_folder, seg_folder):
    subjects = [os.path.join(input_folder, d) for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    for subject in subjects:
        for root, dirs, files in os.walk(subject):
            modalities = []
            seg_file = None

            sorted_files = sort_modalities(files)

            for file in sorted_files:
                if file.endswith(".nii"):
                    modality_path = os.path.join(root, file)
                    if '_seg' in file:
                        print(modality_path)
                        seg_file = modality_path
                    else:
                        modalities.append(modality_path)

            if len(modalities) == 4 and seg_file:
                patient_id = os.path.basename(root)
                output_filename = os.path.join(output_folder, f"{patient_id}_4channel.nii.gz")
                seg_output_filename = os.path.join(seg_folder, f"{patient_id}_seg.nii.gz")

                volumes = load_modalities(modalities)
                save_4_channel_volume(output_filename, volumes)
                print(f"Saved 4-channel volume for {patient_id}.")

                seg_volume = nib.load(seg_file)
                new_seg_image = nib.Nifti1Image(seg_volume.get_fdata(), np.eye(4))
                nib.save(new_seg_image, seg_output_filename)
                print(f"Saved segmentation file for {patient_id}.")


def transform_data(input_folder , output_folder):

    os.makedirs(output_folder, exist_ok=True)
    output_folder_image = os.path.join(output_folder, "train/images")  # Replace with your output folder path
    seg_folder = os.path.join(output_folder,"train/labels")  # Replace with your segmentation folder path
    os.makedirs(output_folder_image, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)

    print("Preparing Data")
    process_brats_data(input_folder, output_folder, seg_folder)

if __name__ == '__main__':
    fire.Fire({
        'transform': transform_data,

    })
