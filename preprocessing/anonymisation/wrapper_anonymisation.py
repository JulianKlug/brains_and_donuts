import os
import nibabel as nib
import numpy as np
import pandas as pd

def treshold(img_data):
    return np.where(img_data > 0, 1, 0)

def load_and_save_data(image_path, save_dir, save_name):
    image = nib.load(image_path)
    image_data = image.get_data()
    tresh_img = treshold(image_data)
    np.savez_compressed(os.path.join(save_dir, save_name), image = image_data)
    np.savez_compressed(os.path.join(save_dir, 'tresh_' + save_name), image = tresh_img)


main_dir = '/Users/julian/master/brain_and_donuts/data/multi_subj'
data_dir = os.path.join(main_dir, 'reorganised_test1')
output_dir = os.path.join(main_dir, 'anon_test1')

# Find default Angio file, no MIP projection
angio_start = 'DE_Angio_CT_075'

anonymisation_key_df = pd.DataFrame(columns=['patient', 'patient_id', 'angio_file'])

subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir,o))]

for subject_index, subject in enumerate(subjects):
    subject_dir = os.path.join(data_dir, subject)
    modalities = [o for o in os.listdir(subject_dir)
                    if os.path.isdir(os.path.join(subject_dir,o))]

    for modality in modalities:
        modality_dir = os.path.join(subject_dir, modality)
        studies = [o for o in os.listdir(modality_dir)
                        if os.path.isfile(os.path.join(modality_dir,o))]

        if modality.startswith('Ct'):
            angio_files = [i for i in os.listdir(modality_dir)
                                if os.path.isfile(os.path.join(modality_dir, i))
                                    and i.startswith('extracted_betted_' + angio_start) and i.endswith(subject + '.nii.gz')]
            if len(angio_files) != 1:
                raise Exception('No Angio file found / or collision', subject, angio_files)

            print('Converting to numpy for', subject)
            save_name = 'patient_' + str(subject_index) + '_vasculature.npz'
            load_and_save_data(os.path.join(modality_dir, angio_files[0]), output_dir, save_name)
            anonymisation_key_df = anonymisation_key_df.append(
                {'patient': subject, 'patient_id': subject_index, 'angio_file':angio_files[0]
                }, ignore_index=True)

anonymisation_key_df.to_excel(os.path.join(data_dir, 'anonymisation_key.xlsx'))
