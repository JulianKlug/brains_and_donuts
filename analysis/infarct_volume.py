import traceback, os, argparse
import data_loader as dl
import numpy as np
import pandas as pd


def compute_infarct_volume(input_dir):
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = dl.load_structured_data(input_dir, 'data_set.npz')

    mri_lesion_GT = np.squeeze(mri_lesion_GT)
    n_sub, n_x, n_y, n_z = mri_lesion_GT.shape

    columns = ['subj', 'infarct_volume']
    volume_df = pd.DataFrame(columns=columns)

    for i in range(n_sub):
        print('Processing subject', ids[i], i, '/', n_sub)
        try:
            subject_volume = np.sum(mri_lesion_GT[i])
            volume_df = volume_df.append(pd.DataFrame([[ids[i], subject_volume]], columns=columns), ignore_index=True)

        except Exception as e:
            tb = traceback.format_exc()
            print('Volume calculation failed.')
            print(e)
            print(tb)
            volume_df = volume_df.append(pd.DataFrame([[ids[i], np.nan]], columns=columns), ignore_index=True)

        volume_df.to_excel(os.path.join(input_dir, 'infarct_volume_df.xlsx'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute infarct volume per subject/image.')
    parser.add_argument('input_directory')
    args = parser.parse_args()
    compute_infarct_volume(args.input_directory)

