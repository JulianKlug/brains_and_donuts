import traceback, os, argparse
import data_loader as dl
import numpy as np
import pandas as pd
import scipy.io as sio


def compute_infarct_volume(input_dir, save_mat):
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = dl.load_structured_data(input_dir, 'data_set.npz')

    ct_lesion_GT = np.squeeze(ct_lesion_GT)
    n_sub, n_x, n_y, n_z = ct_lesion_GT.shape

    columns = ['subj', 'infarct_volume']
    volume_df = pd.DataFrame(columns=columns)

    for i in range(n_sub):
        print('Processing subject', ids[i], i, '/', n_sub)
        try:
            subject_volume = np.sum(ct_lesion_GT[i])
            volume_df = volume_df.append(pd.DataFrame([[ids[i], subject_volume]], columns=columns), ignore_index=True)

        except Exception as e:
            tb = traceback.format_exc()
            print('Volume calculation failed.')
            print(e)
            print(tb)
            volume_df = volume_df.append(pd.DataFrame([[ids[i], np.nan]], columns=columns), ignore_index=True)

        volume_df.to_excel(os.path.join(input_dir, 'infarct_volume_df.xlsx'))

    if save_mat:
        sio.savemat(os.path.join(input_dir, 'infarct_volumes.mat'), {'infarct_volume': volume_df['infarct_volume'].tolist()})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute infarct volume per subject/image.')
    parser.add_argument('input_directory')
    parser.add_argument('-m', '--matlab', action='store_true',
                        help="saves also a .mat file")
    args = parser.parse_args()
    compute_infarct_volume(args.input_directory, save_mat=args.matlab)

