import sys, traceback, os, argparse
sys.path.insert(0, '../')
import data_loader as dl
import numpy as np
import pandas as pd
from giotto.meta_transformers import EntropyGenerator as eg


def compute_persistance_entropy_matrix(input_dir, n_jobs=1):
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = dl.load_structured_data(input_dir, 'data_set.npz')

    ct_inputs = np.squeeze(ct_inputs)
    n_sub, n_x, n_y, n_z = ct_inputs.shape

    columns = ['subj', 'x', 'y', 'z']
    entropy_df = pd.DataFrame(columns=columns)

    homologyDimensions = (0, 1, 2)
    ent = eg(metric='euclidean', max_edge_length=10,
             homology_dimensions=homologyDimensions,
             n_jobs=n_jobs)

    for i in range(n_sub):
        print('Processing subject', ids[i], i, '/', n_sub)
        try:
            subject_input = np.asarray([np.asarray(np.where(ct_inputs[i] == 1)).T])
            subject_entropy = ent.fit_transform(subject_input)
            entropy_df = entropy_df.append(pd.DataFrame([[ids[i], subject_entropy[0], subject_entropy[1], subject_entropy[2]]], columns=columns), ignore_index=True)
        except Exception as e:
            tb = traceback.format_exc()
            print('Entropy calculation failed.')
            print(e)
            print(tb)
            entropy_df = entropy_df.append(pd.DataFrame([[ids[i], np.nan, np.nan, np.nan]], columns=columns), ignore_index=True)

        entropy_df.to_excel(os.path.join(input_dir, 'persistance_df.xlsx'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute persistence entropy per subject/image.')
    parser.add_argument('input_directory')
    parser.add_argument("--jobs", "-j", help='Number of parallel working cores.',
                        type=int, default=1)
    args = parser.parse_args()
    compute_persistance_entropy_matrix(args.input_directory)

