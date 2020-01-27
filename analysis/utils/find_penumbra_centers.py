import sys, os
sys.path.insert(0, '../')
import numpy as np
from scipy import ndimage
import pandas as pd


def find_penumbra_centers(data_path, save=True):
    '''
    Use a dataset containing Tmax to find the center of mass of the penumbra
    :param data_path:
    :param save:
    :return:
    '''
    dataset = np.load(data_path, allow_pickle=True)

    params = dataset['params']
    if not 'Tmax' in params.tolist()['ct_sequences'][0]:
        raise Exception('Tmax should be the first channel in the CT data')

    ids = dataset['ids']
    Tmax = dataset['ct_inputs'][..., 0]
    penumbra = np.zeros(Tmax.shape)
    penumbra[Tmax >= 6] = 1
    penumbra_centers = [ndimage.measurements.center_of_mass(penumbra[subj]) for subj in range(penumbra.shape[0])]
    df = pd.DataFrame({'id': ids, 'penumbra_center': penumbra_centers})
    if save:
        df.to_excel(os.path.join(os.path.dirname(data_path), 'penumbra_center_coordinates.xlsx'))
    return df

def select_withAngio(penumbra_centers_path, withAngio_df_path):
    penumbra_centers = pd.read_excel(penumbra_centers_path)
    withAngio_df = pd.read_excel(withAngio_df_path)
    filtered_penumbra_centers = penumbra_centers.loc[penumbra_centers['id'].isin(withAngio_df['subj'])]
    filtered_penumbra_centers.to_excel(os.path.join(os.path.dirname(penumbra_centers_path), 'filtered_penumbra_center_coordinates.xlsx'))
    print(filtered_penumbra_centers.head())

if __name__ == '__main__':
    path = sys.argv[1]
    find_penumbra_centers(path)
