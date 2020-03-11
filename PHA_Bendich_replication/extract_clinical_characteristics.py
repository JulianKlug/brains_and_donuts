import sys
sys.path.insert(0, '../')
import analysis.data_loader as dl
import pandas as pd
import scipy.io as sio

def extract_age(clinical_data_path, data_set_dir):
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = dl.load_structured_data(
        data_set_dir, 'data_set.npz')
    clinical_df = pd.read_excel(clinical_data_path)
    age_array = [clinical_df.loc[clinical_df['anonymised_id'] == subj_id]['age'].values[0] for subj_id in ids]
    sio.savemat('age_array.mat', {'age': age_array})

def extract_sex(clinical_data_path, data_set_dir):
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = dl.load_structured_data(
        data_set_dir, 'data_set.npz')
    clinical_df = pd.read_excel(clinical_data_path)
    sex_array = [clinical_df.loc[clinical_df['anonymised_id'] == subj_id]['sex'].values[0] for subj_id in ids]
    def encode_sex(sex_string):
        if sex_string == 'F' or sex_string == 'Female':
            return 1
        if sex_string == 'M' or sex_string == 'Male':
            return 2
    sex_array = list(map(encode_sex, sex_array))
    sio.savemat('sex_array.mat', {'sex': sex_array})




# extract_sex('/Users/julian/OneDrive - unige.ch/stroke_research/clinical_data/all_2016_2017/data_23072019/2016_2017_output_df.xlsx', '/Users/julian/temp/vessels_tests')