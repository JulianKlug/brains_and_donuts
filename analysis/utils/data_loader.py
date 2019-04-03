import os
import nibabel as nib
import numpy as np
# import visual

def load_and_save_data(image_path, save_dir, save_name):
    image = nib.load(image_path)
    image_data = image.get_data()
    # tresh_img = treshold(image_data)
    np.savez_compressed(os.path.join(save_dir, save_name), image = image_data)

def load_saved_data(data_path):
    img = np.load(data_path)['image']
    return img

def treshold(img_data):
    return np.where(img_data > 0, 1, 0)

# # Example
# image_dir = '/Users/julian/master/brain_and_donuts/data/extract_test/'
# image_name = 'extracted_betted_DE_Angio_CT_075_Bv40_3_F_06_Perfusion_20160103102300_8.nii'
# img_path = os.path.join(image_dir, image_name)
# 
# # load_and_save_data(img_path, image_dir, 'tresh_vasculature')
# image_data = load_saved_data(os.path.join(image_dir, 'tresh_vasculature.npz'))
# print(image_data.shape)
# visual.display(image_data)
