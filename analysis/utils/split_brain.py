from data_loader import load_saved_data
from visual import display

image_path = '/Users/julian/master/brain_and_donuts/data/multi_subj/anon_test1/tresh_patient_0_vasculature.npz'
image = load_saved_data(image_path)

n_x, n_y, n_z = image.shape

print(n_y/2)
l_img = image[:,0:int(n_y/2),:]
r_img = image[:,int(n_y/2):,:]

display(l_img)
display(r_img)
