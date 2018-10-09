import os
import subprocess

main_dir = '/Users/julian/master/brain_and_donuts/data/extract_test/'
data_dir = os.path.join(main_dir, '')
extract_vx_path = os.path.join(os.getcwd(), 'extract_vx.sh')
print(extract_vx_path)

study = 'betted_DE_Angio_CT_075_Bv40_3_F_06_Perfusion_20160103102300_8.nii'
subprocess.run([extract_vx_path, '-i', study], cwd = data_dir)
