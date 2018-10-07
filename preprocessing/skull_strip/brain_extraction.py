import os
import subprocess

main_dir = '/Users/julian/master/brain_and_donuts/data/'
data_dir = os.path.join(main_dir, 'bet_test')

study = 'DE_Angio_CT_075_Bv40_3_F_06_Perfusion_20160103102300_8.nii'
spc = 'SPC_301mm_Std_Perfusion_20160103102300_11.nii'
spc_path = os.path.join(data_dir, spc)

def extract_brain(image_to_bet, no_contrast_anatomical, data_dir):
    output_path = os.path.join(data_dir, 'betted_' + image_to_bet)

    # Use robustfov (FSL) to get FOV on head only
    print('Setting FOV')
    cropped_path = os.path.join(data_dir, 'cropped_' + study)
    subprocess.run(['robustfov', '-i', study, '-r', cropped_path], cwd = data_dir)

    print('Coregistering to', no_contrast_anatomical)
    coreg_name = 'coreg_crp_' + study + '.gz'
    coreg_path = os.path.join(data_dir, coreg_name)
    subprocess.run([
        'flirt',
        '-in',  cropped_path,
        '-ref', spc_path, '-out', coreg_path, '-omat', os.path.join(data_dir, 'coreg.mat'),
        '-bins', '256', '-cost', 'mutualinfo', '-searchrx', '-90', '90', '-searchry', '-90', '90', '-searchrz', '-90', '90', '-dof', '12', '-interp', 'trilinear'
    ], cwd = data_dir)

    print('Removing skull of', no_contrast_anatomical)
    skull_strip_path = os.path.join(os.getcwd(), 'skull_strip.sh')
    subprocess.run([skull_strip_path, '-i', no_contrast_anatomical], cwd = data_dir)

    print('Applying mask')
    mask_path = os.path.join(data_dir, 'betted_' + no_contrast_anatomical + '_Mask.nii.gz')
    subprocess.run([
        'fslmaths', coreg_path, '-mas', mask_path, output_path
    ], cwd = data_dir)

    print('Done with', image_to_bet)


extract_brain(study, spc, data_dir)
