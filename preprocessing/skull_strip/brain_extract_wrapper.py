import os
import subprocess

# mount = '/run/media/jk/Elements'
# main_dir = os.path.join(mount, 'MASTER/')
main_dir = '/Users/julian/master/brain_and_donuts/data/coreg_test/'
data_dir = os.path.join(main_dir, '')
skull_strip_path = os.path.join(os.getcwd(), 'skull_strip.sh')
print(skull_strip_path)

study = 'SPC_301mm_Std_Perfusion_20160103102300_11.nii'
subprocess.run([skull_strip_path, '-i', study], cwd = data_dir)

# subjects = [o for o in os.listdir(data_dir)
#                 if os.path.isdir(os.path.join(data_dir,o))]
#
# for subject in subjects:
#     subject_dir = os.path.join(data_dir, subject)
#     modalities = [o for o in os.listdir(subject_dir)
#                     if os.path.isdir(os.path.join(subject_dir,o))]
#
#     for modality in modalities:
#         modality_dir = os.path.join(subject_dir, modality)
#         studies = [o for o in os.listdir(modality_dir)
#                         if os.path.isfile(os.path.join(modality_dir,o))]
#
#
#         for study in studies:
#             study_path = os.path.join(modality_dir, study)
#             if modality.startswith('Ct') & study.startswith('SPC'):
#                 subprocess.run([skull_strip_path, '-i', study], cwd = modality_dir)
