# Brains & Donuts Preprocessing
Goal: extract artery tree

## Skull stripping
1. Use robustfov (FSL) to get FOV on head only
2. Use FLIRT with mutual Information (FSL) to coregister to SPC
3. Create Mask by skull stripping the SPC
4. Apply mask to angio: fslmaths 'image_1.nii.gz' -mas 'mask.nii.gz' 'output_image.nii.gz'
