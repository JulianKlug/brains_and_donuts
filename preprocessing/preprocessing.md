# Brains & Donuts Preprocessing
Goal: extract artery tree

## Convert to Nifti
1. to_nii_batch.py

## Reorganise data
1. organise.py

## Skull stripping - brain_extract_wrapper.py
1. Use robustfov (FSL) to get FOV on head only
2. Use FLIRT with mutual Information (FSL) to coregister to SPC
3. Create Mask by skull stripping the SPC
4. Apply mask to angio

## Blood vessel segmentation - wrapper_vx_extraction.py
1. Threshold masked image at 90 HU
