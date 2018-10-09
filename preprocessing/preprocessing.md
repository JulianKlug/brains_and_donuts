# Brains & Donuts Preprocessing
Goal: extract artery tree

## Skull stripping
1. Use robustfov (FSL) to get FOV on head only
2. Use FLIRT with mutual Information (FSL) to coregister to SPC
3. Create Mask by skull stripping the SPC
4. Apply mask to angio

## Blood vessel segmentation
1. Threshold masked image at 90 HU
