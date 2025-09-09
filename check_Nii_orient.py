import os
import nibabel as nib
import SimpleITK as sitk


import numpy as np
from nibabel.orientations import aff2axcodes


input_dir = r'..'
output_dir = r'..\All_Dataset\''

# os.makedirs(output_dir, exist_ok=True)




def check_orientation(folder_path):

    for filename in os.listdir(folder_path):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            filepath = os.path.join(folder_path, filename)
            try:
                img = nib.load(filepath)
                affine = img.affine
                orientation = aff2axcodes(affine)
                print(f"{filename}: orientation = {orientation}")
            except Exception as e:
                print(f"{filename}: error：{e}")


folder = r"..MMWHS\Train\img_orient"
check_orientation(folder)

# def reorient_image(path):

#     img_nii = nib.load(path)
#     affine = img_nii.affine
#     print(f"{os.path.basename(path)} - Affine:\n{affine}\n")
#
# path = r"
# reorient_image(path)
    
    
    
    


# for file in os.listdir(input_dir):
#     if file.endswith('.nii.gz'):
#         input_path = os.path.join(input_dir, file)
#         output_path = os.path.join(output_dir, file)
#         reorient_image(input_path, output_path)


