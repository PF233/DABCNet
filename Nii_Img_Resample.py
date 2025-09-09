import os
import SimpleITK as sitk
from tqdm import tqdm




input_dir = r'..\MMWHS\Train\img_frame'
output_dir = r'..MMWHS\Train\img_sample'



target_spacing = [1.4, 1.4, 10]


os.makedirs(output_dir, exist_ok=True)


for fname in tqdm(os.listdir(input_dir)):
    if not fname.endswith('.nii.gz'):
        continue

    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    
    img = sitk.ReadImage(input_path)
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(img.GetPixelIDValue())

    img_resampled = resample.Execute(img)

    
    sitk.WriteImage(img_resampled, output_path)

print(f"\n  {target_spacing} save at{output_dir}")
