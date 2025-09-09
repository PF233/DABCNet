import os
import SimpleITK as sitk
from tqdm import tqdm




label_input_dir = r'..\MMWHS\Train\gt_frame'
label_output_dir = r'..\MMWHS\Train\gt_frame'


target_spacing = [1.4, 1.4, 10]


os.makedirs(label_output_dir, exist_ok=True)


for fname in tqdm(os.listdir(label_input_dir)):
    if not fname.endswith('.nii.gz'):
        continue

    input_path = os.path.join(label_input_dir, fname)
    output_path = os.path.join(label_output_dir, fname)

    
    seg = sitk.ReadImage(input_path)
    original_spacing = seg.GetSpacing()
    original_size = seg.GetSize()

    
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(seg.GetDirection())
    resample.SetOutputOrigin(seg.GetOrigin())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  
    resample.SetDefaultPixelValue(0)

    seg_resampled = resample.Execute(seg)

    
    sitk.WriteImage(seg_resampled, output_path)

print(f"\n resample at ：{label_output_dir}")
