import os
import numpy as np
import nibabel as nib



def crop_or_pad_depth(image, target_d=16):

    h, w, d = image.shape
    if d > target_d:
        
        start_d = (d - target_d) // 2
        end_d = start_d + target_d
        cropped_image = image[:, :, start_d:end_d]
    else:
        
        cropped_image = np.zeros((h, w, target_d), dtype=image.dtype)
        
        start_d = (target_d - d) // 2
        cropped_image[:, :, start_d:start_d + d] = image

    return cropped_image
def load_nii_image(file_path):

    img = nib.load(file_path)
    return img, img.get_fdata()


def get_heart_region(seg_data):

    indices = np.argwhere(seg_data > 0)  
    if indices.size == 0:
        return None, None

    min_h, min_w = indices[:, 0].min(), indices[:, 1].min()
    max_h, max_w = indices[:, 0].max(), indices[:, 1].max()

    center_h = (min_h + max_h) // 2
    center_w = (min_w + max_w) // 2

    return center_h, center_w


def crop_image(image_data, center_h, center_w, crop_size=128):

    half_crop = crop_size // 2
    h, w, d = image_data.shape

    
    start_h = center_h - half_crop
    end_h = center_h + half_crop
    start_w = center_w - half_crop
    end_w = center_w + half_crop

    
    cropped_img = np.zeros((crop_size, crop_size, d), dtype=image_data.dtype)

    
    valid_start_h = max(start_h, 0)
    valid_end_h = min(end_h, h)
    valid_start_w = max(start_w, 0)
    valid_end_w = min(end_w, w)

    
    crop_start_h = max(0, -start_h)  
    crop_start_w = max(0, -start_w)
    crop_end_h = crop_start_h + (valid_end_h - valid_start_h)
    crop_end_w = crop_start_w + (valid_end_w - valid_start_w)

    
    cropped_img[crop_start_h:crop_end_h, crop_start_w:crop_end_w, :] = image_data[
        valid_start_h:valid_end_h, valid_start_w:valid_end_w, :
    ]

    return cropped_img

def reorient_to_RAS(img, spacing=[1.3, 1.3, 4.0]):

    old_affine = img.affine.copy()
    origin = old_affine[:3, 3]  

    
    new_affine = np.zeros((4, 4))
    new_affine[0, 0] = -spacing[0]  # R: -X
    new_affine[1, 1] = -spacing[1]  # A: -Y
    new_affine[2, 2] = -spacing[2]  # S: -Z
    new_affine[:3, 3] = origin
    new_affine[3, 3] = 1.0

    return new_affine

def process_files(SEG_DIR, OUTPUT_DIR):

    for seg_filename in os.listdir(SEG_DIR):
        if not seg_filename.endswith(".nii.gz"):
            continue

        seg_path = os.path.join(SEG_DIR, seg_filename)
        seg_img, seg_data = load_nii_image(seg_path)

        
        center_h, center_w = get_heart_region(seg_data)
        if center_h is None or center_w is None:
            print(f"skip {seg_filename}")
            continue

        new_affine = reorient_to_RAS(seg_img, spacing=[1.3, 1.3, 4.0])

        
        cropped_img = crop_image(seg_data, center_h, center_w)
        
        cropped_img = crop_or_pad_depth(cropped_img, 32)

        
        cropped_nifti = nib.Nifti1Image(cropped_img, new_affine, header=seg_img.header)
        output_path = os.path.join(OUTPUT_DIR, seg_filename)
        nib.save(cropped_nifti, output_path)

        print(f"{seg_filename}，({center_h}, {center_w}) {cropped_img.shape}")


if __name__ == "__main__":

    
    SEG_DIR = r".."
    OUTPUT_DIR = r".."
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_files(SEG_DIR, OUTPUT_DIR)
    print("All data finished")