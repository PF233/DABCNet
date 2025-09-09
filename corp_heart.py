import os
import numpy as np
import nibabel as nib

def normalize_image(image):

    return (image - np.min(image)) / (np.max(image) - np.min(image))

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
# Oblique (closest to ASR)
def normalize_image_standard(image, mean=109.0, std=216.0, clip_min=-2.0, clip_max=2.0):

    standardized = (image - mean) / std
    standardized = np.clip(standardized, clip_min, clip_max)
    normalized = (standardized - clip_min) / (clip_max - clip_min)
    return normalized

def normalize_per_image(image, clip_min=0.5, clip_max=99.5):

    
    lower = np.percentile(image, clip_min)
    upper = np.percentile(image, clip_max)

    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image



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


def process_files(SEG_DIR, IMG_DIR, OUTPUT_DIR):

    for seg_filename in os.listdir(SEG_DIR):
        if not seg_filename.endswith(".nii.gz"):
            continue

        seg_path = os.path.join(SEG_DIR, seg_filename)
        # seg_img = nib.load(seg_path)
        seg_img, seg_data = load_nii_image(seg_path)

        
        img_filename = seg_filename.replace("_gt", "")
        img_path = os.path.join(IMG_DIR, img_filename)

        if not os.path.exists(img_path):
            print(f"did not find {img_filename}，skip")
            continue

        
        center_h, center_w = get_heart_region(seg_data)
        if center_h is None or center_w is None:
            print(f"skip {seg_filename}, no label")
            continue

        img, img_data = load_nii_image(img_path)

        
        new_affine = reorient_to_RAS(img, spacing=[1.3, 1.3, 4.0])
        
        cropped_img = crop_image(img_data, center_h, center_w, crop_size=128)

        
        cropped_img = crop_or_pad_depth(cropped_img, 32)

        
        normalized_image = normalize_per_image(cropped_img)

        
        cropped_nifti = nib.Nifti1Image(normalized_image, new_affine, header=img.header)
        output_path = os.path.join(OUTPUT_DIR, img_filename)
        nib.save(cropped_nifti, output_path)

        print(f"process {img_filename}，center: ({center_h}, {center_w})，corp size: {cropped_img.shape}")



if __name__ == "__main__":

    
    SEG_DIR = r"..MMWHS\Vali\gt_sample"
    IMG_DIR = r"..MMWHS\Vali\img_sample"
    OUTPUT_DIR = r"..MMWHS\Vali\img_pro"

    
    os.makedirs(OUTPUT_DIR, exist_ok=True)



    process_files(SEG_DIR, IMG_DIR, OUTPUT_DIR)
    print("All data finished")
