import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

#
class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.root = root

        
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

        
        self.base_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))  
        ])


        self.augment_transform = T.RandomAffine(
            degrees=15,                   
            translate=(0.1, 0.1),         
            scale=(0.8, 1.2),             
            shear=2                       
        )

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = os.path.basename(image_path).split('_')[-1][:-len(".jpg")]
        GT_path = os.path.join(self.GT_paths, 'ISIC_' + filename + '_segmentation.png')

        image = Image.open(image_path).convert('L')  
        GT = Image.open(GT_path).convert('L')

        
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        GT = GT.resize((self.image_size, self.image_size), Image.NEAREST)

        
        if self.mode == 'train':
            seed = np.random.randint(2147483647)  
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            torch.manual_seed(seed)
            GT = self.augment_transform(GT)

        
        image = self.base_transform(image)
        GT = self.base_transform(GT)

        return image, GT

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train'):
    """Builds and returns Dataloader."""
    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

