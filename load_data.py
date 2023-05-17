import os

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch import stack
from torch.utils.data import Dataset


class D4Augmentation:
    def __call__(self, img):
        transforms = [
            img,
            T.functional.rotate(img, 90),
            T.functional.rotate(img, 180),
            T.functional.rotate(img, 270),
            T.functional.hflip(img),
            T.functional.hflip(T.functional.rotate(img, 90)),
            T.functional.vflip(img),
            T.functional.vflip(T.functional.rotate(img, 90)),
        ]
        return stack(transforms)


class LoadDataset(Dataset):

    def __init__(self, data_folder, labels_path, dimension, sample_fraction=1.0, augment=False, channels_last=False):

        assert 0.0 <= sample_fraction <= 1.0, 'Sample fraction must be between 0.0 and 1.0'

        self.data_folder = data_folder
        self.dict_labels = pd.read_csv(labels_path).set_index('id')['label'].to_dict()
        self.sample_fraction = sample_fraction
        self.augment = augment
        num_samples = int(len(self.dict_labels) * self.sample_fraction)
        self.image_files = list(self.dict_labels.keys())[:num_samples]
        self.transform = T.Compose([T.CenterCrop(dimension),
                                    T.ToTensor()])
        self.channels_last = channels_last


    def __len__(self):
        if self.augment:
            return 8*len(self.image_files)
        return len(self.image_files)


    def __getitem__(self, index):
        if self.augment:
            transform_index = index % 8
            original_index = index // 8
            image_file = self.image_files[original_index]
        else:
            image_file = self.image_files[index]
        
        image_path = os.path.join(self.data_folder, image_file + '.tif') 
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.augment:
            d4_transformed_img = D4Augmentation()(image)
            image = d4_transformed_img[transform_index]

        label = self.dict_labels[image_file]

        if self.channels_last:
            image = image.permute(1, 2, 0)

        return image, label

