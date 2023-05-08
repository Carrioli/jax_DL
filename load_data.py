from torch import stack
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from random import sample


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

    def __init__(self, data_folder, labels_path, dimension, sample_fraction=1.0, augment=False):
        assert 0.0 <= sample_fraction <= 1.0, 'Sample fraction must be between 0.0 and 1.0'

        self.data_folder = data_folder
        self.dict_labels = pd.read_csv(labels_path).set_index('id')['label'].to_dict()
        self.sample_fraction = sample_fraction
        self.augment = augment

        num_samples = int(len(self.dict_labels) * self.sample_fraction)
        self.image_files = list(self.dict_labels.keys())[:num_samples]
        

        self.transform = T.Compose([T.CenterCrop(dimension),
                                    T.ToTensor()])


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

        return image, label



# sub_path = 'histopathologic-cancer-detection/'
# ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', sample_fraction=0.1, augment=True)
# num_train = int(0.8*len(ds))
# num_test = len(ds) - num_train
# train_ds, test_ds = random_split(ds, [num_train, num_test])
# train_dl = DataLoader(train_ds, batch_size=100, shuffle=True, drop_last=True)
# test_dl = DataLoader(test_ds, batch_size=100, shuffle=True, drop_last=True)


# for (batch_img, batch_label) in iter(dl):
#     print()