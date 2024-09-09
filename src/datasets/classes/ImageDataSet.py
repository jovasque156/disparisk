import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataSet(Dataset):
    def __init__(self, annotations_df, features, root_dir, transform=None):
        '''
        annotations_df: pandas, data frame with the annotations. Must have at least the id of files, 
        features: dict, a dictionary with the structure
                {
                    'photo_name': <name_col>,
                    'sensitive': <name_col>,
                    'label': <name_col>
                }
        root_dir: str, with the root to the folder containing the images
        transform: list, with the transformation to apply to each image
        '''

        self.annotations_df = annotations_df
        self.indexes = self.annotations_df.index
        self.features = features
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.root_dir,
                                self.annotations_df.loc[self.indexes[idx], self.features['photo_name']]))

        if self.transform is not None:
            image = self.transform(image)

        label = self.annotations_df.loc[self.indexes[idx], self.features['label']]
        sensitive = self.annotations_df.loc[self.indexes[idx], self.features['sensitive']]

        return {'X': image, 'sensitive': sensitive, 'label': label}