import logging
import os
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
import json

import os

_FACET_PATH = '/path/to/data_sets/facet/'

# Setting logger
extra = {'app_name':__name__}

# Gets or creates a logger
logger = logging.getLogger(__name__)

# stream handler and fomatter
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(app_name)s : %(message)s')
syslog.setFormatter(formatter)

# set logger level and add handlers
logger.setLevel(logging.INFO)
logger.addHandler(syslog)

# set logger adapter
logger = logging.LoggerAdapter(logger, extra)

def obtain_col_id(df):
        new_dic = {}
        for ed, id_ in enumerate(df.columns):
                new_dic[id_] = ed

        return new_dic

if __name__=='__main__':
        logger.info('Starting processing')
        
        logger.info('Loading dataset')
        ds = pd.read_csv(os.path.join(_FACET_PATH, 'annotations/annotations.csv'))
        
        logger.info('Obtaining columns id')
        columns_id = obtain_col_id(ds)

        to_pil = transforms.ToPILImage()

        new_names = []

        logger.info('Extracting, transforming and loading images')
        for id_ in range(len(ds)):
                image = Image.open(os.path.join(_FACET_PATH, f'imgs/{ds.iloc[id_,columns_id["filename"]]}'))
                bbox = json.loads(ds.iloc[id_, columns_id["bounding_box"]])

                left, top = (bbox['x'], bbox['y'])
                right, bottom = (left + bbox['width'], top + bbox['height'])
                
                # Crop and scale image using bbox
                cropped_image = image.crop((left, top, right, bottom))
                scaled_image = cropped_image.resize((225, 225), Image.ANTIALIAS)
                
                # Save the image by using the person_id, which is unique in ds
                os.makedirs(os.path.join(_FACET_PATH, f'imgs_processed/'), exist_ok=True)
                if os.path.isfile(f'/datasets/CS678/data_sets/facet/imgs_processed/{ds.iloc[id_,columns_id["person_id"]]}.jpg'):
                        logger.debug(f' File {ds.iloc[id_,columns_id["person_id"]]}.jpg alredy exists')
                
                scaled_image.save(f'/datasets/CS678/data_sets/facet/imgs_processed/{ds.iloc[id_,columns_id["person_id"]]}.jpg')
                new_names.append(f'{ds.iloc[id_,columns_id["person_id"]]}.jpg')
        
        logger.info('Saving new annotations')
        ds['filename_id_person'] = new_names
        ds.to_csv(os.path.join(_FACET_PATH, 'annotations/annotations_processed.csv'))
        
        logger.info('Done. The new images has as name the id_person')

        logger.info('Creating the null images')
        image_size = (3, 224, 224)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        null_image = mean.expand(image_size)
        pil_image = to_pil(null_image)
        pil_image.save(os.path.join(_FACET_PATH, 'imgs_processed/null_image.jpg'))

        logger.info('Creating and saving null image for ViT')
        image_size = (3, 384, 384)
        mean_value = 0.5
        null_image = torch.full(image_size, mean_value)
        pil_image = to_pil(null_image)
        pil_image.save(os.path.join(_FACET_PATH, 'imgs_processed/null_image_ViT.jpg'))

