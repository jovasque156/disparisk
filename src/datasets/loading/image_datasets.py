import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

_FACET_PATH = 'path/to/facet/'

__PHOTO_NAME = {
    'facet': 'filename_id_person'
}

# Setting logger
extra = {'app_name':__name__}

# Gets or creates a logger
logger = logging.getLogger(__name__)

# stream handler and fomatter
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(app_name)s - %(levelname)s : %(message)s')
syslog.setFormatter(formatter)

# set logger level and add handlers
logger.setLevel(logging.INFO)
logger.addHandler(syslog)

# set logger adapter
logger = logging.LoggerAdapter(logger, extra)

facet_classes = {
    "astronaut": 0,
    "backpacker": 1,
    "ballplayer": 2,
    "bartender": 3,
    "basketball_player": 4,
    "boatman": 5,
    "carpenter": 6,
    "cheerleader": 7,
    "climber": 8,
    "computer_user": 9,
    "craftsman": 10,
    "dancer": 11,
    "disk_jockey": 12,
    "doctor": 13,
    "drummer": 14,
    "electrician": 15,
    "farmer": 16,
    "fireman": 17,
    "flutist": 18,
    "gardener": 19,
    "guard": 20,
    "guitarist": 21,
    "gymnast": 22,
    "hairdresser": 23,
    "horseman": 24,
    "judge": 25,
    "laborer": 26,
    "lawman": 27,
    "lifeguard": 28,
    "machinist": 29,
    "motorcyclist": 30,
    "nurse": 31,
    "painter": 32,
    "patient": 33,
    "prayer": 34,
    "referee": 35,
    "repairman": 36,
    "reporter": 37,
    "retailer": 38,
    "runner": 39,
    "sculptor": 40,
    "seller": 41,
    "singer": 42,
    "skateboarder": 43,
    "soccer_player": 44,
    "soldier": 45,
    "speaker": 46,
    "student": 47,
    "teacher": 48,
    "tennis_player": 49,
    "trumpeter": 50,
    "waiter": 51,
}

def load_dataset(ds_name, label=None, sensitive=None, random_split=False, processing_configuration=None):
    '''
    ds_name: str, with the values of celeba or facet
    label: str, name of the column in the datasets to use it as label
    sensitive: str, name of the column in the datasets to use it as sensitive
    processing_configuration: dict, with the following structure (value given are defaults)
        {
            stratify_by: 'label',
            test_size: 0.2, # assume val_size=.2
            val_data: True, 
            null_version: False,
            null_photo_name: None,
            seed: 12345
        }
    '''
    if not random_split:
        logger.warning('given random_split is False')
    if random_split or ds_name=='facet':
        if processing_configuration is None:
            logger.info("random_split=True or ds_name='facet', but no processing_configuration was given. Loading the standard configuration")
            processing_configuration = {
                            'stratify_by': 'label',
                            'test_size': 0.2, 
                            'val_data': True, 
                            'seed': 12345
                        }
        assert all([key in processing_configuration for key in ['stratify_by', 'test_size', 'val_data', 'seed']]), 'processing_configuration with all the keys needed [stratify_by,null_version, test_size, val_data, seed]'    
    
    dataset, features, split = load_facet(label, sensitive)
    split = None #facet distributer doesn't provide a splitter

    return dataset, features, split

def load_facet(label=None, sensitive=None, path=_FACET_PATH):
    '''
    Load and return the dataset, features, and split from FACET dataset
    label: str, name of the column used as label
    sensitive: str, name of the column used as sensitive
    path: str, path to the celeba dataset. Inside must contain:
                'annotations/annotations_processed.csv' with the annotations. Note it's a csv file. Also, run facet_processing.py first

    Note: FACET doesn't have a split given by the distributer
    '''
    label = 'class1' if label is None else label
    sensitive = 'gender_presentation_masc' if sensitive is None else sensitive

    # Load dataset annotators and filter the phot_name, sensitive, and label
    dataset = pd.read_csv(os.path.join(path, 'annotations/annotations_processed.csv'))
    dataset = dataset.loc[:,['filename_id_person', sensitive, label]]

    # Process label and sensitive
    # Both features are 0 or 1
    if label=='gender_presentation_masc':
        dataset[label] = (dataset[label] == 0).astype(int).to_frame()
    elif label== 'class1':
        dataset[label] = dataset[label].apply(lambda x: facet_classes[x]).astype(int).to_frame()
    
    if sensitive=='gender_presentation_masc':
        dataset[sensitive] = (dataset[sensitive] == 0).astype(int).to_frame() # let set as 1 the not male (protected group)
    elif sensitive == 'class1':
        dataset[sensitive] = dataset[sensitive].apply(lambda x: facet_classes[x]).astype(int).to_frame() # let set as 1 the not male (protected group)

    # define features
    features = {
        'photo_name': __PHOTO_NAME['facet'],
        'sensitive': sensitive,
        'label': label,
        'output_size': len(dataset[label].unique()),
        'root_dir': 'imgs_processed/',
    }
    
    # Load split and make sure every id is correct
    split = None

    return dataset, features, split
    

def process_dataset(dataset, features, split, processing_configuration):
    '''
    Prepare dataset to be load to ImageDataSet class
    '''

    if split is None:
        if processing_configuration['stratify_by'] in ['label', 'sensitive']:
            dataset_train, dataset_test = train_test_split(dataset, 
                                                            test_size=processing_configuration['test_size'],
                                                            stratify=dataset[features[processing_configuration['stratify_by']]],
                                                            random_state=processing_configuration['seed'])
        elif processing_configuration['stratify_by'] == 'label_sensitive':
            unique_values_s = len(np.unique(dataset[features['sensitive']]))
            s_y = (dataset[features['label']]).astype('float32').values*unique_values_s + dataset[features['sensitive']].astype('float32').values
            dataset_train, dataset_test = train_test_split(dataset, 
                                                            test_size=processing_configuration['test_size'], 
                                                            stratify=s_y, 
                                                            random_state=processing_configuration['seed'])

        if processing_configuration['val_data']:
            if processing_configuration['stratify_by'] in ['label', 'sensitive']:
                dataset_train, dataset_val = train_test_split(dataset_train, 
                                                                test_size=.1/(1-processing_configuration['test_size']),
                                                                stratify=dataset_train[features[processing_configuration['stratify_by']]], 
                                                                random_state=processing_configuration['seed'])
            elif processing_configuration['stratify_by'] == 'label_sensitive':
                unique_values_s = len(np.unique(dataset_train[features['sensitive']]))
                s_y = (dataset_train[features['label']]).astype('float32').values*unique_values_s + dataset_train[features['sensitive']].astype('float32').values
                dataset_train, dataset_val = train_test_split(dataset_train, 
                                                                test_size=.1/(1-processing_configuration['test_size']), 
                                                                stratify=s_y, 
                                                                random_state=processing_configuration['seed'])
        else:
            dataset_val = None
        
    else:
        dataset_train = dataset.loc[(split.values.reshape(-1)=='train'),:]
        dataset_val = dataset.loc[(split.values.reshape(-1)=='val'),:]
        dataset_test = dataset.loc[(split.values.reshape(-1)=='test'),:]

    return dataset_train, dataset_val, dataset_test, features