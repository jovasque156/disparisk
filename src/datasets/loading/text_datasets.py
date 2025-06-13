import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_HATE_SPEECH_PATH = '/path/to/hate_speech/'

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

def load_dataset(ds_name, label=None, sensitive=None, random_split=False, processing_configuration=None):
    '''
    ds_name: str, with the values of celeba or facet
    label: str, name of the column in the datasets to use it as label
    sensitive: str, name of the column in the datasets to use it as sensitive
    processing_configuration: dict, with the following structure (value given are defaults)
        {
            stratify_by: 'label',
            test_size: 0.2, # assume val_size=.2 of the entire dataset
            val_data: True, 
            null_version: False,
            seed: 12345
        }
    '''
    if not random_split:
        logger.warning('given random_split is False')
    if random_split or ds_name=='hate_speech':
        if processing_configuration is None:
            logger.info("random_split=True or ds_name='hate_speech', but no processing_configuration was given. Loading the standard configuration")
            processing_configuration = {
                            'stratify_by': 'label',
                            'test_size': 0.2, 
                            'val_data': True,
                            'null_version': False,
                            'seed': 12345
                        }
        assert all([key in processing_configuration for key in ['stratify_by', 'test_size', 'val_data', 'seed']]), 'processing_configuration with all the keys needed [stratify_by, test_size, val_data, seed]'

    
    dataset, features, split = load_hate_speech(label, sensitive)
    split = split if not random_split else None

    return dataset, features, split

def load_hate_speech(label, sensitive, path=_HATE_SPEECH_PATH):
    '''
    label: str, name of the column used as label
    sensitive: str, name of the column used as sensitive
    path: str, path to the hate_speech dataset, it must contain the labeled_data_AAE.csv file
    '''

    sensitive = 'dialect_class' if sensitive is None else sensitive
    label = 'class_label' if label is None else label

    label_dic = {
        'hate_speech': 0,
        'offensive_language': 1,
        'neither': 2
    }

    sensitive_dic = {
        'aa': 0,
        'hispanic': 1,
        'other': 2,
        'white': 3,
    }

    # Load dataset and required features
    dataset = pd.read_csv(os.path.join(_HATE_SPEECH_PATH, 'labeled_data_AAE.csv'))
    dataset = dataset.loc[:,['tweet', sensitive, label]]

    # Create features dict
    features = {
        'text': 'tweet',
        'sensitive': sensitive,
        'sensitive_dic': sensitive_dic if sensitive=='dialect_class' else label_dic,
        'label': label,
        'label_dic': label_dic if label=='class_label' else sensitive_dic,
        'output_dim': len(label_dic) if label=='class_label' else len(sensitive_dic)
    }

    return dataset, features, None

def process_dataset(dataset, features, split, processing_configuration):
    '''
    Prepare dataset to be loaded to TextDataSet class
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
        dataset_train = dataset.loc[split.values.reshape(-1)=='train',:]
        dataset_val = dataset.loc[split.values.reshape(-1)=='val',:]
        dataset_test = dataset.loc[split.values.reshape(-1)=='test',:]
    
    return dataset_train, dataset_val, dataset_test, features

