import os
import json
import argparse
import logging

import src.datasets.loading.tabular_datasets as tab_ds
import src.datasets.loading.image_datasets as img_ds
import src.datasets.loading.text_datasets as txt_ds

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

NULL_PHOTO_NAME = {
            'vit': 'null_image_ViT.jpg',
            'not_vit': 'null_image.jpg'
        }

if __name__ == "__main__":
    logger.info('Parsing arguments')
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='census_income_kdd', help='Dataset name', choices=['census_income_kdd','facet', 'hate_speech'])
    parser.add_argument('--path_to_dataset', type=str, help='Path to the folder containing the dataset')
    parser.add_argument('--label', type=str, help='Column name used as label in the dataset')
    parser.add_argument('--sensitive', type=str, help='Column name used as sensitive in the dataset')

    parser.add_argument('--random_split', action='store_true', help='Use to disable default splitting and create a new one randomly')
    parser.add_argument('--no_val_set', action='store_true', help='Use to disable the generation of val set')
    parser.add_argument('--val_size', type=float, default=0.1, help='Size --in proportion of the train set size- of the val set')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set')
    parser.add_argument('--stratify_by', type=str, default='label', help='Column name for using during splitting', choices=['label', 'sensitive', 'label_sensitive'])
    parser.add_argument('--scaler', type=str, default='standard', help='Set the scaler to preprocess dataset')
    parser.add_argument('--seed', type=int, default=12345, help='number for seeding')
    
    args = parser.parse_args()

    # Set processing configurations
    logger.info('Setting processing configurations')
    processing_configuration = {
                        'stratify_by': args.stratify_by,
                        'scaler': args.scaler,
                        'test_size': args.test_size,
                        'val_data': not args.no_val_set, 
                        'seed': args.seed
                    }

    logger.info(f'Starting processing of {args.dataset_name} dataset')
    
    if args.dataset_name == 'census_income_kdd':
        tab_ds._CENSUS_INCOME_KDD_PATH = args.path_to_dataset if args.path_to_dataset else tab_ds._CENSUS_INCOME_KDD_PATH

        # Load dataset
        logger.info(f'Loading dataset with label {args.label} and sensitive {args.sensitive}')
        dataset, features, split = tab_ds.load_dataset(
                                                ds_name = args.dataset_name,
                                                label = args.label,
                                                sensitive = args.sensitive,
                                                random_split = args.random_split,
                                                processing_configuration=processing_configuration
                                                )
        # Split and process dataset
        logger.info(f'Splitting and processing dataset of size {dataset.shape}')
        train, val, test, features = tab_ds.process_dataset(
                                                dataset=dataset,
                                                features=features,
                                                split = None if args.random_split else split,
                                                processing_configuration=processing_configuration
                                            )
        logger.info(f'Done. Result, train size: {train.shape}, val size: {val.shape if processing_configuration["val_data"] else 0}, test size: {test.shape}')
        
        # Generate null versions
        logger.info(f'Generating null versions')
        def set_to_null(df, columns):
                df_copy = df.copy()
                df_copy.loc[:, columns] = 0
                return df_copy
        train_null = set_to_null(train, features['input_space'])
        test_null = set_to_null(test, features['input_space'])
        if processing_configuration["val_data"]:
            val_null = set_to_null(val, features['input_space'])

        # Create tuples of datasets for saving
        sets = [
            ('train', train), ('train_null', train_null),
            ('test', test), ('test_null', test_null)
        ]
        if processing_configuration['val_data']:
            sets.append(('val', val))
            sets.append(('val_null', val_null))

        # Create folder if doesn't exist
        logger.info(f'Saving datasets')
        splits = 'train_val_test' if processing_configuration['val_data'] else 'train_test'
        stratify_by = processing_configuration['stratify_by']
        scaler = 'standard' if processing_configuration['scaler']=='standard' else ''
        path = os.path.join(tab_ds._CENSUS_INCOME_KDD_PATH, 'preprocessed', f'label_{args.label}_sensitive_{args.sensitive}' , splits, f'stratify_by_{stratify_by}', scaler)
        os.makedirs(path, exist_ok=True)
        
        # Save resulting files
        for name, set_ in sets:
            logger.info(f'> Saving {name} of size {set_.shape}')
            set_.to_csv(os.path.join(path, f'{name}.csv'))
        
        logger.info(f'> Saving features information')
        with open(os.path.join(path, 'features.json'), 'w') as f:
            json.dump(features, f)
        
    elif args.dataset_name == 'facet':
        img_ds._FACET_PATH = args.path_to_dataset if args.path_to_dataset else img_ds._FACET_PATH

        # Load dataset
        logger.info(f'Loading dataset with label {args.label} and sensitive {args.sensitive}')
        dataset, features, split = img_ds.load_dataset(
                                                ds_name=args.dataset_name,
                                                label=args.label,
                                                sensitive=args.sensitive,
                                                random_split=args.random_split,
                                                processing_configuration=processing_configuration
                                                )

        # Split and process dataset
        logger.info(f'Splitting dataset of size {dataset.shape}')
        train, val, test, features = img_ds.process_dataset(
                                                dataset=dataset,
                                                features=features,
                                                split = None if args.random_split else split,
                                                processing_configuration=processing_configuration
                                            )
        logger.info(f'Done. Result, train size: {train.shape}, val size: {val.shape if processing_configuration["val_data"] else 0}, test size: {test.shape}')

        # Generate null versions
        logger.info(f'Generating null versions')
        def set_to_null(df, columns, photo_name):
                df_copy = df.copy()
                df_copy.loc[:, columns] = photo_name
                return df_copy

        train_null_all = set_to_null(train, features['photo_name'], NULL_PHOTO_NAME['not_vit'])
        train_null_vit = set_to_null(train, features['photo_name'], NULL_PHOTO_NAME['vit'])
        test_null_all = set_to_null(test, features['photo_name'], NULL_PHOTO_NAME['not_vit'])
        test_null_vit = set_to_null(test, features['photo_name'], NULL_PHOTO_NAME['vit'])
        if processing_configuration['val_data']:
            val_null_all = set_to_null(val, features['photo_name'], NULL_PHOTO_NAME['not_vit'])
            val_null_vit = set_to_null(val, features['photo_name'], NULL_PHOTO_NAME['vit'])
        
        # Create tuples of datasets for saving
        logger.info(f'Saving datasets')
        sets = [
            ('train', train), ('train_null', train_null_all), ('train_null_vit', train_null_vit),
            ('test', test), ('test_null', test_null_all), ('test_null_vit', test_null_vit)
        ]
        if processing_configuration['val_data']:
            sets.append(('val', val))
            sets.append(('val_null', val_null_all))
            sets.append(('val_null_vit', val_null_vit))
        
        # Create folder if doesn't exist
        splits = 'train_val_test' if processing_configuration['val_data'] else 'train_test'
        stratify_by = processing_configuration['stratify_by']
        path = os.path.join(img_ds._FACET_PATH, 'preprocessed', f'label_{args.label}_sensitive_{args.sensitive}' , splits, f'stratify_by_{stratify_by}')
        os.makedirs(path, exist_ok=True)

        # Save resulting files
        for name, set_ in sets:
            logger.info(f'> Saving {name} of size {set_.shape}')
            set_.to_csv(os.path.join(path, f'{name}.csv'))

        logger.info(f'> Saving features information')
        with open(os.path.join(path, 'features.json'), 'w') as f:
            json.dump(features, f)
    
    elif args.dataset_name == 'hate_speech':
        txt_ds._HATE_SPEECH_PATH = args.path_to_dataset if args.path_to_dataset else txt_ds._HATE_SPEECH_PATH

        # Load dataset
        logger.info(f'Loading dataset with label {args.label} and sensitive {args.sensitive}')
        dataset, features, split = txt_ds.load_dataset(
                                                ds_name=args.dataset_name,
                                                label=args.label,
                                                sensitive=args.sensitive,
                                                random_split=args.random_split,
                                                processing_configuration=processing_configuration
                                                )

        # Split and process dataset
        logger.info(f'Splitting dataset of size {dataset.shape}')
        train, val, test, features = txt_ds.process_dataset(
                                                dataset=dataset,
                                                features=features,
                                                split = None if args.random_split else split,
                                                processing_configuration=processing_configuration
                                            )
        logger.info(f'Done. Result, train size: {train.shape}, val size: {val.shape if processing_configuration["val_data"] else 0}, test size: {test.shape}')

        # Generate null versions
        def set_to_null(df, columns):
                df_copy = df.copy()
                df_copy.loc[:, columns] = ''
                return df_copy

        train_null = set_to_null(train, features['text'])
        test_null = set_to_null(test, features['text'])
        if processing_configuration['val_data']:
            val_null = set_to_null(val, features['text'])

        # Create tuples of datasets for saving
        sets = [
            ('train', train), ('train_null', train_null),
            ('test', test), ('test_null', test_null)
        ]
        if processing_configuration['val_data']:
            sets.append(('val', val))
            sets.append(('val_null', val_null))

        # Create folder if doesn't exist
        logger.info(f'Saving datasets')
        splits = 'train_val_test' if processing_configuration['val_data'] else 'train_test'
        stratify_by = processing_configuration['stratify_by']
        path = os.path.join(txt_ds._HATE_SPEECH_PATH, 'preprocessed', f'label_{args.label}_sensitive_{args.sensitive}' , splits, f'stratify_by_{stratify_by}')
        os.makedirs(path, exist_ok=True)

        # Save resulting files
        for name, set_ in sets:
            logger.info(f'> Saving {name} of size {set_.shape}')
            set_.to_csv(os.path.join(path, f'{name}.csv'))
        
        logger.info(f'> Saving features information')
        with open(os.path.join(path, 'features.json'), 'w') as f:
            json.dump(features, f)