import os
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

_CENSUS_INCOME_KDD_PATH = '/path/to/census_income_kdd/raw/'
# Setting logger
extra = {'app_name':__name__}

# Gets or creates a logger
logger = logging.getLogger(__name__)

# stream handler and formatter
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
                scaler: 'standard', 
                null_version: False,
                test_size: 0.2, # assume val_size=test_size
                val_data: True, 
                seed: 12345
            }
    '''
    if not random_split:
        logger.warning('given random_split is False')
    if random_split or ds_name=='acs_income':
        if processing_configuration is None:
            logger.info("random_split=True or ds_name='acs_income', but no processing_configuration was given. Loading the standard configuration")
            processing_configuration = {
                                'stratify_by': 'label',
                                'scaler': 'standard',
                                'test_size': 0.2, # assume val_size=test_size (of the entire dataset)
                                'val_data': True, 
                                'seed': 12345
                            }
        assert all([key in processing_configuration for key in ['stratify_by', 'scaler', 'test_size','val_data','seed']]), 'processing_configuration with all the keys needed [stratify_by, scaler, test_size, val_data, seed]'
        # if not processing_configuration['null_version']:
        #     assert 'scaler' in processing_configuration.keys(), 'null_version set False, but no scaler is given in processing_configuration'
    
    elif ds_name=='census_income_kdd':
        if processing_configuration is None:
            logger.info('A processing_configuration is not given. Loading the standard configuration with only scaler=standard,val_data=True,null_version=False,seed=12345')
            processing_configuration = {
                                'scaler': 'standard',
                                "null_version": False,
                                'val_data': True,
                                'seed': 12345
                            }

    dataset, features, split = load_census_income_kdd_data(label=label, sensitive=sensitive)
    split = split if not random_split else None
    
    return dataset, features, split

def load_census_income_kdd_data(path=_CENSUS_INCOME_KDD_PATH, label= 'class', sensitive="sex"):
    label = 'class' if label is None else label
    sensitive = 'sex' if sensitive is None else sensitive

    column_names = ["age","workclass","industry_code","occupation_code","education","wage_per_hour","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","race","hispanic_origin","sex","member_of_a_labour_union","reason_for_unemployment",
    "employment_status","capital_gains","capital_losses","dividend_from_stocks","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","num_persons_worked_for_employer","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","weeks_worked_in_year","year","class"]

    categorical_features = [
    "workclass","industry_code","occupation_code","education","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","race", "hispanic_origin","sex", "member_of_a_labour_union","reason_for_unemployment",
    "employment_status","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","year",'class'
    ]

    X = [col for col in column_names if col not in [label, sensitive]]
    
    df1 = pd.read_csv(os.path.join(path, "census-income.data"),header=None,names=column_names)
    df2 = pd.read_csv(os.path.join(path, "census-income.test"),header=None,names=column_names)

    # Create validation split
    df1['split'] = 'train'
    df2['split'] = 'test'
    test_porc = df2.shape[0]/(df1.shape[0]+df2.shape[0])
    num_validation_samples = int(len(df1) * 0.10/(1-test_porc)) # assume val is equal to .1 of the entire dataset
    validation_samples = df1.sample(n=num_validation_samples, random_state=12345)  # random_state for reproducibility
    df1.loc[validation_samples.index, 'split'] = 'val'

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(keep="first", inplace=False)

    if sensitive == "race":
        df = df[df["race"].isin([" White", " Black"])]
        df['race'] = df['race'].apply(lambda x: 1 if x == " Black" else 0).astype(int).to_frame()
    if sensitive == "sex":
        df['sex'] = df['sex'].apply(lambda x: 1 if x == " Female" else 0).astype(int).to_frame()
    if sensitive == 'class':
        df['class'] = df['class'].apply(lambda x: 1 if x == " - 50000." else 0).astype(int).to_frame()    

    # # labels; 1 , otherwise 0
    if label == "race":
        df = df[df["race"].isin([" White", " Black"])]
        df['race'] = df['race'].apply(lambda x: 1 if x == " Black" else 0).astype(int).to_frame()
    if label == "sex":
        df['sex'] = df['sex'].apply(lambda x: 1 if x == " Female" else 0).astype(int).to_frame()
    if label == 'class':
        df['class'] = df['class'].apply(lambda x: 1 if x == " - 50000." else 0).astype(int).to_frame()    

    # features
    if label in categorical_features:
        categorical_features.remove(label)
    if sensitive in categorical_features:
        categorical_features.remove(sensitive)
    df[categorical_features] = df[categorical_features].astype("string")

    # Convert all non-uint8 columns to float32
    string_cols = df.select_dtypes(exclude="string").columns.drop('split')
    df[string_cols] = df[string_cols].astype("float32")

    features = {
        'input_space': X if isinstance(X, list) else [X],
        'sensitive': sensitive,
        'label': label,
        'output_dim': len(np.unique(df[label]))
    }
    
    return df[features['input_space']+[features['sensitive']]+[features['label']]], features, df['split'].to_frame()

def process_dataset(dataset, features, split, processing_configuration):
    categorical_cols = dataset[features['input_space']].select_dtypes("string").columns
    if len(categorical_cols) > 0:
        dataset = pd.get_dummies(dataset, columns=categorical_cols, dtype='float32', drop_first=True)

    if split is None:
        if processing_configuration['stratify_by'] in ['label', 'sensitive']:
            dataset_train, dataset_test = train_test_split(dataset, test_size=processing_configuration['test_size'], stratify=dataset[features[processing_configuration['stratify_by']]], random_state=processing_configuration['seed'])
        elif processing_configuration['stratify_by'] == 'label_sensitive':
            unique_values_s = len(np.unique(dataset[features['sensitive']]))
            s_y = (dataset[features['label']]).astype('float32').values*unique_values_s + dataset[features['sensitive']].astype('float32').values
            dataset_train, dataset_test = train_test_split(dataset, test_size=processing_configuration['test_size'], stratify=s_y, random_state=processing_configuration['seed'])

        if processing_configuration['val_data']:
            if processing_configuration['stratify_by'] in ['label', 'sensitive']:
                dataset_train, dataset_val = train_test_split(dataset_train, test_size=.1/(1-processing_configuration['test_size']), stratify=dataset_train[features[processing_configuration['stratify_by']]], random_state=processing_configuration['seed'])
            elif processing_configuration['stratify_by'] == 'label_sensitive':
                unique_values_s = len(np.unique(dataset_train[features['sensitive']]))
                s_y = (dataset_train[features['label']]).astype('float32').values*unique_values_s + dataset_train[features['sensitive']].astype('float32').values
                dataset_train, dataset_val = train_test_split(dataset_train, test_size=.1/(1-processing_configuration['test_size']), stratify=s_y, random_state=processing_configuration['seed'])
        else:
            dataset_val = None
    else:
        dataset_train = dataset.loc[(split.values.reshape(-1)=='train'),:]
        dataset_val = dataset.loc[(split.values.reshape(-1)=='val'),:]
        dataset_test = dataset.loc[(split.values.reshape(-1)=='test'),:]

    #Update features
    features['input_space'] = [col for col in dataset_train.columns if col not in [features['label'], features['sensitive']]]

    # Identify numerical cols
    numerical_cols = dataset_train.select_dtypes("float32").columns
    
    # Drop label and sensitive in case they are in the set
    if features['label'] in numerical_cols:
        numerical_cols = numerical_cols.drop(features['label'])
    if features['sensitive'] in numerical_cols:
        numerical_cols = numerical_cols.drop(features['sensitive'])
    
    # Apply transformation
    if len(numerical_cols) > 0:
        if processing_configuration['scaler'] in ['standard', 'minmax']:
            scaler = StandardScaler().fit(dataset_train[numerical_cols]) if processing_configuration['scaler']=='standard' else MinMaxScaler().fit(dataset_train[numerical_cols])

            def scale_df(df, scaler):
                return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

            dataset_train.loc[:,numerical_cols] = dataset_train.loc[:, numerical_cols].pipe(scale_df, scaler)
            dataset_test.loc[:,numerical_cols]  = dataset_test.loc[:, numerical_cols].pipe(scale_df, scaler)
            if processing_configuration['val_data']:
                dataset_val.loc[:,numerical_cols]   = dataset_val.loc[:, numerical_cols].pipe(scale_df, scaler)
    
    return dataset_train, dataset_val, dataset_test, features