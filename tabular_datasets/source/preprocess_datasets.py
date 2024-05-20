import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

#Pipelines
import sys
sys.path.insert(1, '../')
from source.utils import apply_preprocessing, train_preprocessing

DIR_DATA = {
    'census_income':'../data/census_income/',
    'compas': '../data/compas/',
    'dutch_census': '../data/dutch_census/',
    'credit_card': '../data/credit_card/'
    }

#for Dutch, check this: https://arxiv.org/pdf/2110.00530.pdf

DIR_DATA_TRAIN = {
        'census_income':'adult.data',
        'compas': 'compas_train.csv',
        'dutch_census': 'dutch_census_train.csv',
        'credit_card': 'credit_card.csv'
        # 'german_data': 'german_data.csv'
    }

DIR_DATA_TEST = {
        'census_income':'adult.test',
        'compas': 'compas_test.csv',
        'dutch_census': 'dutch_census_test.csv',
        'credit_card': None
    }

FEATURES = {
    'census_income': ['age',
                    'workclass',
                    'education', 
                    'education-num', 
                    'marital-status', 
                    'occupation', 
                    'relationship', 
                    'race',
                    'sex', 
                    'capital-gain', 
                    'capital-loss', 
                    'hours-per-week',
                    'native-country'],
    'compas': ['sex',
            'age',
            'age_cat',
            'race',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
            'priors_count',
            'c_days_jail',
            'c_charge_degree'],
    'dutch_census': ['sex', 
                    'age',
                    'household_position',
                    'household_size',
                    'prev_residence_place',
                    'citizenship',
                    'country_birth',
                    'edu_level',
                    'economic_status',
                    'cur_eco_activity',
                    'Marital_status'],
    'credit_card': [
        'LIMIT_BAL',
        'SEX',
        'EDUCATION',
        'MARRIAGE',
        'AGE',
        'PAY_0',
        'PAY_2',
        'PAY_3',
        'PAY_4',
        'PAY_5',
        'PAY_6',
        'BILL_AMT1',
        'BILL_AMT2',
        'BILL_AMT3',
        'BILL_AMT4',
        'BILL_AMT5',
        'BILL_AMT6',
        'PAY_AMT1',
        'PAY_AMT2',
        'PAY_AMT3',
        'PAY_AMT4',
        'PAY_AMT5',
        'PAY_AMT6',
    ]
}

NOMINAL = {
    'census_income': ['workclass', 
                    'education', 
                    'marital-status', 
                    'occupation', 
                    'relationship', 
                    'race', 
                    'native-country'],
    'dutch_census': ['sex',
                    'age',
                    'household_position',
                    'household_size',
                    'prev_residence_place',
                    'citizenship',
                    'country_birth',
                    'edu_level',
                    'economic_status',
                    'cur_eco_activity',
                    'Marital_status'],
    'compas': ['age_cat',
                'sex',
                'race',
                'c_charge_degree'],
    'credit_card': [
        'SEX',
        'EDUCATION',
        'MARRIAGE',
        'PAY_0',
        'PAY_2',
        'PAY_3',
        'PAY_4',
        'PAY_5',
        'PAY_6',
    ]
    }


#The first is the name of the attribute, and the second is the groups
#The list of the groups should start with the protected group.
SENSITIVE_ATTRIBUTE = {
    'census_income': {'sex': ['Female', 'Male']}, # majoirty Male
    'compas': {'race': ['African-American', 'Caucasian', 'Hispanic', 'Native American', 'Other']}, #Majority the others
    'dutch_census': {'sex': ['Female', 'Male']}, # majoirty Male
    'credit_card': {'SEX': [2, 1]} # majoirty 2
    
    }

LABEL = {
    'census_income': {'income': ['>50K', '<=50K']},
    'compas': {'two_year_recid': [1, 0], 'is_recid': [1, 0]},
    'dutch_census': {'occupation': ['high_level', 'low-level']},
    'credit_card': {'default_payment': [1, 0]}
}

def preprocess_datasets(args):
    '''
    Preprocess the datasets and save them in the datasets/ folder

    Output:
        - X_train: sparse matrix, representing the features
        - S_train: numpy, representing the sensitive attribute. Assuming binary
        - Y_train: numpy, representing the label.
    '''
    # Load the data
    args.target_variable = args.target_variable if args.target_variable else [k for k in LABEL[args.dataset]][0]
    args.sensitive_attribute = args.sensitive_attribute if args.sensitive_attribute else [k for k in SENSITIVE_ATTRIBUTE[args.dataset]][0]
    
    if None in [DIR_DATA_TEST[args.dataset], DIR_DATA_TRAIN[args.dataset]]:
        df = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TRAIN[args.dataset])    
        df_train, df_test = train_test_split(df, test_size=args.test_size, stratify=df[args.sensitive_attribute])
        df_train.to_csv(DIR_DATA[args.dataset]+args.dataset+'_train.csv')
        df_test.to_csv(DIR_DATA[args.dataset]+args.dataset+'_test.csv')
    else:
        df_train = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TRAIN[args.dataset])
        df_test = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TEST[args.dataset])

    # Check target and sensitive
    args.target_variable = args.target_variable if args.target_variable else [k for k in LABEL[args.dataset]][0]
    args.sensitive_attribute = args.sensitive_attribute if args.sensitive_attribute else [k for k in SENSITIVE_ATTRIBUTE[args.dataset]][0]

    features = FEATURES[args.dataset]
    features.remove(args.sensitive_attribute)

    # Retrieve variables
    Y_train = df_train.loc[:, [args.target_variable]].to_numpy().flatten()
    Y_train = 1*(Y_train == LABEL[args.dataset][args.target_variable][0])
    S_train = df_train.loc[:, [args.sensitive_attribute]].to_numpy().flatten()
    if args.binarize_sensitive:
        S_train = 1*(S_train == SENSITIVE_ATTRIBUTE[args.dataset][args.sensitive_attribute][0])
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(S_train)
        S_train = label_encoder.transform(S_train)
    X_train = df_train.loc[:, features]
    
    Y_test = df_test.loc[:, [args.target_variable]].to_numpy().flatten()
    Y_test = 1*(Y_test == LABEL[args.dataset][args.target_variable][0])
    S_test = df_test.loc[:, [args.sensitive_attribute]].to_numpy().flatten()
    if args.binarize_sensitive:
        S_test = 1*(S_test == SENSITIVE_ATTRIBUTE[args.dataset][args.sensitive_attribute][0])
    else:
        S_test = label_encoder.transform(S_test)
    X_test = df_test.loc[:, features]

    # Get nominal names of features and remove the sensitive attribute
    nominal_names = NOMINAL[args.dataset]
    nominal_names.remove(args.sensitive_attribute)

    # Get id_numerical
    id_numerical = [i 
                    for i, f in enumerate(X_train.columns)
                    if f not in nominal_names]

    # Encode the categorical features
    (outcome) = train_preprocessing(X_train, 
                                    idnumerical=id_numerical, 
                                    imputation=args.not_imputation, 
                                    encode=args.nominal_encode, 
                                    standardscale=args.standardscale,
                                    normalize = args.normalize)
    
    X_train, pipe_num, pipe_nom, pipe_normalize, numerical_features, nominal_features = outcome
    
    X_test = apply_preprocessing(X_test, 
                                pipe_nom, 
                                pipe_num, 
                                pipe_normalize, 
                                idnumerical=id_numerical)

    result = {
            'train': (X_train, S_train, Y_train),
            'test': (X_test, S_test, Y_test),
            'pipes': (pipe_nom, pipe_num, pipe_normalize),
            'features': (numerical_features, nominal_features),
            }

    with open(DIR_DATA[args.dataset]+args.dataset+'.pkl', 'wb') as f:
        pickle.dump(result, f, protocol = pickle.HIGHEST_PROTOCOL)
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='census_income', help='Dataset to preprocess')
    parser.add_argument('--test_size', type=float, default=0.3, help='Size of test if is not defined')
    parser.add_argument('--sensitive_attribute', type=str, default='', help='features used as sensitive attribute')
    parser.add_argument('--binarize_sensitive', action="store_true", help='binarize sensitive attribute by considering 1 the first class and 0 the rest')
    parser.add_argument('--target_variable', type=str, default='', help='target variable')
    parser.add_argument('--nominal_encode', type=str, default='label', help='Type of encoding for nominal features')
    parser.add_argument('--standardscale', action="store_true", help='Apply standard scale transformation')
    parser.add_argument('--normalize', action="store_true", help='Apply normalization transformation')
    parser.add_argument('--not_imputation', action="store_false", help='Set false to not apply imputation on missing values')
    
    parser.add_argument('--target_multi_class', action='store_true', help='target variable as multi-class')
    parser.add_argument('--sa_multi_class', action='store_true', help='sensitive attribute as multi-class')
    
    args = parser.parse_args()

    print(f'Preprocessing {args.dataset} dataset...')
    preprocess_datasets(args)
    print('Done!')