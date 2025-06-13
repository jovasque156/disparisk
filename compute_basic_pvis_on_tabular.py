# Import basics
import os
import argparse
import logging
import json
from pandas import read_csv as pd_read_csv
from pandas import DataFrame

# Import data modules
from torch.utils.data import DataLoader
from src.datasets.classes.TabularDataSet import TabularDataSet

# Import model modules
from src.models.base_models.FNN import FNN
from configs.config_classifiers import config_FNN

# Import utils
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from transformers import set_seed
from torch.nn import Softmax
from torch import no_grad as torch_no_grad
from torch import log2 as torch_log2
from torch import arange as torch_arange
from torch import load as torch_load


# Setting logger
extra = {'app_name':__name__}
logger = logging.getLogger(__name__)
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(app_name)s - %(levelname)s : %(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(syslog)
logger = logging.LoggerAdapter(logger, extra)

DEVICE = 'cuda' if cuda_is_available else 'cpu'
BASE_DATASET_PATH = '/path/to/data_sets'
BASE_RESULTS_PATH = '/path/to/results/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing pvis')

    parser.add_argument('--dataset_name', type=str, default='census_income_kdd', help='Name of dataset in [census_income_kdd]')
    parser.add_argument('--label', type=str, help='Name of the feature set as label. If not given, default configuration will be set. Check load_dataset.')
    parser.add_argument('--sensitive', type=str, help='Name of the feature set as sensitive. If not given, default configuration will be set. Check load_dataset.')
    parser.add_argument('--split', type=str, default='train_val_test', help='Split to use', choices=['train_val_test', 'train_test'])
    parser.add_argument('--stratify_by', type=str, default='label', help='Stratification used for training')
    parser.add_argument('--scaler', type=str, default='scaler', help='Scaler used for training')
    
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--set', type=str, default='test', help='Set to use for computation', choices=['train', 'test', 'val'])

    parser.add_argument('--model_config', type=str, default='h1', help='Id of the configuration to load. Default is set to h1')

    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training. Default value is 12345")

    args = parser.parse_args()

    path_to_dataset = os.path.join(BASE_DATASET_PATH, args.dataset_name, 'raw/preprocessed', f'label_{args.label}_sensitive_{args.sensitive}', 
                                args.split, f'stratify_by_{args.stratify_by}', args.scaler)

    if args.seed is not None:
        set_seed(args.seed)

    logger.info(f'Loading {args.set}-null-version dataset')
    with open(os.path.join(path_to_dataset, 'features.json'), 'r') as file:
        features = json.load(file)
    ds = pd_read_csv(os.path.join(path_to_dataset, f'{args.set}_null.csv'))

    tab_tds = TabularDataSet(
                            tabular_df = ds,
                            features = features
                            )

    ds_loader = DataLoader(tab_tds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = DataFrame()
    results['id'] = ds['Unnamed: 0'] if 'Unnamed: 0' in ds.columns else ds.iloc[:,0]
    results[f'sensitive_{args.sensitive}'] = ds[features['sensitive']]
    results[f'label_{args.label}'] = ds[features['label']]
    
    for set_ in ['null', 'std']:
        epochs = 2 if set_ == 'null' else 6

        if set_ =='std':
            logger.info(f'Loading {args.set}-std-version dataset')
            ds = pd_read_csv(os.path.join(path_to_dataset, f'{args.set}.csv'))

            tab_tds = TabularDataSet(
                                    tabular_df = ds,
                                    features = features
                                    )

            ds_loader = DataLoader(tab_tds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        for epoch in range(1,epochs):
            logger.info(f'Loading model from checkpoint {epoch}')
            
            n_features = len(features['input_space'])
            model = FNN(
                            input_size = n_features,
                            hidden_layers=config_FNN[args.model_config]['n_hidden_layers']*[n_features],
                            output_size=features['output_dim'],
                            config_model={
                                            'activation_hidden': config_FNN[args.model_config]['activation_hidden'],
                                            'activation_output': config_FNN[args.model_config]['activation_output'],
                                            'p_dropout': config_FNN[args.model_config]['p_dropout'],
                                            'norm': config_FNN[args.model_config]['norm']
                                    },
            )

            path_to_model = os.path.join(BASE_RESULTS_PATH, args.dataset_name, args.model_config, f'{args.label}_{set_}')
    
            checkpoint = torch_load(os.path.join(path_to_model, f'checkpoint_epoch_{epoch}.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            
            model.to(DEVICE)
            
            model.eval()
            distr = Softmax(dim=1)
            sensitives = []
            labels = []
            predictions = []
            H_y = []
            
            for batch in ds_loader:
                X, label = batch['X'].to(DEVICE), batch['label'].to(DEVICE)
                
                epsilon = 1e-3
                with torch_no_grad():
                    outputs = distr(model(X)).clamp(min=epsilon, max=1-epsilon)
                
                predictions += outputs.argmax(dim=1).cpu().tolist()
                sensitives += batch['sensitive'].cpu().numpy().astype(int).tolist()
                labels += label.cpu().numpy().astype(int).tolist()
                H_y += (-1*torch_log2(outputs[torch_arange(outputs.size(0)), label])).cpu().tolist()

            empty_cache()

            results[f'Hy_{set_}_{args.label}_epoch{epoch}'] = H_y
            results[f'prediction_{args.label}_epoch{epoch}'] = predictions
            name_file = f'pvis_on_{args.set}.csv'
            results.to_csv(os.path.join(path_to_model, name_file))
        logger.info(f'Saving results at {os.path.join(path_to_model, name_file)}')
            