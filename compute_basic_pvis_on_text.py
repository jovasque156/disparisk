# Import basics
import os
import argparse
import logging
import json
from pandas import read_csv as pd_read_csv
from pandas import DataFrame

# Import data modules
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from src.datasets.classes.TextDataSet import TextDataSet

# Import model modules
from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        set_seed
)

# Import utils
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from transformers import set_seed
from torch.nn import Softmax
from torch import no_grad as torch_no_grad
from torch import log2 as torch_log2
from torch import arange as torch_arange
from torch import load as torch_load

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

DICT_LABEL_HATE_SPEECH = {
        'hate_speech':0,
        'offensive_language':1,
        'neither':2
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing pvis')

    # Setting dataset config
    parser.add_argument('--dataset_name', type=str, default='hate_speech', help='Name of dataset')
    parser.add_argument('--label', type=str, help='Name of the feature set as label. If not given, default configuration will be set. Check load_dataset.')
    parser.add_argument('--sensitive', type=str, help='Name of the feature set as sensitive. If not given, default configuration will be set. Check load_dataset.')
    parser.add_argument('--split', type=str, default='train_val_test', help='Split to use', choices=['train_val_test', 'train_test'])
    parser.add_argument('--stratify_by', type=str, default='label', help='Stratification used for training')
    parser.add_argument('--binary', action='store_true', help='Use to set target as binary')
    parser.add_argument('--class_to_binarize', type=str, default="lawman", help='Name of the class in label to binarize')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--set', type=str, default='test', help='Set to use for computation', choices=['train', 'test', 'val'])

    # Setting model config
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='Id of the configuration to load. Default is set to h1')
    parser.add_argument('--pretrained', action='store_true', help='Use to load a pretrained model')

    # Setting additional configs
    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training. Default value is 12345")

    args = parser.parse_args()

    # Set some basics
    path_to_dataset = os.path.join(BASE_DATASET_PATH, args.dataset_name, 'preprocessed', f'label_{args.label}_sensitive_{args.sensitive}', 
                                args.split, f'stratify_by_{args.stratify_by}')

    if args.seed is not None:
        set_seed(args.seed)

    logger.info(f'Loading {args.set}-null-version dataset')
    with open(os.path.join(path_to_dataset, 'features.json'), 'r') as file:
        features = json.load(file)
        if args.binary:
            features["output_dim"]=2
    ds = pd_read_csv(os.path.join(path_to_dataset, f'{args.set}_null.csv'))
    ds[features['text']] = ""
    
    logger.info('Creating dataloaders with tokenizers')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(args.model_name, num_labels=features['output_dim'], pad_token_id=tokenizer.eos_token_id) # ADDED
    else:
        config = AutoConfig.from_pretrained(args.model_name, num_labels=features['output_dim'])

    txt_ds = TextDataSet(text_df=ds, features=features, tokenizer=tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    
    ds_loader = DataLoader(txt_ds, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    
    results = DataFrame()
    results['id'] = ds['Unnamed: 0'] if 'Unnamed: 0' in ds.columns else ds.iloc[:,0]
    results[f'sensitive_{args.sensitive}'] = ds[features['sensitive']]
    results[f'label_{args.label}'] = ds[features['label']]
    
    for set_ in ['null', 'std']:
        epochs = 2 if set_ == 'null' else 6

        if args.binary:
            path_to_model = os.path.join(BASE_RESULTS_PATH, args.dataset_name, args.model_name, f'{args.label}_{args.class_to_binarize}_{set_}')
        else:
            path_to_model = os.path.join(BASE_RESULTS_PATH, args.dataset_name, args.model_name, f'{args.label}_{set_}')

        if set_ =='std':
            logger.info(f'Loading {args.set}-std-version dataset')
            with open(os.path.join(path_to_dataset, 'features.json'), 'r') as file:
                features = json.load(file)
                if args.binary:
                    features['output_dim'] = 2
            ds = pd_read_csv(os.path.join(path_to_dataset, f'{args.set}.csv'))
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)

            if tokenizer.pad_token is None:
                if tokenizer.eos_token is None:
                    tokenizer.eos_token = "<|endoftext|>"
                tokenizer.pad_token = tokenizer.eos_token
                config = AutoConfig.from_pretrained(args.model_name, num_labels=features['output_dim'], pad_token_id=tokenizer.eos_token_id) # ADDED
            else:
                config = AutoConfig.from_pretrained(args.model_name, num_labels=features['output_dim'])

            txt_ds = TextDataSet(text_df=ds, features=features, tokenizer=tokenizer)

            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
            
            ds_loader = DataLoader(txt_ds, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)

            model = AutoModelForSequenceClassification.from_pretrained(
                                                args.model_name,
                                                from_tf=bool(".ckpt" in args.model_name),
                                                config=config,
                                            )

        for epoch in range(1,epochs):
            if not os.path.isfile(os.path.join(path_to_model, f'checkpoint_epoch_{epoch}.pt')):
                logger.info(f"Model at {set_} and epoch {epoch} doesn't exist. Skipping computation at this set")
                continue

            checkpoint = torch_load(os.path.join(path_to_model, f'checkpoint_epoch_{epoch}.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if features['label_dic'] is not None:
                model.config.label2id = features['label_dic']
                model.config.id2label = {id: label for label, id in config.label2id.items()}

            model.to(DEVICE)
            
            model.eval()
            distr = Softmax(dim=1)
            sensitives = []
            labels = []
            predictions = []
            H_y = []
            for batch in ds_loader:
                prepare_batch = {k:v.to(DEVICE) for k,v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                if args.binary:
                    prepare_batch["labels"] = (prepare_batch["labels"]==DICT_LABEL_HATE_SPEECH[args.class_to_binarize]).long()
                
                epsilon = 1e-3
                with torch_no_grad():
                    outputs = model(**prepare_batch)
                    _, outputs = outputs[:2]
                    outputs = distr(outputs).clamp(min=epsilon, max=1-epsilon)

                predictions += outputs.argmax(dim=1).cpu().tolist()
                H_y += (-1*torch_log2(outputs[torch_arange(outputs.size(0)), prepare_batch['labels']])).cpu().tolist()
                
            empty_cache()
            results[f'Hy_{set_}_{args.label}_epoch{epoch}'] = H_y
            if set_=='null':
                results[f'prediction_{args.label}_null'] = predictions    
            results[f'prediction_{args.label}_epoch{epoch}'] = predictions
            name_file = f'pvis_on_{args.set}.csv'
            results.to_csv(os.path.join(path_to_model, name_file))
        if os.path.isfile(os.path.join(path_to_model, f'checkpoint_epoch_{epoch}.pt')):
            logger.info(f'Saving results at {os.path.join(path_to_model, name_file)}')
            