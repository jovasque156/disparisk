# Import basics
import argparse
import logging
import math
import os
import json
from pandas import read_csv as pd_read_csv
from pandas import DataFrame as pd_DataFrame

# Setting logger
extra = {'app_name':__name__}
logger = logging.getLogger(__name__)
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(app_name)s - %(levelname)s : %(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(syslog)
logger = logging.LoggerAdapter(logger, extra)

# Import data modules
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from src.datasets.classes.TextDataSet import TextDataSet

# Import model modules
from torch.hub import load as model_load

# Import optimizers
from torch.optim import (
        AdamW, 
        SGD
        )
from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        SchedulerType,
        get_scheduler,
        set_seed
        )

# Import torch utils
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from torch import backends
from accelerate import Accelerator

DEVICE = 'cuda' if cuda_is_available else 'cpu'

BASE_PATH = '/path/to/results/'

DICT_LABEL_HATE_SPEECH = {
        'hate_speech':0,
        'offensive_language':1,
        'neither':2
    }

def get_optimizer(opt):
    optimizers = {
        'AdamW': AdamW,
    }
    return optimizers.get(opt, SGD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DispaRisk Experiments")

    # Setting for dataset
    parser.add_argument('--path_to_dataset', type=str, help='Path to the folder containing the features.json file and the csv with texts, sensitive, and labels')
    parser.add_argument('--use_null_version', action='store_true', help='Use to use null version of train set')
    parser.add_argument('--path_to_save_results', type=str, help='Path to folder where results will be saved')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchsize for training. If not given, it is set to 32.') 
    parser.add_argument('--binary', action='store_true', help='Use to set target as binary')
    parser.add_argument('--class_to_binarize', type=str, default="lawman", help='Name of the class in label to binarize')
    
    # Selecting model configuration
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Name of the model to load choices: [bert-base-cased, distilbert-base-uncased, roberta-large, gpt2, facebook-bart-large]. Default is set to alexnet')
    parser.add_argument('--pretrained', action='store_true', help='Use to load a pretrained model')

    # Setting Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Name of the optimizer to use. Deafult is set to AdamW')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay to use.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Total number of training epochs to perform.')
    parser.add_argument('--max_train_steps', type=int, default=None, help='Total number of training steps to perform. If provided, overrides num_train_epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--scheduler_type', type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training. Default value is 12345")

    # Load arguments
    args = parser.parse_args()

    os.makedirs(args.path_to_save_results, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16")

    # Loading datasets
    name_train = 'train_null.csv' if args.use_null_version else 'train.csv'
    train = pd_read_csv(os.path.join(args.path_to_dataset, name_train))
    with open(os.path.join(args.path_to_dataset, 'features.json'), 'r') as file:
        features = json.load(file)
    if args.use_null_version:
        train[features['text']] = ""
    if args.binary:
        features["output_dim"] = 2

    # Load tokenizer
    loaders = []
    logger.info('Creating dataloaders with tokenizers')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ipdb.set_trace()
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(args.model_name, num_labels=features['output_dim'], pad_token_id=tokenizer.eos_token_id) # ADDED
    else:
        config = AutoConfig.from_pretrained(args.model_name, num_labels=features['output_dim'])

    # Create dataloaders
    train_ds = TextDataSet(text_df=train, features=features, tokenizer=tokenizer)
    
    # DataLoaders creation:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision=='fp16' else None))
    
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    
    # load model
    logger.info(f'Loading model {args.model_name}')
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        from_tf=bool(".ckpt" in args.model_name),
        config=config,
    )

    if features['label_dic'] is not None:
        model.config.label2id = features['label_dic']
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Load optimizer and scheduler
    logger.info(f'Loading optimizer {args.optimizer} and scheduler {args.scheduler_type}')
    optimizer_grouped_parameters = [
        {
            "params": [p 
                        for n, p in model.named_parameters() 
                        if not (("bias" in n) or (f"LayerNorm" in n and 'weight' in n) or ("BatchNorm" in n and 'weight' in n))],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p 
                        for n, p in model.named_parameters() 
                        if (("bias" in n) or (f"LayerNorm" in n and 'weight' in n) or ("BatchNorm" in n and 'weight' in n))],
            "weight_decay": 0.0,
        },
    ]

    optimizer = get_optimizer(args.optimizer)(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    else:
        args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # train model
    logger.info('Starting training')
    train_losses = []

    # accelerator
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    empty_cache()
    backends.cudnn.benchmark = True
    # ipdb.set_trace()
    for epoch in range(1, args.num_epochs+1):
        results = pd_DataFrame()
        model.train()
        total_loss_train = 0
        
        for step, batch in enumerate(train_loader):
            # with accelerator
            with accelerator.accumulate(model):
                prepare_batch = {k:v for k,v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                if args.binary:
                    prepare_batch["labels"] = (prepare_batch["labels"]==DICT_LABEL_HATE_SPEECH[args.class_to_binarize]).long()

                with accelerator.autocast():
                    outputs = model(**prepare_batch)
                    loss, logits = outputs[:2]

                accelerator.backward(loss) # accelerator
                total_loss_train += loss.item()*len(batch['labels'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            empty_cache()

        total_loss_train = total_loss_train / len(train_loader.dataset)
        train_losses.append(total_loss_train)
        prepare_batch = batch = None
        
        logger.info(f"epoch {epoch}: train {total_loss_train}")
        
        # Accelerator
        accelerator.wait_for_everyone()
        accelerator.save({
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'model_structure': str(model),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'optimizer_structure': str(optimizer),
            'scheduler': str(lr_scheduler),
            'parameters': vars(args)
        },os.path.join(args.path_to_save_results, f'checkpoint_epoch_{epoch}.pt')
        )

        
        epochs_id = [x for x in range(1, epoch+1)]
        results['epoch'] = epochs_id
        results['train_loss'] = train_losses
        results.to_csv(os.path.join(args.path_to_save_results, 'results_training.csv'))

    logger.info(f'Training ends. Check logs and results at {args.path_to_save_results}')
    logger.info(f'')
    accelerator.end_training()