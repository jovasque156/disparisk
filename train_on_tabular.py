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
from src.datasets.classes.TabularDataSet import TabularDataSet

# Import model modules
from src.models.base_models.FNN import FNN
from configs.config_classifiers import config_FNN

# Import optimizers
from torch.optim import (
        AdamW, 
        SGD
        )
from transformers import (
        SchedulerType,
        get_scheduler,
        set_seed
        )

# Import torch utils
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from torch import backends
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss

DEVICE = 'cuda' if cuda_is_available else 'cpu'

BASE_PATH = '/path/to/results/'

def get_optimizer(opt):
    optimizers = {
        'AdamW': AdamW,
    }
    return optimizers.get(opt, SGD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DispaRisk Experiments")

    # Setting for dataset
    parser.add_argument('--path_to_train_dataset', type=str, help='Path to the folder with the train dataset and features json')
    parser.add_argument('--use_null_version', action='store_true', help='Use to use null version of train set')
    parser.add_argument('--path_to_save_results', type=str, help='Path to folder where results will be saved')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchsize for training. If not given, it is set to 32.') 
        
    # Selecting model configuration
    parser.add_argument('--model_config', type=str, default='h1', help='Id of the configuration to load. Default is set to h1')

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
    logger.info('Loading dataset and features')

    train = pd_read_csv(os.path.join(args.path_to_train_dataset, f'train{"_null" if args.use_null_version else ""}.csv'))
    with open(os.path.join(args.path_to_train_dataset, 'features.json'), 'r') as file:
        features = json.load(file)

    tds = TabularDataSet(tabular_df=train, 
                        features = features
                            )

    train_loader = DataLoader(tds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # load model
    logger.info(f'Loading model {args.model_config}')
    n_features = len(features['input_space'])
    model = FNN(
                input_size=n_features,
                hidden_layers=config_FNN[args.model_config]['n_hidden_layers']*[n_features],
                output_size=features['output_dim'],
                config_model={
                                'activation_hidden': config_FNN[args.model_config]['activation_hidden'],
                                'activation_output': config_FNN[args.model_config]['activation_output'],
                                'p_dropout': config_FNN[args.model_config]['p_dropout'],
                                'norm': config_FNN[args.model_config]['norm']
                            },
                )

    # Load optimizer and scheduler
    logger.info(f'Loading optimizer {args.optimizer} and scheduler {args.scheduler_type}')
    optimizer_grouped_parameters = [
        {
            "params": [p 
                        for n, p in model.named_parameters() 
                        if not (("bias" in n) or (f"{config_FNN[args.model_config]['norm']}" in n and 'weight' in n))],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p 
                        for n, p in model.named_parameters() 
                        if (("bias" in n) or (f"{config_FNN[args.model_config]['norm']}" in n and 'weight' in n))],
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
    ce = CrossEntropyLoss()
    train_losses = []

    # accelerator
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
    empty_cache()
    backends.cudnn.benchmark = True
    
    for epoch in range(1, args.num_epochs+1):
        results = pd_DataFrame()
        model.train()
        total_loss_train = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                X, label = batch['X'], batch['label']
                batch = None

                with accelerator.autocast():
                    outputs = model(X)
                    loss = ce(outputs, label)

                accelerator.backward(loss)
                total_loss_train += loss.item()*X.shape[0]
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            empty_cache()

        total_loss_train = total_loss_train / len(train_loader.dataset)
        train_losses.append(total_loss_train)
        
        logger.info(f"epoch {epoch}: train {total_loss_train}")

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
    logger.info(f' ')
    accelerator.end_training()