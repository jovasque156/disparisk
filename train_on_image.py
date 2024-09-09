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
from src.datasets.classes.ImageDataSet import ImageDataSet

# Import model modules
from torch.hub import load as model_load

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
import torchvision.transforms as transforms
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from torch.nn import CrossEntropyLoss
from torch.nn import Linear
from torch import backends
from torch import Tensor as torch_Tensor
from accelerate import Accelerator

DEVICE = 'cuda' if cuda_is_available else 'cpu'

BASE_PATH = '/data/to/results/'

DICT_LABEL_FACET = {
    "lawman": 27,
    "nurse": 31,
    }

class_to_binarize_choice = [x for x in DICT_LABEL_FACET]

def get_optimizer(opt):
    optimizers = {
        'AdamW': AdamW,
    }
    return optimizers.get(opt, SGD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DispaRisk Experiments")

    # Setting for dataset
    parser.add_argument('--path_to_train_dataset', type=str, help="Path to the folder where imgs_processed/ folder is contained")
    parser.add_argument('--path_to_img_df', type=str, help="Path to the folder containing the csv with the files' names and sensitive,label featuers")
    parser.add_argument('--use_null_version', action='store_true', help='Use to use null version of train set')
    parser.add_argument('--path_to_save_results', type=str, help='Path to folder where results will be saved')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchsize for training. If not given, it is set to 32.')
    parser.add_argument('--binary', action='store_true', help='Use to set target as binary')
    parser.add_argument('--class_to_binarize', type=str, default="lawman", help='Name of the class in label to binarize')
    
    # Selecting model configuration
    parser.add_argument('--model_name', type=str, default='alexnet', help='Name of the model to load choices: [alexnet,vgg19_bn,resnet152,densenet161,googlenet,mobilenet_v3_large,ViT]. Default is set to alexnet')
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

    assert args.class_to_binarize in class_to_binarize_choice, f"{args.class_to_binarize} not in the class1 variable"

    os.makedirs(args.path_to_save_results, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16")

    logger.info('Creating dataloaders with transforms')
    if 'ViT' in args.model_name:
        preprocess = transforms.Compose([
                                        transforms.Resize((384, 384)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5),
                                    ])

        data_transforms = {
                        'train': transforms.Compose([
                            transforms.RandomResizedCrop((384,384)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5),
                        ])
                    }
    else:
        resize_arg = 299 if args.model_name == 'inception_v3' else 256
        centercorp_arg = 299 if args.model_name == 'inception_v3' else 224

        data_transforms = {
                        'train': transforms.Compose([
                            transforms.RandomResizedCrop(resize_arg),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
                        ])
                    }
    
    if args.use_null_version:
        name_train = 'train_null_vit.csv' if 'ViT' in args.model_name else 'train_null.csv'
    else:
        name_train = 'train.csv'
    
    image_df = pd_read_csv(os.path.join(args.path_to_img_df, name_train))
    with open(os.path.join(args.path_to_img_df, 'features.json'), 'r') as file:
        features = json.load(file)
    if args.binary:
        features["output_size"] = 2

    img_ds = ImageDataSet(
                        annotations_df=image_df, 
                        features = features,
                        root_dir = os.path.join(args.path_to_train_dataset,features['root_dir']),
                        transform = data_transforms['train']
                            )

    train_loader = DataLoader(img_ds, batch_size=args.batch_size, shuffle=True)

    img_ds = None #free memory
    data_transforms = None #free memory

    # load model
    logger.info(f'Loading model {args.model_name}')
    
    if 'ViT' in args.model_name:
        from pytorch_pretrained_vit import ViT
        if args.model_name=='ViT_32':
            if args.pretrained:
                model = ViT('L_32_imagenet1k',
                    # image_size = min(X_celeba.shape[-1], X_celeba.shape[-2]),
                    num_classes = features['output_size'],
                    pretrained=args.pretrained)
            else:
                from pytorch_pretrained_vit.model import ViT
                model = ViT('L_32_imagenet1k', num_classes=features['output_size'])
        elif args.model_name=='ViT_16':
            if args.pretrained:
                model = ViT('L_16_imagenet1k',
                    # image_size = min(X_celeba.shape[-1], X_celeba.shape[-2]),
                    num_classes = features['output_size'],
                    pretrained=args.pretrained)
            else:
                from pytorch_pretrained_vit.model import ViT
                model = ViT('L_16_imagenet1k', num_classes=features['output_size'])
    else:
        if args.pretrained:
            model = model_load('pytorch/vision:v0.10.0', args.model_name, pretrained=True)
        else:
            if 'resnet152' in args.model_name:
                from torchvision.models.resnet import resnet152
                model = resnet152()
            if 'resnet18' in args.model_name:
                from torchvision.models.resnet import resnet18
                model = resnet18()
            if 'googlenet' in args.model_name:
                from torchvision.models.googlenet import googlenet
                model = googlenet()
            if 'inception_v3' in args.model_name:
                from torchvision.models.inception import inception_v3
                model == inception_v3()
            if 'mobilenet_v3_large' in args.model_name:
                from torchvision.models.mobilenetv3 import mobilenet_v3_large
                model = mobilenet_v3_large()
            if 'mobilenet_v3_small' in args.model_name:
                from torchvision.models.mobilenetv3 import mobilenet_v3_small
                model = mobilenet_v3_small()
    
        if args.model_name == 'resnet152':
            model.fc = Linear(2048, features['output_size'])
        elif args.model_name == 'resnet18':
            model.fc = Linear(512, features['output_size'])
        elif args.model_name == 'googlenet':
            model.fc = Linear(1024, features['output_size'])
        elif args.model_name == 'inception_v3':
            model.fc = Linear(2048, features['output_size'])
        elif args.model_name == 'mobilenet_v3_large':
            model.classifier[3] = Linear(1280, features['output_size'])
        elif args.model_name == 'mobilenet_v3_small':
            model.classifier[3] = Linear(1024, features['output_size'])
    
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
            # with accelerator
            with accelerator.accumulate(model):
                X, label = batch['X'], batch['label']
                if args.binary:
                    label = (label==DICT_LABEL_FACET[args.class_to_binarize]).long()
                batch = None
                
                with accelerator.autocast():
                    if 'vgg' in args.model_name:
                        X = X.contiguous()
                    outputs = model(X) 
                    if not isinstance(outputs, torch_Tensor):
                        outputs = outputs.logits
                    
                    loss = ce(outputs, label)

                accelerator.backward(loss) 
                total_loss_train += loss.item()*X.shape[0]
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            empty_cache()
        
        total_loss_train = total_loss_train / len(train_loader.dataset)
        train_losses.append(total_loss_train)
        X = None
        label = None

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