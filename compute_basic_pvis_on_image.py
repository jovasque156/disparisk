# Import basics
import os
import argparse
import logging
import json
import dill as pickle
from pandas import read_csv as pd_read_csv
from pandas import DataFrame

# Import data modules
from torch.utils.data import DataLoader
from src.datasets.classes.ImageDataSet import ImageDataSet

# Import model modules
from torch.hub import load as model_load

# Import utils
import torchvision.transforms as transforms
from torch import no_grad as torch_no_grad
from torch import log2 as torch_log2
from torch import arange as torch_arange
from torch import load as torch_load
from torch import Tensor as torch_Tensor
from torch import from_numpy as torch_from_numpy
from torch.cuda import is_available as cuda_is_available
from torch.cuda import empty_cache
from transformers import set_seed
from torch.nn import Softmax
from torch.nn import Linear

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

DICT_LABEL_FACET = {
    "lawman": 27,
    "nurse": 31,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computing pvis')

    # Setting dataset config
    parser.add_argument('--dataset_name', type=str, default='facet', help='Name of dataset')
    parser.add_argument('--label', type=str, help='Name of the feature set as label. If not given, default configuration will be set. Check load_dataset.')
    parser.add_argument('--sensitive', type=str, help='Name of the feature set as sensitive. If not given, default configuration will be set. Check load_dataset.')
    parser.add_argument('--split', type=str, default='train_val_test', help='Split to use', choices=['train_val_test', 'train_test'])
    parser.add_argument('--stratify_by', type=str, default='label', help='Stratification used for training')
    parser.add_argument('--scaler', type=str, default='scaler', help='Scaler used for training')
    parser.add_argument('--binary', action='store_true', help='Use to set target as binary')
    parser.add_argument('--class_to_binarize', type=str, default="lawman", help='Name of the class in label to binarize')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--set', type=str, default='test', help='Set to use for computation', choices=['train', 'test', 'val'])

    parser.add_argument('--model_name', type=str, default='alexnet', help='Name of the model to load choices: [alexnet,vgg19_bn,resnet152,densenet161,googlenet,mobilenet_v3_large,ViT]. Default is set to alexnet')
    parser.add_argument('--pretrained', action='store_true', help='Use to load a pretrained model')
    parser.add_argument('--use_calibrator', action='store_true', help='Use calibrator')
    parser.add_argument('--set_calibrator', type=str, default='train', help='Set which calibrator to use from [train, val, test]', choices=['train', 'val', 'test'])

    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training. Default value is 12345")

    args = parser.parse_args()

    path_to_img_df = os.path.join(BASE_DATASET_PATH, args.dataset_name, 'preprocessed', f'label_{args.label}_sensitive_{args.sensitive}', 
                                args.split, f'stratify_by_{args.stratify_by}')

    path_to_dataset = os.path.join(BASE_DATASET_PATH, args.dataset_name)

    if args.seed is not None:
        set_seed(args.seed)

    logger.info('Creating dataloaders with transforms')
    if 'ViT' in args.model_name:
        preprocess = transforms.Compose([
                                        transforms.Resize((384, 384)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5),
                                    ])

        data_transforms = {
                        'val': transforms.Compose([
                            transforms.Resize((384, 384)), 
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5),
                        ])
                    }
    else:
        resize_arg = 299 if args.model_name == 'inception_v3' else 256
        centercorp_arg = 299 if args.model_name == 'inception_v3' else 224

        data_transforms = {
                        'val': transforms.Compose([
                            transforms.Resize(resize_arg),
                            transforms.CenterCrop(centercorp_arg),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
                        ]),
                    }

    logger.info(f'Loading {args.set}-null-version dataset')
    name_df = f'{args.set}_null_vit.csv' if 'ViT' in args.model_name else f'{args.set}_null.csv'
    image_df = pd_read_csv(os.path.join(path_to_img_df, name_df))
    with open(os.path.join(path_to_img_df, 'features.json'), 'r') as file:
        features = json.load(file)

    img_ds = ImageDataSet(
                        annotations_df=image_df, 
                        features = features,
                        root_dir = os.path.join(path_to_dataset,features['root_dir']),
                        transform = data_transforms['val']
                            )

    ds_loader = DataLoader(img_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = DataFrame()
    results['id'] = image_df['Unnamed: 0'] if 'Unnamed: 0' in image_df.columns else image_df.iloc[:,0]
    results[f'sensitive_{args.sensitive}'] = image_df[features['sensitive']]
    results[f'label_{args.label}'] = image_df[features['label']]

    for set_ in ['null', 'std']:
        epochs = 2 if set_ == 'null' else 6

        if set_ =='std':
            logger.info(f'Loading {args.set}-std-version dataset')
            image_df = pd_read_csv(os.path.join(path_to_img_df, f'{args.set}.csv'))
            with open(os.path.join(path_to_img_df, 'features.json'), 'r') as file:
                features = json.load(file)

            img_ds = ImageDataSet(
                                annotations_df=image_df, 
                                features = features,
                                root_dir = os.path.join(path_to_dataset,features['root_dir']),
                                transform = data_transforms['val']
                                    )

            ds_loader = DataLoader(img_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        for epoch in range(1,epochs):
            logger.info(f'Loading model from checkpoint {epoch}')

            if args.binary:
                features["output_size"]=2
                path_to_model = os.path.join(BASE_RESULTS_PATH, args.dataset_name, args.model_name, f'{args.label}_{args.class_to_binarize}_{set_}')
            else:
                path_to_model = os.path.join(BASE_RESULTS_PATH, args.dataset_name, args.model_name, f'{args.label}_{set_}')
            
            if not os.path.isfile(os.path.join(path_to_model, f'checkpoint_epoch_{epoch}.pt')):
                logger.info(f"Model at {set_} and epoch {epoch} doesn't exist. Skipping computation at this set")
                continue

            if 'ViT' in args.model_name:
                from pytorch_pretrained_vit import ViT
                if args.model_name=='ViT_32':
                    if args.pretrained:
                        model = ViT('L_32_imagenet1k',
                            num_classes = features['output_size'],
                            pretrained=args.pretrained)
                    else:
                        from pytorch_pretrained_vit.model import ViT
                        model = ViT('L_32_imagenet1k', num_classes=features['output_size'])
                elif args.model_name=='ViT_16':
                    if args.pretrained:
                        model = ViT('L_16_imagenet1k',
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
            
            checkpoint = torch_load(os.path.join(path_to_model, f'checkpoint_epoch_{epoch}.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)

            if args.use_calibrator:
                path_to_cal = os.path.join(path_to_model, f'calibrator_{set_}_{args.set_calibrator}_epoch{epoch}.sav')
                with open(path_to_cal, 'rb') as f:
                    calibrator = pickle.load(f)
            
            model.eval()
            distr = Softmax(dim=1)
            sensitives = []
            labels = []
            predictions = []
            prob_pred = []
            H_y = []
            for batch in ds_loader:
                X, label = batch['X'].to(DEVICE), batch['label'].to(DEVICE)
                if args.binary:
                    label = (label==DICT_LABEL_FACET[args.class_to_binarize]).long()
                
                epsilon = 1e-3
                with torch_no_grad():
                    if 'vgg' in args.model_name:
                        X = X.contiguous()
                    outputs = model(X)
                    if not isinstance(outputs, torch_Tensor):
                        outputs = outputs.logits
                    if args.use_calibrator:
                        outputs = calibrator.calibrate(distr(model(X)).cpu().tolist())
                        outputs = torch_from_numpy(outputs)
                    else:
                        outputs = distr(model(X)).clamp(min=epsilon, max=1-epsilon)
                
                predictions += outputs.argmax(dim=1).cpu().tolist()
                sensitives += batch['sensitive'].cpu().numpy().astype(int).tolist()
                labels += label.cpu().numpy().astype(int).tolist()
                H_y += (-1*torch_log2(outputs[torch_arange(outputs.size(0)), label])).cpu().tolist()

            empty_cache()
            
            results[f'Hy_{set_}_{args.label}_epoch{epoch}'] = H_y
            if set_=='null':
                results[f'prediction_{args.label}_null'] = predictions    
            results[f'prediction_{args.label}_epoch{epoch}'] = predictions
            name_file = f'pvis_on_{args.set}.csv' if not args.use_calibrator else f'pvis_on_{args.set}_calibrator_{args.set_calibrator}.csv'
            results.to_csv(os.path.join(path_to_model, name_file))
        
        if os.path.isfile(os.path.join(path_to_model, f'checkpoint_epoch_{epoch}.pt')):
            logger.info(f'Saving results at {os.path.join(path_to_model, name_file)}')
            