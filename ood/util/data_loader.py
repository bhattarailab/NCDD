import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from ylib.dataloader.tinyimages_80mn_loader import TinyImages
from ylib.dataloader.imagenet_loader import ImageNet
from ylib.dataloader.svhn_loader import SVHN
from ylib.dataloader.random_data import GaussianRandom, LowFreqRandom


#kvasir transform same transforms as when training
k_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),    
    transforms.Normalize(mean=(0.5705, 0.3812, 0.3293), std=(0.3164, 0.2236, 0.2200)), 
])

g_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4960, 0.3191, 0.2511), std=(0.3268, 0.2357, 0.2142)) #gastrovision
])

kwargs = {'num_workers': 2, 'pin_memory': True}
num_classes_dict = {'Kvasir_id':3, 'gastro_id':11}

def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'batch_size': args.batch_size,
            'k_transform': k_transform,
            'g_transform' : g_transform,
        },
        "eval": {
            'batch_size': args.batch_size,
            'k_transform': k_transform,
            'g_transform' : g_transform,
        },
    })[config_type]

       
    if args.in_dataset == "Kvasir_id":
        #data loading code
        if 'train' in split:
            trainset = torchvision.datasets.ImageFolder(root=args.id_path_train, transform=config.k_transform)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size)
        if 'val' in split:
            valset = torchvision.datasets.ImageFolder(root=args.id_path_valid, transform=config.k_transform)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size)

    elif args.in_dataset == "gastro_id":
        #data loading code
        if 'train' in split:
            trainset = torchvision.datasets.ImageFolder(root=args.id_path_train, transform=config.g_transform)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size)
        if 'val' in split:
            valset = torchvision.datasets.ImageFolder(root=args.id_path_valid, transform=config.g_transform)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size)



    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        # "lr_schedule": lr_schedule,
        "num_classes": num_classes_dict[args.in_dataset],
    })

def get_loader_out(args, dataset=('tim', 'noise'), config_type='default', split=('train', 'val')):

    config = EasyDict({
        "default": {
            'batch_size': args.batch_size,
            'k_transform': k_transform,
            'g_transform' : g_transform,
        },
    })[config_type]
    train_ood_loader, val_ood_loader = None, None

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch_size
         
        if val_dataset == 'Kvasir_ood':
            val_ood_set = torchvision.datasets.ImageFolder(args.ood_path,transform=config.k_transform)
            val_ood_loader = torch.utils.data.DataLoader(val_ood_set,batch_size=batch_size, shuffle=False, num_workers=2)

        elif val_dataset == 'gastro_ood':
            val_ood_set = torchvision.datasets.ImageFolder(args.ood_path,transform=config.g_transform)
            val_ood_loader = torch.utils.data.DataLoader(val_ood_set,batch_size=batch_size, shuffle=False, num_workers=2)


    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader": val_ood_loader,
    })
