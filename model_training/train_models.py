# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
import timm
import os
import argparse
import pandas as pd
import csv
import time
from models import *
from utils import progress_bar
from torchvision.models import resnet50, resnet18
import neptune

API_token = os.environ.get('NEPTUNEAPI')
Project_name = os.environ.get('NEPTUNEPROJECT')

run = neptune.init_run(
    project=Project_name,
    api_token=API_token,  #
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(101)

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="224")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--train_path', default='./kvasir_data/training', help='path to training data')
parser.add_argument('--valid_path', default='./kvasir_data/testing/ID', help='path to valid dataset')

args = parser.parse_args()

num_classes = 3
bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize



#custom transform bhandari
k_transform_train = transforms.Compose(
                    [
                        transforms.Resize(size=(224, 224)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomInvert(0.3),
                        transforms.RandomEqualize(0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5705, 0.3812, 0.3293), std=(0.3164, 0.2236, 0.2200)), #kvasir norm
                    ]
                )

k_transform_test = transforms.Compose(
                    [
                        transforms.Resize(size=(224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5705, 0.3812, 0.3293), std=(0.3164, 0.2236, 0.2200)), #kvasir norm
                    ]
                )

g_transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.4960, 0.3191, 0.2511), std=(0.3268, 0.2357, 0.2142)) #gastrovision
])


#kvasir test transforms
g_transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4960, 0.3191, 0.2511), std=(0.3268, 0.2357, 0.2142)) #gastrovision
])
#custom from image_folder
# k_transform_train = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5705, 0.3812, 0.3293), std=(0.3164, 0.2236, 0.2200)),
# ])

train_path = args.train_path
valid_path = args.valid_path
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=k_transform_train)
valset = torchvision.datasets.ImageFolder(root=valid_path, transform=k_transform_test)

print(len(trainset), len(valset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True,num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(valset,batch_size=bs, shuffle=False,num_workers=4, pin_memory=True)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')

if args.net=='vgg':
    net =timm.create_model('vgg16.tv_in1k', pretrained=True,num_classes =num_classes)
# elif args.net=="convmixer":
#     # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
#     net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    net = timm.create_model('mixer_b16_224.miil_in21k_ft_in1k', pretrained=True,num_classes=num_classes)

elif args.net=="convmixer":
    net = timm.create_model('convmixer_768_32.in1k', pretrained=True,num_classes=num_classes)
elif args.net=="deit":
    net=timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True,num_classes=num_classes)
elif args.net=="swinv2":
    net=timm.create_model('swinv2_cr_small_ns_224.sw_in1k', pretrained=True,num_classes=num_classes)
    



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-gastro_ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


print(args.net)
# Loss is CE
# criterion = losses.SupConLoss() # Custom Implementation
# criterion = nn.CrossEntropyLoss()

# criterion1 = losses.SupConLoss()
criterion = nn.CrossEntropyLoss()


#optimizer
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs,eta_min=1e-7)


##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs = batch["image"]
        # targets = batch["label"]
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_loss = train_loss/(batch_idx+1)
    
    run["train/loss"].append(train_loss)
    return train_loss

##### Validation
best_test_loss = 1000
def test(epoch):
    global best_acc
    global best_test_loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # inputs = batch['image']
            # targets = batch['label']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    run["val/acc"].append(acc)
    run["val/loss"].append(test_loss)
    if test_loss<best_test_loss:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}_kv_ckpt.t7'.format(args.patch))
        best_test_loss= test_loss
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

    
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)


    
run.stop()