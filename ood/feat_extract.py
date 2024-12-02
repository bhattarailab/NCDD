#! /usr/bin/env python3

import torch
import os
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time
import copy
import timm

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes)
batch_size = args.batch_size
FORCE_RUN = False


dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
output = model(dummy_input)
if args.name in ['swinv2','vgg','resnet18']:
    features = F.adaptive_avg_pool2d(model.forward_features(dummy_input), 1).squeeze() 
else:
    features = model.forward_features(dummy_input)[:,0].squeeze()


print(output.shape)
print(features.shape)
feat_dim = features.shape[0]

begin = time.time()

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:

    cache_name = f"./cache/{args.in_dataset}_{split}_{args.name}_in_alllayers.npy.npz"
    if FORCE_RUN or not os.path.exists(cache_name):

        feat_log = np.zeros((len(in_loader.dataset), feat_dim))

        score_log = np.zeros((len(in_loader.dataset), num_classes))
        label_log = np.zeros(len(in_loader.dataset))


        # model_features.eval()
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(in_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

            # score, feature_list = model.feature_list(inputs)
            score = model(inputs)
            if args.name in ['swinv2','vgg','resnet18']:
                out = F.adaptive_avg_pool2d(model.forward_features(inputs), 1).squeeze()
            else:
                out = model.forward_features(inputs)[:,0]

            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            label_log[start_ind:end_ind] = targets.data.cpu().numpy()
            score_log[start_ind:end_ind] = score.data.cpu().numpy()

            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(in_loader)}")

        np.savez(cache_name, feat_log.T, score_log.T, label_log)


for ood_dataset in args.out_datasets:
    loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
    out_loader = loader_test_dict.val_ood_loader
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy.npz"
    if FORCE_RUN or not os.path.exists(cache_name):
        ood_feat_log = np.zeros((len(out_loader.dataset),feat_dim))
        ood_score_log = np.zeros((len(out_loader.dataset), num_classes))

        model.eval()
        for batch_idx, (inputs, _) in enumerate(out_loader):
            inputs = inputs.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

            score =  model(inputs)
            if args.name in ['swinv2','vgg','resnet18']:
                out = F.adaptive_avg_pool2d(model.forward_features(inputs), 1).squeeze() 
            else:
                out = model.forward_features(inputs)[:,0]
            
            ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(out_loader)}")
        # np.save(cache_name, (ood_feat_log.T, ood_score_log.T))
        np.savez(cache_name, ood_feat_log.T, ood_score_log.T)

print(time.time() - begin)