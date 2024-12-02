#! /usr/bin/env python3

import torch
import os
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score
import cv_uncertainty as unc

START = time.time()

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes

model=get_model(args, num_classes)

x = torch.randn([1,3,224,224]).cuda()
feat = model.forward_features(x)

print(feat.shape)

batch_size = args.batch_size

UNC_METHODS = [
    # unc.MSP(),
    # unc.Odin(), 
    # unc.EnergyQuant(),
    # unc.EntropyQuant(),
    # unc.MaxLogit(),
    unc.Blood(),
    unc.Neco(),
    unc.FDBD()
]


for ood_dataset in args.out_datasets:
    loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
    out_loader = loader_test_dict.val_ood_loader
    
    # model.eval()
    for uncertainty in UNC_METHODS:
        print(f"\t\t\t{uncertainty.name} - {time.time()-START}s")
        
        if uncertainty.name in ['neco']:
            scores_in, scores_ood = uncertainty.quantify(
                                            model=model,
                                            train_loader = trainloaderIn,
                                            val_loader = testloaderIn,
                                            ood_loader = out_loader,
                                            model_name = args.name
                                            )
        elif uncertainty.name == 'fdbd':
            scores_in, scores_ood = uncertainty.quantify(
                                            model=model,
                                            train_loader = trainloaderIn,
                                            val_loader = testloaderIn,
                                            ood_loader = out_loader,
                                            model_name = args.name,
                                            id_dataset = args.in_dataset
                                            )
        
        else:

            scores_in = uncertainty.quantify(
                data_loader=testloaderIn,
                model=model,
            )
            scores_ood = uncertainty.quantify(
                data_loader=out_loader,
                model=model,
            )
        
        
        DATA = [0 for _ in range(len(scores_in))] + [
            1 for _ in range(len(scores_ood))
        ]
        aucs = roc_auc_score(DATA, scores_in.tolist() + scores_ood.tolist())
        num_ind = len(scores_in)
        recall_num = int(np.floor(0.95 * num_ind))
        thresh = np.sort(scores_in)[recall_num]
        fpr = np.sum(scores_ood <= thresh)/len(scores_ood)
        
        
        print(
            f" AUC = {round(aucs.mean()*100, 2):.2f} FPR@95TPR = {round(fpr.mean()*100, 2):.2f}"
        )


 
            