import os
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np
from sklearn.covariance import EmpiricalCovariance
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

cache_name = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers.npy.npz"
npz_data = np.load(cache_name, allow_pickle=True)
feat_log = npz_data['arr_0'].T.astype(np.float32)
score_log = npz_data['arr_1'].T.astype(np.float32)
label_log = npz_data['arr_2']

num_classes = score_log.shape[1]

global_mean = np.zeros((1, feat_log.shape[1]))
global_mean[0] = np.mean(feat_log, axis=0)  

feat_log = feat_log - global_mean

mean_features = np.zeros((num_classes, feat_log.shape[1]))

        # Calculate mean features for each class
for i in range(num_classes):
    class_indices = (label_log == i)
    class_features = feat_log[class_indices]
    mean_features[i] = np.mean(class_features, axis=0)


class_num = score_log.shape[1]

cache_name = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers.npy.npz"
npz_data_val = np.load(cache_name, allow_pickle=True)
feat_log_val = npz_data_val['arr_0'].T.astype(np.float32)
score_log_val = npz_data_val['arr_1'].T.astype(np.float32)
label_log_val = npz_data_val['arr_2']


alpha11 =np.linalg.norm(feat_log_val,ord=1,axis=1)


ood_feat_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy.npz"
    npz_data_ood = np.load(cache_name, allow_pickle=True)
    ood_feat_log = npz_data_ood['arr_0'].T.astype(np.float32)
    ood_score_log = npz_data_ood['arr_1'].T.astype(np.float32)
    alpha22 = np.linalg.norm(ood_feat_log,ord=1,axis=1)
    ood_feat_log = ood_feat_log-global_mean
    ood_feat_log_all[ood_dataset] = ood_feat_log

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

in_max_logit = np.max(score_log_val,axis=1)
ood_max_logit = np.max(ood_score_log,axis=1)

in_min_logit = np.min(score_log_val,axis=1)
ood_min_logit = np.min(ood_score_log,axis=1)

in_avg_logit = np.average(score_log_val,axis=1)
ood_avg_logit = np.average(ood_score_log,axis=1)

feat_log_val = feat_log_val - global_mean

z1 =np.linalg.norm(feat_log_val,ord=1,axis=1)
z2 = np.linalg.norm(ood_feat_log,ord=1,axis=1)

fmean = prepos_feat(mean_features)
ftrain = prepos_feat(feat_log)
ftest = prepos_feat(feat_log_val)
global_mean = prepos_feat(global_mean)
ec = EmpiricalCovariance(assume_centered=True)
ec.fit(np.array(ftrain).astype(np.float64))

food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])


for ood_dataset, food in food_all.items():
    
    index = faiss.IndexFlatL2(ftest.shape[1])
    index.add(fmean)
    for K in [num_classes]:

        score, _ = index.search(ftest, K)

        score_ood, _ = index.search(food, K)


all_results = []
alpha1, alpha2 = 4,1  #hyper parameters
scores_in = np.sum(score[:,1:],axis=1) *np.log(z1/pow(10,alpha1)) -score[:,0]*np.log(z1/pow(10,alpha2))
scores_ood_test = np.sum(score_ood[:,1:],axis=1 ) *np.log(z2/pow(10,alpha1)) -score_ood[:,0]*np.log(z2/pow(10,alpha2))
results = metrics.cal_metric(scores_in, scores_ood_test)
all_results.append(results)
metrics.print_all_results(all_results, args.out_datasets, f"NCDD")
print()




