import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.autograd import grad



class EntropyQuant:
    name = "entropy"

    def quantify(self, model, data_loader, **kwargs):
        probs_list =[]
        with torch.no_grad():
            for imgs,classes in data_loader:
                imgs = imgs.cuda()
                classes = classes.cuda()
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs.detach().cpu())
        probs_list = torch.cat(probs_list).numpy()
        probs_list = np.clip(probs_list, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs_list * np.log(probs_list), axis=1)
        return entropies


class EnergyQuant:
    name = "energy"

    def quantify(self, model, data_loader, **kwargs):
        logits = []
        for X,y in data_loader:
            X = X.cuda()
            logits.extend(model(X).tolist())
        logits = np.array(logits)
        return -np.log(np.sum(np.exp(logits), axis=1))
    

class MaxLogit:
    name = "maxlogit"

    def quantify(self, model, data_loader, **kwargs):
        logits = []
        for X,y in data_loader:
            X = X.cuda()
            logit = model(X)
            logits.extend((-logit.max(dim=1).values).tolist())
        logits = np.array(logits)
        return logits
    

def odin_score(model, inputs):
    """
    Calculates softmax outlier scores on ODIN pre-processed inputs.

    :param x: input tensor
    :return: outlier scores for each sample
    """
    temper = 100
    noiseMagnitude1 = 0.0005

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # outputs = model(inputs)
    # # outputs = forward_func(inputs, model)

    # maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # # Using temperature scaling
    # outputs = outputs / temper

    # labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    # loss = criterion(outputs, labels)
    # loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, gradient,alpha=-noiseMagnitude1)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        # outputs = forward_func(tempInputs, model)
        outputs = model(tempInputs)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return -scores


class Odin:
    name = "odin"
    

    def quantify(self, model, data_loader,**kwargs):
        scores = []
        # base_model = copy.deepcopy(model)
        for x,_ in data_loader:
            x = x.cuda()
            # new_model = copy.deepcopy(model)
            # new_model.cuda()
            # new_model.train()
            score = odin_score(model,x)
            # x_hat = odin_preprocessing(new_model, x)
            # score = -model(x_hat).softmax(dim=1).max(dim=1).values
            scores.extend(score)
        return np.array(scores)
        
class MSP:
    name= "msp"
    
    def quantify(self, model, data_loader,**kwargs):
        scores = []
        with torch.no_grad():
            for x,_ in data_loader:
                x = x.cuda()
                logits = model(x)
                scores.extend(-np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1))
        return np.array(scores)
          
class Neco:

    # ---------------------------------------
    name = 'neco'

    def quantify(self, model,train_loader,val_loader, ood_loader,model_name="x",neco_dim =100 ,**kwargs):
        train_features = []
        val_features = []
        ood_features = []
        
        train_logits = []
        val_logits = []
        ood_logits = []
        
        with torch.no_grad():
            if model_name in ['swinv2','vgg', 'convmixer','resnet18']:
                for x,_ in train_loader:
                    x = x.cuda()
                    train_features.extend(F.adaptive_avg_pool2d(model.forward_features(x), 1).squeeze().tolist())
                    train_logits.extend(model(x).tolist())
                for x,_ in val_loader:
                    x = x.cuda()
                    val_features.extend(F.adaptive_avg_pool2d(model.forward_features(x), 1).squeeze().tolist())    
                    val_logits.extend(model(x).tolist())

                for x,_ in ood_loader:
                    x = x.cuda()
                    ood_features.extend(F.adaptive_avg_pool2d(model.forward_features(x), 1).squeeze().tolist())  
                    ood_logits.extend(model(x).tolist())
        
            else:
                for x,_ in train_loader:
                    x = x.cuda()
                    train_features.extend(model.forward_features(x)[:,0].tolist())
                    train_logits.extend(model(x).tolist())
                for x,_ in val_loader:
                    x = x.cuda()
                    val_features.extend(model.forward_features(x)[:,0].tolist())  
                    val_logits.extend(model(x).tolist())
                for x,_ in ood_loader:
                    x = x.cuda()
                    ood_features.extend(model.forward_features(x)[:,0].tolist())   
                    ood_logits.extend(model(x).tolist())

        train_features = np.array(train_features)
        val_features = np.array(val_features)
        ood_features = np.array(ood_features)
        
        train_logits = np.array(train_logits)
        logit_id_val = np.array(val_logits)
        logit_ood = np.array(ood_logits)

        print(train_features.shape)
    
        ss = StandardScaler() 
        complete_vectors_train = ss.fit_transform(train_features)
        complete_vectors_test = ss.transform(val_features)
        complete_vectors_ood = ss.transform(ood_features)

        pca_estimator = PCA(train_features.shape[1])
        _ = pca_estimator.fit_transform(complete_vectors_train)
        cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
        cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)

        score_id_maxlogit = logit_id_val.max(axis=-1)
        score_ood_maxlogit = logit_ood.max(axis=-1)
        if model_name in ['deit', 'swin']:
            complete_vectors_train = train_features
            complete_vectors_test = val_features
            complete_vectors_ood = ood_features

        cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
        cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]
        l_ID = []
        l_OOD = []

        for i in range(cls_test_reduced.shape[0]):
            sc_complet = LA.norm((complete_vectors_test[i, :]))
            sc = LA.norm(cls_test_reduced[i, :])
            sc_finale = sc/sc_complet
            l_ID.append(sc_finale)
        for i in range(cls_ood_reduced.shape[0]):
            sc_complet = LA.norm((complete_vectors_ood[i, :]))
            sc = LA.norm(cls_ood_reduced[i, :])
            sc_finale = sc/sc_complet
            l_OOD.append(sc_finale)
        l_OOD = np.array(l_OOD)
        l_ID = np.array(l_ID)
        #############################################################
        score_id = l_ID
        score_ood = l_OOD
        if model_name != 'resnet18':
            score_id *= score_id_maxlogit
            score_ood *= score_ood_maxlogit

        return -score_id, -score_ood

class FDBD:
    name = "fdbd"
    
    def quantify(self, model, train_loader, val_loader, ood_loader, model_name="x",id_dataset='Kvasir_ood',**kwargs):
        print(model_name)
        scores = []
        train_features = []
        val_features = []
        ood_features = []
        
        train_logits = []
        val_logits = []
        ood_logits = []
        
        
        with torch.no_grad():
            if model_name in ['swinv2','vgg', 'convmixer','resnet18']:
                for x, _ in train_loader:
                    x = x.cuda()
                    train_features.append(F.adaptive_avg_pool2d(model.forward_features(x), 1).squeeze())
                    train_logits.append(model(x))
                for x, _ in val_loader:
                    x = x.cuda()
                    val_features.append(F.adaptive_avg_pool2d(model.forward_features(x), 1).squeeze())    
                    val_logits.append(model(x))
                for x, _ in ood_loader:
                    x = x.cuda()
                    ood_features.append(F.adaptive_avg_pool2d(model.forward_features(x), 1).squeeze())  
                    ood_logits.append(model(x))
            else:
                for x, _ in train_loader:
                    x = x.cuda()
                    train_features.append(model.forward_features(x)[:,0])
                    train_logits.append(model(x))
                for x, _ in val_loader:
                    x = x.cuda()
                    val_features.append(model.forward_features(x)[:,0])  
                    val_logits.append(model(x))
                for x, _ in ood_loader:
                    x = x.cuda()
                    ood_features.append(model.forward_features(x)[:,0])   
                    ood_logits.append(model(x))

        train_features = torch.cat(train_features)
        val_features = torch.cat(val_features)
        ood_features = torch.cat(ood_features)
        
        train_logits = torch.cat(train_logits)
        val_logits = torch.cat(val_logits)
        ood_logits = torch.cat(ood_logits)
        
        if model_name == 'resnet18':
            linear_layer = model.fc.state_dict()
        else:
            linear_layer = model.head.state_dict()

        if model_name in ['swinv2', 'vgg']:
            w = linear_layer['fc.weight'].cpu().detach()
            b = linear_layer['fc.bias'].cpu().detach()
        else:
            w = linear_layer['weight'].cpu().detach()
            b = linear_layer['bias'].cpu().detach()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        train_mean = torch.mean(train_features, dim=0).to(device)

        if id_dataset == 'Kvasir_id':
            mat_dim =3
        else:
            mat_dim = 11

        denominator_matrix = torch.zeros((mat_dim,mat_dim)).to(device)#for kvasir 3, gastro 11
        for p in range(mat_dim):
            w_p = w - w[p, :]
            denominator = torch.norm(w_p, dim=1)
            denominator[p] = 1
            denominator_matrix[p, :] = denominator

        #################### fDBD score OOD detection #################

        values, nn_idx = val_logits.max(1)
        logits_sub = torch.abs(val_logits - values.view(-1, 1).repeat(1, mat_dim))
        score_in = torch.sum(logits_sub / denominator_matrix[nn_idx], dim=1) / torch.norm(val_features - train_mean, dim=1)
        score_in = score_in.float().cpu().numpy()

        values, nn_idx = ood_logits.max(1)
        logits_sub = torch.abs(ood_logits - values.view(-1, 1).repeat(1, mat_dim))
        scores_out_test = torch.sum(logits_sub / denominator_matrix[nn_idx], dim=1) / torch.norm(ood_features - train_mean, dim=1)
        scores_out_test = scores_out_test.float().cpu().numpy()
        
        return -score_in, -scores_out_test

class Blood:

    # ---------------------------------------
    name = 'blood'


    def vit_forward_features(self,model,x):
        intermediates = []
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        for block in model.blocks:
            x = block(x)
            intermediates.append(x)
        x = model.norm(x)
        y = model.forward_head(x)
        return y,intermediates
    
    def quantify(self, model,data_loader, n_estimators=15, estimator=True,device='cuda:0'):
        model.eval()
        layer_norms = []
        cnt = 0
        norms = []
        for X,_ in data_loader:
            model.zero_grad()
            X = X.to(device)
            # _, out =model.forward_intermediates(X,output_fmt='NLC')#for mlpmixer
            _,out = self.vit_forward_features(model,X)#for deit, vit
            norms = []
            for i in range(1, len(out) - 1):
                emb_X = out[
                    i
                ]  # (batch_size, squence_length, embeding_size)
                emb_Y = out[
                    i + 1
                ]  # (batch_size, squence_length, embeding_size)

                if estimator:
                    ests = []
                    for n in range(n_estimators):
                        v = torch.randn((emb_Y.shape[0], emb_Y.shape[2])).to(
                            device
                        )
                        est = grad(
                            (emb_Y[:, 0, :] * v).sum(), emb_X, retain_graph=True
                        )[0][
                            :, 0, :
                        ]  # (batch_size, embedding_size)
                        w = torch.randn((emb_Y.shape[0], emb_Y.shape[2])).to(
                            device
                        )
                        ests.append(((est * w).sum(dim=1) ** 2).cpu())  # (batch_size)

                    norm_ests = torch.stack(ests, dim=1)  # (batch_size, n_estimators)
                    norms.append(norm_ests.mean(dim=1))  # (batch_size)

                else:
                    grads = [
                        grad(emb_Y[:, 0, j].sum(), emb_X, retain_graph=True)[0][
                            :, 0, :
                        ].cpu()
                        for j in range(emb_Y.shape[2])
                    ]
                    norm_ests = torch.cat(
                        grads, dim=1
                    )  # (batch_size, embeding_size*embeding_size)
                    norms.append((norm_ests**2).sum(dim=1))  # (batch_size)

            layer_norms.append(
                torch.stack(norms, dim=0)
            )  # (num_layers-1 (11), batch_size)
        norms = torch.cat(layer_norms, dim=1)
        score = norms[-1,:].cpu().numpy()
        return score




