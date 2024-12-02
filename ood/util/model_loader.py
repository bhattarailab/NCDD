import os
# import models.densenet as dn
# import models.wideresnet as wn


import torch
import timm

def get_model(args, num_classes, load_ckpt=True, load_epoch=None):

    weights_path = args.weights
    if args.model_arch == 'vit':
        if(args.in_dataset=='Kvasir_id'):
            # weights_path = args.weights
            model = timm.create_model("vit_small_patch16_224_in21k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)
            #do this to extract features
            # model_feats = copy.deepcopy(model)
            # model_feats.head_drop = nn.Identity()
            # model_feats.head = nn.Identity()
        elif(args.in_dataset=='gastro_id'):
            # weights_path = 'checkpoint/vit_timm_gastro_pt-4_ckpt_small16.t7'
            model = timm.create_model("vit_small_patch16_224_in21k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)


    elif args.model_arch == 'resnet18':
            if(args.in_dataset=='Kvasir_id'):
                from models.resnet import resnet18_kvasir
                model = resnet18_kvasir()  
                # checkpoint = torch.load("checkpoint/checkpoint_2.t7", map_location='cpu')
                checkpoint = torch.load(args.weights,map_location="cpu")
                model.load_state_dict(checkpoint["model"])  

            elif (args.in_dataset=='gastro_id'):
                from models.resnet import resnet18_gastrovision
                model = resnet18_gastrovision()
                # checkpoint = torch.load("checkpoint/checkpoint_23.pth", map_location='cpu')
                checkpoint = torch.load(args.weights,map_location="cpu")
                model.load_state_dict(checkpoint)

    elif args.model_arch == 'mlpmixer':
        if(args.in_dataset=='Kvasir_id'):
            # weights_path = 'checkpoint/mlpmixer-4_kv_ckpt.t7'
            model = timm.create_model("mixer_b16_224.miil_in21k_ft_in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)
        
        elif (args.in_dataset=='gastro_id'):
            # weights_path = 'checkpoint/mlpmixer-4_ckpt.t7'
            model = timm.create_model("mixer_b16_224.miil_in21k_ft_in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)
    
    elif args.model_arch == 'convmixer':
        if(args.in_dataset=='Kvasir_id'):
            # weights_path = 'checkpoint/convmixer-4_ckpt.t7'
            model = timm.create_model("convmixer_768_32.in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)

        elif(args.in_dataset=='gastro_id'):
            # weights_path = 'checkpoint/convmixer-4_ckpt.t7'
            model = timm.create_model("convmixer_768_32.in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)

    elif args.model_arch == 'deit':
        if(args.in_dataset=='Kvasir_id'):
            # weights_path = 'checkpoint/deit-4_kv_ckpt.t7'
            model = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)

        elif(args.in_dataset=='gastro_id'):
            # weights_path = 'checkpoint/deit-4_gastro_ckpt.t7'
            model = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)

    elif args.model_arch == 'swinv2':
        if(args.in_dataset=='Kvasir_id'):
            # weights_path = 'checkpoint/swinv2-4_ckpt.t7'
            model = timm.create_model("swinv2_cr_small_ns_224.sw_in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)

    elif args.model_arch == 'vgg':
        if(args.in_dataset=='Kvasir_id'):
            # weights_path = 'checkpoint/vgg-4_ckpt.t7'
            model = timm.create_model("vgg16.tv_in1k", pretrained=False, num_classes=num_classes, checkpoint_path=weights_path)


    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model
