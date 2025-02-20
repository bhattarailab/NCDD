python feat_extract.py --in-dataset Kvasir_id --id_path_train './data/kvasir_data/training' --id_path_valid './data/kvasir_data/testing/ID' --out-datasets Kvasir_ood --ood_path './data/kvasir_data/testing/OOD' --name vit --model-arch vit --weights model_training/checkpoint/vit_timm_kvasir_pt-4_ckpt_small16.t7
python run_ncdd.py --in-dataset Kvasir_id --out-datasets Kvasir_ood --name vit --model-arch vit
python run_other_algos.py --in-dataset Kvasir_id --id_path_train './data/kvasir_data/training' --id_path_valid './data/kvasir_data/testing/ID' --out-datasets Kvasir_ood --ood_path './data/kvasir_data/testing/OOD' --name vit --model-arch vit --weights model_training/checkpoint/vit_timm_kvasir_pt-4_ckpt_small16.t7

# python run_other_algos.py --in-dataset Kvasir_id --out-datasets Kvasir_ood --name vit --model-arch vit --weights ./model_training/checkpoint/vit_timm_kvasir_pt-4_ckpt_small16.t7
