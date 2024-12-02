python feat_extract.py --in-dataset gastro_id --out-datasets gastro_ood --name resnet18 --model-arch resnet18
# python feat_extract.py --in-dataset gastro_id --out-datasets gastro_ood --name mlpmixer --model-arch mlpmixer
python run_other_algos.py --in-dataset gastro_id --out-datasets gastro_ood --name resnet18 --model-arch resnet18
