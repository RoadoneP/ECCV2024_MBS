python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 main.py --config ./configs/voc.yaml --log ours

