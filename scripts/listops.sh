# CUDA_VISIBLE_DEVICES=0,1,2,3,4
CUDA_VISIBLE_DEVICES=0,1 python classification.py --dataset=listops --layer=multires --d_model=128 --lr=0.003 --epochs=100 --dropout=0.1 --n_layers=10 --batch_size=50 --batchnorm --warmup=1 --tree_select=fading --weight_decay=0.03 --indep_res_init --port=52669 --kernel_size=4
