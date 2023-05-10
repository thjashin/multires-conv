# CUDA_VISIBLE_DEVICES=0,1,2,3,4
CUDA_VISIBLE_DEVICES=0,1 python classification.py --layer=multires --d_model=256 --lr=0.0045 --epochs=250 --dropout=0.25 --n_layers=10 --batch_size=50 --port=12668 --tree_select=fading
