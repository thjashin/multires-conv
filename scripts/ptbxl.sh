#CUDA_VISIBLE_DEVICES=0 python classification.py --dataset=ptbxl --layer=multires --d_model=128 --lr=0.0045 --epochs=100 --dropout=0.2 --n_layers=6 --batch_size=50 --port=12670  --warmup 5 --tree_select=fading --weight_decay=0.06 --task rhythm &
#CUDA_VISIBLE_DEVICES=1 python classification.py --dataset=ptbxl --layer=multires --d_model=128 --lr=0.0045 --epochs=100 --dropout=0.2 --n_layers=6 --batch_size=50 --port=12668  --warmup 5 --tree_select=fading --weight_decay=0.06 --task subdiagnostic &
#CUDA_VISIBLE_DEVICES=2 python classification.py --dataset=ptbxl --layer=multires --d_model=128 --lr=0.0045 --epochs=100 --dropout=0.2 --n_layers=6 --batch_size=50 --port=12669  --warmup 5 --tree_select=fading --weight_decay=0.06 --task form &
#CUDA_VISIBLE_DEVICES=3 python classification.py --dataset=ptbxl --layer=multires --d_model=128 --lr=0.0045 --epochs=100 --dropout=0.2 --n_layers=6 --batch_size=50 --port=12667  --warmup 5 --tree_select=fading --weight_decay=0.06 --task diagnostic &
CUDA_VISIBLE_DEVICES=4 python classification.py --dataset=ptbxl --layer=multires --d_model=128 --lr=0.0045 --epochs=100 --dropout=0.2 --n_layers=6 --batch_size=50 --port=12671  --warmup 5 --tree_select=fading --weight_decay=0.06 --task superdiagnostic #&
#CUDA_VISIBLE_DEVICES=5 python classification.py --dataset=ptbxl --layer=multires --d_model=128 --lr=0.0045 --epochs=100 --dropout=0.2 --n_layers=6 --batch_size=50 --port=12672  --warmup 5 --tree_select=fading --weight_decay=0.06 --task all &