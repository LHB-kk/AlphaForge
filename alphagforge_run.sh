# csi300
python train_AFF.py --instruments=csi300 --train_end_year=2021 --seeds=1 --save_name=test --zoo_size=100

python combine_AFF.py --instruments=csi300 --train_end_year=2021 --seeds=1 --save_name=test --n_factors=10 --window=inf

# csi500
python train_AFF.py --instruments=csi500 --train_end_year=2021 --seeds=1 --save_name=test --zoo_size=100

python combine_AFF.py --instruments=csi500 --train_end_year=2021 --seeds=1 --save_name=test --n_factors=10 --window=inf

python 