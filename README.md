# CaFBI

# Environment
- python 3.9
- tensorflow 2.12
- tqdm 4.65.0
- numpy 1.24.2
- scipy 1.10.1
- pandas 1.5.3
- toolz 0.12.0

# Data preprocessing
python data_pre/movielens.py

# Model training
python run.py --config configs/ml1m.json --log exp/logs/ml1m/CaFBI_ML.log

# Evaluation
python eval.py --best_epoch 99 --config configs/ml1m.json --log exp/logs/ml1m/CaFBI_ML_eval.log
