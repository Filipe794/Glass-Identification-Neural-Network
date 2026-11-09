import hashlib

import pandas as pd

from src.config import TARGET, TEST_CSV, TRAIN_CSV


def _load(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Coloque o glass.data em data/raw/ e rode: python -m src.make_split")
    df = pd.read_csv(path)
    return df.drop(TARGET, axis=1), df[TARGET]


def load_train():
    return _load(TRAIN_CSV)


def load_test():  # SOMENTE final_test.py e as análises posteriores a ele (mlp_vs_baselines.py)
    return _load(TEST_CSV)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
