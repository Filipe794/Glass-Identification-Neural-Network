"""Prepara os dados: lê data/raw/glass.data e gera data/split/train.csv e test.csv.

Preparação (feita ANTES de dividir, para a mesma amostra não cair no treino e no teste):
  1. descarta a coluna Id_number (o arquivo é ordenado por classe: o Id vazaria o rótulo);
  2. remove linhas duplicadas;
  3. divisão estratificada treino/teste (TEST_SIZE, SEED).

O conjunto de teste fica isolado até o final_test.py.

Uso: python -m src.make_split [--force]
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_COLS, RAW_FILE, SEED, TARGET, TEST_CSV, TEST_SIZE, TRAIN_CSV


def load_raw():
    if not RAW_FILE.exists():
        raise SystemExit(
            f"{RAW_FILE} não existe. Baixe o glass.data em "
            "https://archive.ics.uci.edu/dataset/42/glass+identification (botão Download, "
            "arquivo glass.data dentro do zip) e coloque em data/raw/.")
    df = pd.read_csv(RAW_FILE, header=None, names=RAW_COLS)
    n0 = len(df)
    df = df.drop(columns="Id_number")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Linhas no arquivo: {n0} | duplicatas removidas: {n0 - len(df)} | restantes: {len(df)}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regera o split (invalida a comparação com execuções antigas)")
    args = ap.parse_args()

    if TRAIN_CSV.exists() and not args.force:
        print(f"Split já existe em {TRAIN_CSV.parent}. Nada feito (use --force para regerar).")
        return

    df = load_raw()
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                   stratify=df[TARGET])
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"train: {len(train)} linhas | test: {len(test)} linhas")
    tab = pd.DataFrame({"total": df[TARGET].value_counts(), "train": train[TARGET].value_counts(),
                        "test": test[TARGET].value_counts()}).sort_index().fillna(0).astype(int)
    print("\nAmostras por classe:\n" + tab.to_string())


if __name__ == "__main__":
    main()
