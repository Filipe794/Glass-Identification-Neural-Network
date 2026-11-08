"""Etapa 3 (balancing): reamostragem para as classes raras. Só atua no fit (treino do fold),
porque o pipeline é do imblearn: o fold de validação nunca é reamostrado.

O MLPClassifier não tem class_weight; por isso o desbalanceamento é tratado por reamostragem.

  random_over  duplica amostras das classes menores (sorteio com reposição)
  smote        cria amostras sintéticas por interpolação entre vizinhos (Chawla et al., 2002)

Cuidado com a classe 6 (9 amostras na base, ~5 no treino de cada fold da CV): o SMOTE precisa
de pelo menos k_neighbors + 1 amostras em cada classe. O padrão aqui é k_neighbors = 3.
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler

from src.config import SEED


def random_over(**params):
    return RandomOverSampler(random_state=SEED, **params)


def smote(k_neighbors=3, **params):
    return SMOTE(k_neighbors=k_neighbors, random_state=SEED, **params)
