"""Etapa 2 (selection). Não é usada nas rodadas: são só 9 variáveis e a rede já pondera cada uma.

Fica disponível para testes avulsos.
"""
from functools import partial

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from src.config import SEED


def kbest(k=6, score="f_classif"):
    # mutual_info é aleatório: sem seed, a mesma config daria notas diferentes
    funcs = {"f_classif": f_classif,
             "mutual_info": partial(mutual_info_classif, random_state=SEED)}
    return SelectKBest(funcs[score], k=k)
