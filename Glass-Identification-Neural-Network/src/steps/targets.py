"""Alvo do experimento (agrupamento de classes, R5).

Não é uma etapa do pipeline: cada função recebe (X, y) e devolve (X, y) no formato do problema.
O runner aplica a mesma função ao treino (antes da validação cruzada) e ao teste (no
final_test.py), então a rede nunca vê o alvo original.

  six_classes        o problema original (classes 1, 2, 3, 5, 6, 7); é o padrão
  window_vs_other    1 = janela (classes 1, 2, 3); 0 = outros (5, 6, 7)
  float_vs_nonfloat  só as janelas (1, 2, 3): 1 = float (1 e 3); 0 = não float (2).
                     É a tarefa do estudo original citado na página da UCI.
                     Descarta as classes 5, 6 e 7.

Os resultados de alvos diferentes NÃO são comparáveis (problemas e amostras diferentes): o
compare.py se recusa a compará-los.
"""


def six_classes(X, y):
    return X, y


def window_vs_other(X, y):
    return X, y.isin([1, 2, 3]).astype(int)


def float_vs_nonfloat(X, y):
    keep = y.isin([1, 2, 3])
    return X[keep].reset_index(drop=True), y[keep].isin([1, 3]).astype(int).reset_index(drop=True)
