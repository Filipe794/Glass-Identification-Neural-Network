"""Etapa 1 (scaler).

As variáveis estão em escalas muito diferentes (RI em torno de 1,51 a 1,53; Si em torno de
70 a 75; Ba e Fe quase sempre zero). Redes treinadas por gradiente dependem disso.
  none      sem padronização (referência para medir o efeito)
  standard  média 0 e desvio 1
  robust    mediana e intervalo interquartil (menos sensível aos valores extremos, que nesta
            base muitas vezes são o que caracteriza as classes raras, ex.: Ba nos faróis)
  minmax    reescala para [0, 1]
Todos aprendem os parâmetros no fit (só no treino do fold) e apenas aplicam no transform.
"""
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def standard():
    return StandardScaler()


def robust():
    return RobustScaler()


def minmax():
    return MinMaxScaler()
