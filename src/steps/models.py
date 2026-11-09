"""Etapa 4 (model): a rede e os modelos de referência.

Os parâmetros escritos no YAML vão direto para o construtor. `random_state` é fixado em SEED,
mas pode ser trocado no YAML (é o que o R10 faz para medir a robustez à inicialização).
Os padrões fixados aqui (ex.: max_iter=2000) saem da config, para a mesma rede ter sempre o
mesmo hash, escrita com ou sem eles (ver builder.normalize_model).
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.config import SEED


def mlp(**params):
    """MLPClassifier (perceptron multicamadas) com saída softmax e perda de entropia cruzada.

    max_iter alto (2000): com os 100 neurônios e max_iter=200 do padrão do scikit-learn a rede
    em geral não converge nesta base. hidden_layer_sizes aceita lista no YAML ([16, 8]).
    """
    params.setdefault("random_state", SEED)
    params.setdefault("max_iter", 2000)
    if isinstance(params.get("hidden_layer_sizes"), list):
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return MLPClassifier(**params)


def logreg(**params):
    """Regressão logística multinomial: o "modelo simples" de referência (sem não linearidade)."""
    params.setdefault("random_state", SEED)
    params.setdefault("max_iter", 2000)
    return LogisticRegression(**params)


def dummy(**params):
    """Baseline de classe majoritária (most_frequent) ou aleatório estratificado (stratified)."""
    params.setdefault("strategy", "most_frequent")
    params.setdefault("random_state", SEED)
    return DummyClassifier(**params)
