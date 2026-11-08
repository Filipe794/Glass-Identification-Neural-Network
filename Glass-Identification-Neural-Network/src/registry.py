"""Catálogo: nome usado no YAML -> função que constrói o passo.

Método novo = função no arquivo da etapa + uma linha aqui.
"""
from src.steps import balancing, models, scalers, selection, targets

# Ordem fixa do pipeline. NÃO alterar por experimento.
# O scaler fica DENTRO do pipeline: é ajustado só no treino de cada fold (anti-vazamento).
# O balancing (reamostragem) é a última etapa antes do modelo e só atua no fit.
STEP_ORDER = ["scaler", "selection", "balancing"]  # + "model" no fim
# Etapas que podem ser omitidas no YAML. `target` não é um passo do pipeline: define o
# problema (6 classes ou um agrupamento) e é aplicado a X e y antes da validação cruzada.
OPTIONAL_STEPS = ["target"]
# Valor de uma etapa opcional equivalente a "não escrita"
OPTIONAL_DEFAULT = {"target": "six_classes"}

REGISTRY = {
    "scaler": {
        "none": None,
        "standard": scalers.standard,
        "robust": scalers.robust,
        "minmax": scalers.minmax,
    },
    "selection": {
        "none": None,
        "kbest": selection.kbest,
    },
    "balancing": {
        "none": None,
        "random_over": balancing.random_over,
        "smote": balancing.smote,
    },
    "model": {
        "mlp": models.mlp,
        "logreg": models.logreg,
        "dummy": models.dummy,
    },
    "target": {
        "six_classes": targets.six_classes,
        "window_vs_other": targets.window_vs_other,
        "float_vs_nonfloat": targets.float_vs_nonfloat,
    },
}
