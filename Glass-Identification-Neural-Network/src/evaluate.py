"""Avaliação de um pipeline treinado num conjunto de dados: métricas, relatório por classe e
matriz de confusão. Usado pelo final_test.py e pelo mlp_vs_baselines.py."""
import matplotlib
matplotlib.use("Agg")  # grava PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES, SCORERS


def class_labels(classes):
    """Rótulos legíveis: nomes das classes do vidro no problema de 6 classes, '0'/'1' nos agrupamentos."""
    classes = list(classes)
    if set(classes) <= {0, 1}:
        return [str(c) for c in classes]
    return [CLASS_NAMES.get(int(c), str(c)) for c in classes]


def score_all(pipe, X, y):
    """Todas as métricas de config.SCORERS. Uma métrica que não pode ser calculada vira NaN."""
    out = {}
    for name, scorer in SCORERS.items():
        try:
            out[name] = float(scorer(pipe, X, y))
        except ValueError as e:  # ex.: roc_auc com uma classe ausente no conjunto
            out[name] = float("nan")
            print(f"Aviso: {name} não calculada ({e})")
    return out


def per_class_report(pipe, X, y):
    """Precisão, recall, F1 e nº de amostras por classe (DataFrame)."""
    classes = list(pipe.classes_)
    rep = classification_report(y, pipe.predict(X), labels=classes,
                                target_names=class_labels(classes), output_dict=True,
                                zero_division=0)
    return pd.DataFrame(rep).T


def confusion(pipe, X, y):
    """Matriz de confusão (DataFrame real x predito)."""
    classes = list(pipe.classes_)
    names = class_labels(classes)
    cm = confusion_matrix(y, pipe.predict(X), labels=classes)
    return pd.DataFrame(cm, index=[f"real_{n}" for n in names],
                        columns=[f"pred_{n}" for n in names])


def plot_confusion(cm, out, title):
    """Matriz de confusão: cor = proporção da classe real; número = contagem."""
    counts = cm.to_numpy()
    norm = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    labels = [c.removeprefix("pred_") for c in cm.columns]
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 1.0 * len(labels) + 2.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    ax.set_title(title)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="proporção da classe real")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return np.asarray(norm)
