from pathlib import Path

from sklearn.metrics import (f1_score, get_scorer, log_loss, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold

# Raiz do projeto calculada a partir deste arquivo: funciona local e no Colab
BASE = Path(__file__).resolve().parents[1]
RAW_FILE = BASE / "data/raw/glass.data"
TRAIN_CSV = BASE / "data/split/train.csv"
TEST_CSV = BASE / "data/split/test.csv"
RUNS_DIR = BASE / "results/runs"
FINAL_DIR = BASE / "results/final"
SUMMARY_CSV = BASE / "results/summary.csv"
COMPARACOES_DIR = BASE / "results/comparacoes"
BASELINES_DIR = BASE / "results/mlp_vs_baselines"
STABILITY_DIR = BASE / "results/stability"

SEED = 42
# Fração do conjunto de teste, isolado até o fim. Com 213 amostras (depois de remover a
# duplicata), a classe 6 tem 9: com 0.2, ficam 7 no treino e 2 no teste.
TEST_SIZE = 0.2

# glass.data não tem cabeçalho: a primeira coluna é o Id (descartado) e a última é a classe
RAW_COLS = ["Id_number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
TARGET = "Type"
FEATURES = [c for c in RAW_COLS if c not in ("Id_number", TARGET)]
# A classe 4 (janela de veículo, não float) existe na documentação, mas não tem amostras.
CLASS_NAMES = {1: "1 construção float", 2: "2 construção não float", 3: "3 veículo float",
               5: "5 recipientes", 6: "6 louça de mesa", 7: "7 faróis"}

# Validação cruzada do treino: estratificada e repetida, sempre a mesma (mesmos folds em todas
# as execuções => as notas podem ser comparadas fold a fold).
N_SPLITS, N_REPEATS = 5, 10
CV = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)


def logloss(est, X, y):
    """Log loss (entropia cruzada) da probabilidade prevista (menor é melhor)."""
    return log_loss(y, est.predict_proba(X), labels=est.classes_)


# Nome da métrica -> função (est, X, y). A única lista usada pelo runner, final_test e afins.
# zero_division=0: uma classe nunca prevista (comum nos baselines) vale 0, sem aviso.
SCORERS = {
    "accuracy": get_scorer("accuracy"),
    "balanced_accuracy": get_scorer("balanced_accuracy"),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "roc_auc_ovr": get_scorer("roc_auc_ovr"),   # AUC um-contra-todos, média macro
    "log_loss": logloss,
}
SCORING = list(SCORERS)
# Métricas em que menor é melhor (o compare.py e o summarize.py invertem a ordenação)
LOWER_IS_BETTER = {"log_loss"}
# Macro = todas as classes pesam igual, inclusive as de 9 a 17 amostras. A acurácia sozinha
# esconde o colapso nas classes raras (ver docs/baseline-e-plano-de-treinamento.md).
PRIMARY_METRIC = "f1_macro"

# Importância por permutação: repetições por coluna, em cada fold de validação
PERM_REPEATS = 5

# Nível de significância do teste t corrigido (bilateral) usado pelo compare.py.
# Não entra no hash das execuções: mudar o ALPHA só muda a leitura das comparações.
ALPHA = 0.05
