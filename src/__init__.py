import warnings

from sklearn import set_config
from sklearn.exceptions import ConvergenceWarning

# Transformadores devolvem DataFrame => colunas mantêm o nome entre as etapas.
# Fica aqui para valer em qualquer import de src (inclusive ao carregar um pipeline.joblib).
set_config(transform_output="pandas")

# O MLP avisa a cada fold que não convergiu dentro de max_iter. Isso não se perde: o runner
# grava n_iter e convergiu em fold_scores.csv (e o summarize.py mostra a % de folds que
# convergiram). O aviso repetido só polui a saída.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
