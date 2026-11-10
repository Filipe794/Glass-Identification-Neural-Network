"""Roda um YAML de experimentos com OUTRA semente de CV, sem alterar nenhum arquivo do projeto.

Serve para checar se uma escolha (ex.: o vencedor do R2b) depende de como o treino foi
particionado em folds. Só a partição muda: a inicialização da rede e a reamostragem continuam
com a SEED do config.py (42). Tudo acontece na validação cruzada do treino; o teste não é aberto.

Uso:
  python cv_seed_check.py 7 experiments/r2b_interacao.yaml
  python cv_seed_check.py 7 experiments/r2b_interacao.yaml --only R2b_baseline
  python cv_seed_check.py 7 experiments/r2b_interacao.yaml --only R2b_interacao

Como as execuções ficam separadas das originais:
  - runner.SEED e runner.CV são trocados em memória. A SEED entra no hash, então as execuções
    novas têm hash diferente e nada é pulado nem sobrescrito (cada execução vai para uma pasta
    nova e imutável).
  - O nome ganha o prefixo S<semente>_ (ex.: S7_R2b_interacao__...), para que
    `compare.py --rodada R2b_` não pegue as execuções novas.

CUIDADO ao comparar: o compare.py confere o train.csv e o alvo, mas NÃO a semente da CV.
Compare sempre dentro de uma semente (uma rodada por vez) e nunca use --incluir com uma
execução de outra semente, senão o teste t pareado compara folds diferentes (inválido):
  python compare.py --rodada S7_R2b_ --salvar
  python compare.py --rodada S7_R2b_ --metric balanced_accuracy --salvar
O baseline do YAML também roda com a nova semente, então não precisa de --incluir.
"""
import sys

from sklearn.model_selection import RepeatedStratifiedKFold

import run_experiments
import src.runner as runner
from src.config import N_REPEATS, N_SPLITS, SEED


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    try:
        seed = int(sys.argv.pop(1))
    except ValueError:
        sys.exit("O primeiro argumento deve ser a semente da CV (inteiro), ex.: 7")
    if seed == SEED:
        sys.exit(f"A semente {SEED} é a das execuções existentes: o hash seria igual e o runner "
                 f"pularia tudo. Use outra.")

    # Troca em memória (nenhum arquivo é alterado). A CV usa a nova semente; a SEED do runner
    # entra no hash e no meta.json, separando estas execuções das originais.
    runner.SEED = seed
    runner.CV = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)

    # Prefixo no nome: o nome é só rótulo (não entra no hash), mas o compare.py --rodada filtra
    # por ele.
    original_run = run_experiments.run
    run_experiments.run = lambda name, cfg, **kw: original_run(f"S{seed}_{name}", cfg, **kw)

    print(f"CV com semente {seed} (as originais usam {SEED}); nomes com prefixo S{seed}_")
    run_experiments.main()


if __name__ == "__main__":
    main()
