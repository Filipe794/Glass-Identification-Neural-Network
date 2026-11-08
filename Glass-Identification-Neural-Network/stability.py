"""Robustez e estabilidade (R10): a rede vencedora depende da semente de inicialização? As
importâncias das variáveis se mantêm entre folds e entre sementes?

Roda depois do teste final. Não muda a escolha do modelo.

Uso:
  python run_experiments.py experiments/r10_robustez.yaml
  python stability.py --run <vencedor> --seeds R10_

Grava em results/stability/<run>/:
  por_semente.csv         métricas da CV (média e desvio) do vencedor e de cada semente
  ranking_importancias.csv  concordância do ranking das variáveis entre as sementes (Spearman)
  folds_ranking.csv       concordância do ranking entre os folds do vencedor (Spearman)
  importancias_folds.png  importância por permutação em cada fold (boxplot)
"""
import argparse
import itertools
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compare import read_run, run_path, select_round
from src.config import PRIMARY_METRIC, SCORING, STABILITY_DIR


def spearman_pairs(wide):
    """Spearman entre todos os pares de linhas (folds ou sementes) de uma tabela linha x variável."""
    rhos = []
    for a, b in itertools.combinations(range(len(wide)), 2):
        rho = spearmanr(wide.iloc[a], wide.iloc[b]).statistic
        if not np.isnan(rho):
            rhos.append(rho)
    return np.array(rhos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pasta do vencedor em results/runs/")
    ap.add_argument("--seeds", required=True, metavar="PREFIXO",
                    help="prefixo do nome das execuções do R10 (ex.: R10_)")
    args = ap.parse_args()

    win = run_path(args.run)
    name_w, sha_w, tgt_w, cv_w = read_run(win)
    seed_dirs = [d for d in select_round(args.seeds) if d.resolve() != win.resolve()]
    if not seed_dirs:
        sys.exit(f"Nenhuma execução com nome começando por '{args.seeds}'. "
                 "Rode antes experiments/r10_robustez.yaml.")

    rows, perm = [], {}
    for d in [win] + seed_dirs:
        name, sha, tgt, cv = read_run(d)
        if sha != sha_w or tgt != tgt_w:
            print(f"Aviso: '{name}' usa outro split/alvo; pulei.", file=sys.stderr)
            continue
        cfg_seed = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("name", name)
        row = {"execução": name, "nome": cfg_seed}
        for m in SCORING:
            row[f"{m}_mean"], row[f"{m}_std"] = cv[m].mean(), cv[m].std(ddof=0)
        if "convergiu" in cv:
            row["convergiu_pct"] = 100 * cv["convergiu"].mean()
        rows.append(row)
        imp = pd.read_csv(d / "importances_permutation.csv").set_index("feature")["mean"]
        perm[name] = imp
    out = STABILITY_DIR / win.name
    out.mkdir(parents=True, exist_ok=True)

    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(out / "por_semente.csv", index=False)
    col = f"{PRIMARY_METRIC}_mean"
    print(f"{PRIMARY_METRIC} na CV por execução (vencedor + {len(rows) - 1} sementes):")
    print(by_seed[["execução", col, f"{PRIMARY_METRIC}_std"]].to_string(
        index=False, float_format="%.3f"))
    print(f"\nEntre as execuções: média {by_seed[col].mean():.3f} | desvio {by_seed[col].std(ddof=1):.3f}"
          f" | mín {by_seed[col].min():.3f} | máx {by_seed[col].max():.3f}")

    # Ranking das variáveis entre as sementes (importância da rede final, medida na CV)
    wide_seeds = pd.DataFrame(perm).T
    rho_s = spearman_pairs(wide_seeds)
    tops = wide_seeds.idxmax(axis=1).value_counts()
    summ = pd.DataFrame([{"pares": len(rho_s), "spearman_media": rho_s.mean(),
                          "spearman_min": rho_s.min(),
                          "variavel_mais_importante": tops.index[0],
                          "execucoes_com_ela_no_topo": int(tops.iloc[0]),
                          "execucoes": len(wide_seeds)}])
    summ.to_csv(out / "ranking_importancias.csv", index=False)
    print("\nConcordância do ranking de importância entre as execuções:")
    print(summ.to_string(index=False, float_format="%.3f"))

    # Ranking entre os folds do vencedor
    fi = pd.read_csv(win / "fold_importances.csv")
    wide_f = fi.pivot_table(index=["repeat", "fold"], columns="feature", values="importance")
    rho_f = spearman_pairs(wide_f)
    top_f = wide_f.idxmax(axis=1).value_counts()
    sf = pd.DataFrame([{"pares": len(rho_f), "spearman_media": rho_f.mean(),
                        "spearman_min": rho_f.min(),
                        "variavel_mais_importante": top_f.index[0],
                        "folds_com_ela_no_topo": int(top_f.iloc[0]), "folds": len(wide_f)}])
    sf.to_csv(out / "folds_ranking.csv", index=False)
    print("\nConcordância do ranking de importância entre os folds do vencedor:")
    print(sf.to_string(index=False, float_format="%.3f"))

    order = wide_f.mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([wide_f[c] for c in order], tick_labels=list(order))
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel(f"queda do {PRIMARY_METRIC} ao embaralhar")
    ax.set_title(f"Importância por permutação em cada fold ({len(wide_f)} folds)")
    fig.tight_layout()
    fig.savefig(out / "importancias_folds.png", dpi=120)
    plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
