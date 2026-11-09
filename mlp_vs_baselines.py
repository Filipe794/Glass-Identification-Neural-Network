"""Compara a rede vencedora com os modelos de referência (R9): na validação cruzada, fold a fold,
e no teste, lado a lado. Só roda depois do teste final da rede (final_test.py).

Os baselines são execuções comuns do R0 (dummy, regressão logística), então já têm a CV e o
pipeline.joblib. Todos precisam ter o mesmo split e o mesmo alvo da rede.

Uso:
  python mlp_vs_baselines.py --mlp <vencedor> --baselines <R0_dummy> <R0_logreg> ...

Grava em results/mlp_vs_baselines/<mlp>/:
  cv.csv                métricas da CV lado a lado, com o teste t corrigido (rede x cada baseline)
  teste.csv             métricas no teste dos modelos salvos, com as mesmas funções
  f1_por_classe.csv/png F1 de cada classe no teste, rede x baselines
"""
import argparse
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar o joblib
from compare import SIG_COL, corrected_ttest, read_run, run_path, significant
from src.builder import apply_target
from src.config import BASELINES_DIR, FINAL_DIR, LOWER_IS_BETTER, SCORING
from src.data import load_test
from src.evaluate import per_class_report, score_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", required=True, help="pasta da rede vencedora em results/runs/")
    ap.add_argument("--baselines", nargs="+", required=True, help="pastas dos baselines (R0)")
    args = ap.parse_args()

    mlp_dir = run_path(args.mlp)
    if not (FINAL_DIR / mlp_dir.name / "metrics.json").exists():
        sys.exit(f"Rode antes o teste final: python final_test.py --run {mlp_dir.name}")

    name_m, sha_m, tgt_m, cv_m = read_run(mlp_dir)
    cfg = yaml.safe_load((mlp_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg.pop("name", None)
    X_test, y_test = apply_target(cfg, *load_test())

    models = {name_m: joblib.load(mlp_dir / "pipeline.joblib")}
    cv_rows, test_rows, class_rows = [], [], {}
    test_rows.append({"modelo": name_m, **score_all(models[name_m], X_test, y_test)})
    class_rows[name_m] = per_class_report(models[name_m], X_test, y_test)["f1-score"]

    for arg in args.baselines:
        path = run_path(arg)
        name_b, sha_b, tgt_b, cv_b = read_run(path)
        if sha_b != sha_m or tgt_b != tgt_m:
            sys.exit(f"'{name_b}' usa outro split ou outro alvo que a rede: não é comparável.")
        if not cv_b[["repeat", "fold"]].equals(cv_m[["repeat", "fold"]]):
            sys.exit(f"'{name_b}' não tem os mesmos folds da rede.")
        pipe_b = joblib.load(path / "pipeline.joblib")
        test_rows.append({"modelo": name_b, **score_all(pipe_b, X_test, y_test)})
        class_rows[name_b] = per_class_report(pipe_b, X_test, y_test)["f1-score"]
        for m in SCORING:
            d = (cv_m[m] - cv_b[m]).to_numpy()   # rede - baseline
            t, p = corrected_ttest(d)
            cv_rows.append({"baseline": name_b, "métrica": m, "média rede": cv_m[m].mean(),
                            "média baseline": cv_b[m].mean(), "dif (rede-baseline)": d.mean(),
                            "t": t, "p": p, SIG_COL: significant(p)})

    out = BASELINES_DIR / mlp_dir.name
    out.mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "cv.csv", index=False)
    test = pd.DataFrame(test_rows)
    test.to_csv(out / "teste.csv", index=False)
    f1c = pd.DataFrame(class_rows).drop(index=["accuracy", "macro avg", "weighted avg"],
                                        errors="ignore")
    f1c.to_csv(out / "f1_por_classe.csv")

    ax = f1c.plot.bar(figsize=(9, 4.5), width=0.8)
    ax.set_ylabel("F1 no teste")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 por classe no teste: rede x baselines")
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "f1_por_classe.png", dpi=120)
    plt.close()

    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print("Validação cruzada (teste t corrigido, rede - baseline):")
        print(cv[cv["métrica"].isin(["f1_macro", "balanced_accuracy", "accuracy"])]
              .to_string(index=False, float_format="%.3f"))
        low = [m for m in SCORING if m in LOWER_IS_BETTER]
        print(f"(em {', '.join(low)}, menor é melhor: dif negativa favorece a rede)")
        print("\nTeste:")
        print(test.to_string(index=False, float_format="%.3f"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
