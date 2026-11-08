"""Avaliação final no conjunto de teste (rodar só no fim, uma vez, na execução escolhida).

Uso: python final_test.py --run 20261112_143012_a3f9c1_R6_combinacao

Grava em results/final/<run>/: metrics.json, classification_report.csv (por classe),
confusion_matrix.csv e .png, baseline_majoritario.json e meta.json.
"""
import argparse
import json
import sys
from datetime import datetime

import joblib
import yaml
from sklearn.dummy import DummyClassifier

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar o joblib
from src.builder import apply_target, build_pipeline, target_name
from src.config import FINAL_DIR, RUNS_DIR
from src.data import load_test, load_train
from src.evaluate import confusion, per_class_report, plot_confusion, score_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="nome da pasta em results/runs/")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run
    if not (run_dir / "config.yaml").exists():
        sys.exit(f"Execução '{args.run}' não encontrada em {RUNS_DIR}")
    out_dir = FINAL_DIR / args.run
    if out_dir.exists():
        print(f"Já avaliado no teste. Resultado guardado em {out_dir}:")
        print((out_dir / "metrics.json").read_text(encoding="utf-8"))
        return

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg.pop("name", None)
    X_train, y_train = apply_target(cfg, *load_train())
    X_test, y_test = apply_target(cfg, *load_test())
    if (run_dir / "pipeline.joblib").exists():
        # Avalia exatamente a rede salva na execução
        pipe = joblib.load(run_dir / "pipeline.joblib")
    else:
        pipe = build_pipeline(cfg).fit(X_train, y_train)

    # Mesmas métricas e mesmos scorers da validação cruzada (src/config.py:SCORERS)
    metrics = score_all(pipe, X_test, y_test)
    report = per_class_report(pipe, X_test, y_test)
    cm = confusion(pipe, X_test, y_test)
    # Referência no mesmo teste: sempre prever a classe mais frequente do treino
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base = score_all(dummy, X_test, y_test)

    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "baseline_majoritario.json").write_text(json.dumps(base, indent=2),
                                                       encoding="utf-8")
    report.to_csv(out_dir / "classification_report.csv")
    cm.to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion(cm, out_dir / "confusion_matrix.png",
                   f"Matriz de confusão no teste ({len(y_test)} amostras)")
    meta = {"run": args.run, "date": datetime.now().isoformat(timespec="seconds"),
            "target": target_name(cfg), "n_train": len(X_train), "n_test": len(X_test)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"{'métrica':>18} {'rede':>8} {'maioritária':>12}")
    for m, v in metrics.items():
        print(f"{m:>18} {v:8.3f} {base[m]:12.3f}")
    print(f"\nPor classe:\n{report.round(3).to_string()}")
    print(f"\nMatriz de confusão:\n{cm.to_string()}\n-> {out_dir}")
    print("\nAtenção: o teste é pequeno (as classes raras têm 2 a 3 amostras); leia as notas "
          "por classe como indicativo.")


if __name__ == "__main__":
    main()
