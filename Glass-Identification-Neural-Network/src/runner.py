"""Roda a validação cruzada de um experimento e grava numa pasta nova e imutável."""
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime

import imblearn
import joblib
import matplotlib
matplotlib.use("Agg")  # grava PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from joblib import parallel_config
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.utils.parallel import Parallel, delayed  # repassa o set_config às threads

from src.builder import apply_target, build_pipeline, mlp_of, model_of, target_name
from src.config import (CV, N_REPEATS, N_SPLITS, PERM_REPEATS, PRIMARY_METRIC, RUNS_DIR,
                        SCORERS, SCORING, SEED, TRAIN_CSV)
from src.data import file_sha256, load_train


def config_hash(cfg, train_sha):
    """Identidade do experimento: config completa + seed + CV + métricas + treino."""
    payload = {"config": cfg, "seed": SEED, "cv": [N_SPLITS, N_REPEATS],
               "scoring": SCORING, "train_sha256": train_sha}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def find_existing(full_hash):
    if not RUNS_DIR.exists():
        return None
    for meta in sorted(RUNS_DIR.glob("*/meta.json")):
        if json.loads(meta.read_text(encoding="utf-8")).get("hash") == full_hash:
            return meta.parent
    return None


# ------------------------------------------------------------------ por fold

def fold_importances(est, X_val, y_val, i):
    """Importância por permutação, medida no fold de validação (dados que a rede não viu).

    Embaralha cada variável ORIGINAL e mede a queda da métrica principal (f1_macro). Passa por
    todo o pipeline (padronização, reamostragem não atua na predição), então é a importância
    da variável para a rede inteira.
    """
    repeat, fold = divmod(i, N_SPLITS)
    perm = permutation_importance(est, X_val, y_val, scoring=SCORERS[PRIMARY_METRIC],
                                  n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
    return [{"repeat": repeat, "fold": fold, "feature": str(f), "importance": v}
            for f, v in zip(X_val.columns, perm.importances_mean)]


def fold_network(est):
    """Iterações usadas e se o treino terminou antes de max_iter (convergiu). NaN fora do MLP."""
    m = mlp_of(est)
    if m is None:
        return {"n_iter": np.nan, "convergiu": np.nan}
    return {"n_iter": int(m.n_iter_), "convergiu": int(m.n_iter_ < m.max_iter)}


# ------------------------------------------------------------------ artefatos do modelo final

def network_stats(model):
    """Resumo do modelo final: arquitetura, nº de parâmetros, convergência, perda final."""
    mlp = model if isinstance(model, MLPClassifier) else None
    if mlp is None:
        return {"tipo": type(model).__name__}
    n_params = sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    stats = {"tipo": "MLPClassifier",
             "camadas_ocultas": [int(n) for n in np.atleast_1d(mlp.hidden_layer_sizes)],
             "n_parametros": int(n_params), "n_iter": int(mlp.n_iter_),
             "max_iter": int(mlp.max_iter), "convergiu": bool(mlp.n_iter_ < mlp.max_iter),
             "loss_final": float(mlp.loss_)}
    if mlp.early_stopping:
        stats["melhor_validacao_interna"] = float(mlp.best_validation_score_)
    return stats


def plot_loss_curve(model, out):
    """Perda de treino por época (e a nota da validação interna, se houver early_stopping)."""
    curve = getattr(model, "loss_curve_", None)  # o lbfgs não guarda a curva
    if not curve:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve, color="#4C72B0", label="loss de treino")
    ax.set_xlabel("época")
    ax.set_ylabel("loss (entropia cruzada)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    if model.early_stopping and getattr(model, "validation_scores_", None):
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, color="#DD8452", label="validação interna")
        ax2.set_ylabel("acurácia na validação interna")
    ax.set_title("Curva de perda da rede final (treino inteiro)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_model(final_pipe, out_dir, fold_imp):
    """Grava o pipeline treinado e o que dá para ler dele."""
    joblib.dump(final_pipe, out_dir / "pipeline.joblib", compress=3)
    model = model_of(final_pipe)
    (out_dir / "network_stats.json").write_text(json.dumps(network_stats(model), indent=2),
                                                encoding="utf-8")
    perm = (fold_imp.groupby("feature")["importance"].agg(["mean", "std"])
            .sort_values("mean", ascending=False).reset_index())
    perm.to_csv(out_dir / "importances_permutation.csv", index=False)
    fold_imp.to_csv(out_dir / "fold_importances.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.45 * len(perm))))
    p = perm.iloc[::-1]
    ax.barh(p["feature"], p["mean"], xerr=p["std"], color="#DD8452")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(f"Permutação (queda do {PRIMARY_METRIC}), média dos {N_SPLITS * N_REPEATS} folds")
    fig.tight_layout()
    fig.savefig(out_dir / "importances.png", dpi=120)
    plt.close(fig)
    plot_loss_curve(model, out_dir / "loss_curve.png")


# ------------------------------------------------------------------ execução

def run(name, cfg, force=False, n_jobs=-1):
    train_sha = file_sha256(TRAIN_CSV)
    full_hash = config_hash(cfg, train_sha)
    existing = find_existing(full_hash)
    if existing and not force:
        print(f"[{name}] já executado em {existing.name}, pulando (use --force para repetir)")
        return existing

    X, y = apply_target(cfg, *load_train())   # agrupamento de classes (R5), se houver
    pipe = build_pipeline(cfg)
    t0 = time.time()
    # Threads em vez de processos: no Python 3.14 o joblib (loky) quebra quando a pasta
    # temporária tem acento ("Máquina"). Com ~170 linhas, threads custam o mesmo.
    with parallel_config(backend="threading"):
        res = cross_validate(pipe, X, y, cv=CV, scoring=SCORERS, n_jobs=n_jobs,
                             return_estimator=True, return_indices=True)
        val_idx = res["indices"]["test"]
        imp_rows = Parallel(n_jobs=n_jobs)(
            delayed(fold_importances)(est, X.iloc[val], y.iloc[val], i)
            for i, (est, val) in enumerate(zip(res["estimator"], val_idx)))
    elapsed = time.time() - t0
    fold_imp = pd.DataFrame([r for rows in imp_rows for r in rows])
    net = pd.DataFrame([fold_network(est) for est in res["estimator"]])

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{full_hash[:6]}_{name}"
    final_dir = RUNS_DIR / run_id
    tmp_dir = RUNS_DIR / f".tmp_{run_id}"
    tmp_dir.mkdir(parents=True)
    try:
        (tmp_dir / "config.yaml").write_text(
            yaml.safe_dump({"name": name, **cfg}, sort_keys=False, allow_unicode=True),
            encoding="utf-8")

        metrics = {m: {"mean": float(np.nanmean(res[f"test_{m}"])),
                       "std": float(np.nanstd(res[f"test_{m}"]))} for m in SCORING}
        (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        n = len(res["fit_time"])
        folds = pd.DataFrame({"repeat": np.arange(n) // N_SPLITS, "fold": np.arange(n) % N_SPLITS,
                              **{m: res[f"test_{m}"] for m in SCORING},
                              **net, "fit_time": res["fit_time"]})
        folds.to_csv(tmp_dir / "fold_scores.csv", index=False)

        # Rede final: mesmo pipeline treinado no treino inteiro (as do CV são descartadas)
        final_pipe = build_pipeline(cfg).fit(X, y)
        save_model(final_pipe, tmp_dir, fold_imp)

        meta = {"run_id": run_id, "name": name, "hash": full_hash,
                "date": datetime.now().isoformat(timespec="seconds"),
                "train_sha256": train_sha, "n_train": int(len(X)), "seed": SEED,
                "target": target_name(cfg),
                "cv": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})",
                "elapsed_s": round(elapsed, 2),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "imblearn": imblearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__}}
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Só vira uma execução "válida" depois de tudo gravado
        for tentativa in range(3):
            try:
                tmp_dir.rename(final_dir)
                break
            except PermissionError:  # Windows: antivírus/indexador segurando a pasta
                if tentativa == 2:
                    raise
                time.sleep(1)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    show = ["f1_macro", "balanced_accuracy", "accuracy", "log_loss"]
    summary = " | ".join(f"{m}={metrics[m]['mean']:.3f}±{metrics[m]['std']:.3f}" for m in show)
    conv = net["convergiu"].mean()
    tail = "" if np.isnan(conv) else f" | convergiu em {conv:.0%} dos folds"
    print(f"[{name}] {summary}{tail} ({elapsed:.0f}s)  ->  {run_id}")
    return final_dir
