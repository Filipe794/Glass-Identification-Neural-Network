"""config (dict) -> Pipeline do imblearn."""
import copy

from imblearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from src.registry import OPTIONAL_DEFAULT, OPTIONAL_STEPS, REGISTRY, STEP_ORDER


class ConfigError(ValueError):
    pass


def normalize_step(value):
    """'standard' -> {'name': 'standard'};  {'name': 'smote', 'k_neighbors': 2} fica igual."""
    if value is None:
        return {"name": "none"}
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, dict) and "name" in value:
        return dict(value)
    raise ConfigError(f"Formato inválido: {value!r}. Use 'nome' ou {{name: nome, ...}}")


def _canon(v):
    """Lista e tupla valem o mesmo ([16] = (16,)); bool não se mistura com número."""
    if isinstance(v, (list, tuple)):
        return tuple(_canon(i) for i in v)
    return v


def _same(a, b):
    a, b = _canon(a), _canon(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def normalize_model(value):
    """O model é escrito como {name: mlp, hidden_layer_sizes: [16], alpha: 0.001, ...}.

    Sem `name`, vale mlp. O nome sempre entra na config normalizada (e no hash).
    Parâmetros iguais ao padrão do modelo (ex.: max_iter: 2000, activation: relu) saem da
    config: a mesma rede tem sempre o mesmo hash, escrita com ou sem eles, e o runner não a
    executa duas vezes.
    """
    if value is None:
        value = {}
    if isinstance(value, str):
        value = {"name": value}
    if not isinstance(value, dict):
        raise ConfigError(f"model inválido: {value!r}. Use {{name: mlp, ...}}")
    name = value.get("name", "mlp")
    if name not in REGISTRY["model"]:
        raise ConfigError(f"model '{name}' não existe, opções: {', '.join(REGISTRY['model'])}")
    defaults = REGISTRY["model"][name]().get_params()
    # Tupla vira lista: o YAML só tem listas, e assim a config resolvida é sempre a mesma
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in value.items()
              if k != "name" and not (k in defaults and _same(defaults[k], v))}
    return {"name": name, **params}


def resolve(baseline, override=None):
    """Junta baseline + override. O override SUBSTITUI a etapa inteira."""
    cfg = copy.deepcopy(baseline)
    for key, value in (override or {}).items():
        cfg[key] = copy.deepcopy(value)
    valid = STEP_ORDER + ["model"] + OPTIONAL_STEPS
    unknown = [k for k in cfg if k not in valid]
    if unknown:
        raise ConfigError(f"Etapa '{unknown[0]}' não existe. Etapas: {', '.join(valid)}")
    missing = [s for s in STEP_ORDER + ["model"] if s not in cfg]
    if missing:
        raise ConfigError(f"Config sem as etapas: {', '.join(missing)}")
    # Opcionais só entram na config se escritas (e diferentes do padrão)
    for step in STEP_ORDER + [s for s in OPTIONAL_STEPS if s in cfg]:
        cfg[step] = normalize_step(cfg[step])
    cfg["model"] = normalize_model(cfg["model"])
    # Opcional com o valor padrão (ou none) = não escrita: o mesmo hash de não ter a chave
    for step in OPTIONAL_STEPS:
        if step in cfg and cfg[step]["name"] in ("none", OPTIONAL_DEFAULT[step]):
            if len(cfg[step]) > 1:
                raise ConfigError(f"{step} '{cfg[step]['name']}' não aceita parâmetros: {cfg[step]}")
            del cfg[step]
    # Ordem estável das chaves (etapas, model, opcionais): o hash não depende da ordem do YAML
    order = STEP_ORDER + ["model"] + OPTIONAL_STEPS
    return {k: cfg[k] for k in order if k in cfg}


def _make(step, spec, *args):
    """Constrói o passo pelo REGISTRY; devolve None para 'none'."""
    params = dict(spec)
    name = params.pop("name")
    options = REGISTRY[step]
    if name not in options:
        raise ConfigError(f"{step} '{name}' não existe, opções: {', '.join(options)}")
    factory = options[name]
    if factory is None:
        if params:
            raise ConfigError(f"{step} 'none' não aceita parâmetros: {params}")
        return None
    try:
        return factory(*args, **params)
    except TypeError as e:
        raise ConfigError(f"{step} '{name}': parâmetros inválidos {params} ({e})") from e


# ------------------------------------------------------------------ alvo (agrupamento de classes)

def target_name(cfg):
    return cfg.get("target", {"name": OPTIONAL_DEFAULT["target"]})["name"]


def apply_target(cfg, X, y):
    """Aplica o alvo da config (padrão: as 6 classes) a (X, y). Devolve (X, y)."""
    spec = cfg.get("target", {"name": OPTIONAL_DEFAULT["target"]})
    name = spec["name"]
    if name not in REGISTRY["target"]:
        raise ConfigError(f"target '{name}' não existe, opções: {', '.join(REGISTRY['target'])}")
    if len(spec) > 1:
        raise ConfigError(f"target '{name}' não aceita parâmetros: {spec}")
    return REGISTRY["target"][name](X, y)


# ------------------------------------------------------------------ pipeline

def build_pipeline(cfg):
    target = target_name(cfg)  # valida o nome
    if target not in REGISTRY["target"]:
        raise ConfigError(f"target '{target}' não existe, opções: {', '.join(REGISTRY['target'])}")
    steps = [(s, obj) for s in STEP_ORDER if (obj := _make(s, cfg[s])) is not None]
    if cfg["balancing"]["name"] != "none" and cfg["model"].get("early_stopping"):
        # O early_stopping do MLP separa uma parte do treino para validar. Depois da
        # reamostragem, cópias da mesma amostra caem nos dois lados: a validação fica otimista.
        raise ConfigError("balancing com early_stopping: a reamostragem duplica amostras "
                          "antes do MLP separar a validação interna (vazamento). "
                          "Use early_stopping: false, ou balancing: none")
    model = _make("model", cfg["model"])
    # Hiperparâmetro com nome errado só apareceria no fit: confere já
    try:
        model.set_params(**{k: v for k, v in cfg["model"].items() if k != "name"})
        model._validate_params()
    except (ValueError, TypeError) as e:
        raise ConfigError(f"model: hiperparâmetro inválido ({e})") from e
    return Pipeline(steps + [("model", model)])


def model_of(fitted):
    """O modelo (último passo) de um pipeline treinado."""
    return fitted.named_steps["model"]


def mlp_of(fitted):
    """O MLPClassifier de um pipeline treinado, ou None se o modelo for outro (baselines)."""
    model = model_of(fitted)
    return model if isinstance(model, MLPClassifier) else None
