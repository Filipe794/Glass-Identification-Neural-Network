# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Projeto de mestrado: rede neural (MLP, scikit-learn) na base Glass Identification, com um sistema
declarativo de experimentos. Segue a mesma arquitetura dos projetos `DecisionTree` e
`RandomForest` (Pima Indians Diabetes), adaptada a uma base multiclasse, pequena e desbalanceada.
A documentação de uso para humanos está em `README.md`, em português; mantenha o código, os
comentários e as mensagens em português.

## Comandos

Todos rodam a partir da raiz, com o `.venv` (Python 3.14):

```bash
pip install -r requirements.txt
python -m src.make_split                                                  # data/raw/glass.data -> data/split/ (uma vez)
python run_experiments.py experiments/r0_baselines.yaml                   # uma rodada
python run_experiments.py experiments/r1_padronizacao.yaml --only R1_padronizacao  # um experimento (ou um sweep inteiro)
python run_experiments.py experiments/r1_padronizacao.yaml --force        # repete, cria pasta nova
python summarize.py [--metric balanced_accuracy]                          # regenera results/summary.csv
python compare.py <run_A> <run_B> [--metric f1_macro]                     # teste t corrigido (Nadeau-Bengio) fold a fold
python compare.py --rodada R1_ [--metric f1_macro|log_loss] [--ref <pasta>] [--incluir <pasta> ...] [--salvar]
python final_test.py --run <pasta em results/runs>
# depois do teste final:
python mlp_vs_baselines.py --mlp <vencedor> --baselines <R0_dummy> <R0_logreg> ...
python run_experiments.py experiments/r10_robustez.yaml
python stability.py --run <vencedor> --seeds R10_
```

Não há testes, linter nem build. Para validar uma mudança, rode os experimentos afetados com
`--only`. Evite gravar resultados de teste em `results/` do usuário; use uma cópia do projeto.
Para validar um YAML sem rodar nada, use `load_experiments()` + `build_pipeline()` de
`run_experiments.py` num script temporário.

## Arquitetura

O fluxo é YAML → `resolve` → `build_pipeline` → `cross_validate` → pasta imutável em `results/runs/`.

- **Etapas fixas e ordenadas:** `src/registry.py` define `STEP_ORDER` (scaler → selection →
  balancing), seguida do `model`. Cada etapa recebe uma opção pelo nome em `REGISTRY`; `none`
  pula a etapa. O pipeline é o do **imblearn** (`imblearn.pipeline.Pipeline`): o `balancing`
  (reamostragem) só atua no fit, então a validação de cada fold nunca é reamostrada.
- **Model:** escrito como `{name: mlp | logreg | dummy, **hiperparâmetros}` (sem `name`, vale
  `mlp`). As fábricas em `src/steps/models.py` fixam `random_state=SEED` e `max_iter=2000`
  (mlp e logreg). `logreg` e `dummy` são os modelos de referência do R0.
- **`target` (etapa opcional, `OPTIONAL_STEPS`):** não é um passo do pipeline. Define o problema
  (`six_classes`, `window_vs_other`, `float_vs_nonfloat`; ver `src/steps/targets.py`) e é aplicado
  a `(X, y)` por `builder.apply_target()` ANTES da validação cruzada, no `runner`, e também ao
  teste no `final_test.py`. `six_classes` é o padrão e sai da config (mesmo hash de não escrever
  a chave). Alvos diferentes são problemas diferentes: o `compare.py` recusa compará-los.
- **Normalização da config (`src/builder.py:resolve`):** baseline + override (o override
  **substitui a etapa inteira**, inclusive `model`); etapas viram `{name: ..., **params}`;
  parâmetros do model iguais ao padrão da fábrica são removidos (mesma rede = mesmo hash; lista
  e tupla valem o mesmo: `[16]` = `(16,)`); uma etapa opcional com o valor padrão é removida; as
  chaves saem em ordem fixa. `expand_sweep` chama `resolve` de novo em cada combinação e
  formata listas no nome (`hidden_layer_sizes=16x8`).
- **Método novo:** escreva a função-fábrica em `src/steps/<etapa>.py` e acrescente uma linha no
  `REGISTRY`. O builder, o runner e o YAML não mudam. Transformadores que aprendem algo dos dados
  calculam no `fit` (anti-vazamento).
- **Não altere o comportamento de uma opção existente:** o hash não inclui o código, então
  execuções antigas ficariam com o mesmo hash e outro resultado. Crie uma opção nova.
- **Pipeline:** `src/__init__.py` aplica `set_config(transform_output="pandas")` globalmente
  (necessário também ao carregar um `pipeline.joblib`) e silencia o `ConvergenceWarning` (a
  convergência é registrada em `fold_scores.csv`). O config do sklearn é por thread: em
  paralelismo próprio, use `sklearn.utils.parallel.Parallel/delayed`.
- **`builder.model_of()` / `mlp_of()`:** `mlp_of` devolve o `MLPClassifier` de um pipeline
  treinado, ou `None` nos baselines (o runner e o compare tratam o `None` com NaN).
- **Runner (`src/runner.py`):**
  - o hash da config completa + seed + CV + métricas + sha256 do `train.csv` define a identidade
    da execução; um hash já existente é pulado;
  - `cross_validate` com `return_estimator` e `return_indices`; depois, para cada fold,
    importância por permutação (`f1_macro`, `PERM_REPEATS`) no fold de validação →
    `fold_importances.csv`, e `n_iter`/`convergiu` de cada rede → colunas de `fold_scores.csv`;
  - grava em `.tmp_*` e renomeia no fim; nunca apaga nem edita execuções existentes;
  - rede final no treino inteiro: `pipeline.joblib`, `network_stats.json`,
    `importances_permutation.csv`, `importances.png`, `loss_curve.png` (o solver `lbfgs` não
    guarda a curva e não gera o PNG).
- **Métricas:** `SCORERS` e `SCORING` em `src/config.py` são a única lista (accuracy,
  balanced_accuracy, f1_macro, precision_macro, recall_macro, roc_auc_ovr, log_loss;
  `PRIMARY_METRIC = f1_macro`). `log_loss` é função própria; `LOWER_IS_BETTER` diz ao
  `compare.py` e ao `summarize.py` que, nela, menor é melhor. Mudar o `SCORING` muda o hash de
  todas as configs.
- **`compare.py`:** teste t corrigido de Nadeau e Bengio, referência pela maior média (menor no
  `log_loss`) entre as execuções do alvo `six_classes`, colunas de nº de parâmetros, iterações e
  % de folds que convergiram, e aviso quando algum fold não convergiu.
- **`src/evaluate.py`:** `score_all`, `per_class_report`, `confusion` e `plot_confusion`, usados
  pelo `final_test.py` e pelo `mlp_vs_baselines.py`.

## Rodadas de experimentos

Um YAML por rodada em `experiments/`; o vencedor de uma rodada vira o baseline da seguinte.

| Arquivo | Pergunta | Critério |
|---|---|---|
| `r0_baselines.yaml` | piso (dummy) e modelo simples (regressão logística) | não há escolha |
| `r1_padronizacao.yaml` | `scaler` (none, standard, robust, minmax) | `f1_macro`, a mais simples no empate |
| `r2_tamanho_rede.yaml` | `hidden_layer_sizes`, `activation`, `solver` | `f1_macro`, a menor rede no empate |
| `r2b_interacao.yaml` | grid dos 3 melhores `hidden_layer_sizes` x `activation` x `solver` (interações que o R2 não vê) | `f1_macro`, a menor rede no empate; o vencedor é o final do R2 |
| `r3_regularizacao.yaml` | `alpha` x `early_stopping` | `f1_macro`, o maior `alpha` no empate |
| `r4_desbalanceamento.yaml` | `balancing` (none, random_over, smote) e `k_neighbors` | `f1_macro` e `balanced_accuracy` |
| `r5_agrupamento.yaml` | `target` (janela x outros, float x não float); por alvo, também dummy, logreg balanceada e a rede sem reamostragem | exploratória: não escolhe vencedor; comparar por alvo (`--rodada R5_janela` / `R5_float`) |
| `r6_combinacao.yaml` | combinação + ablação | o critério da rodada que decidiu cada ponto |
| `r10_robustez.yaml` | o vencedor com 5 outras sementes (só depois do teste final) | — |

Em todas: entre as configs sem diferença significativa para a melhor (`compare.py`), fica a mais
simples. R7 (escolha), R8 (`final_test.py`, uma vez) e R9 (`mlp_vs_baselines.py`) não têm YAML.

- Do R2 em diante, linhas com `# ATUALIZAR: vencedor de Rx` são provisórias e devem ser trocadas
  pelo usuário antes de rodar. Não preencha vencedores por conta própria.
- O baseline de cada rodada é a mesma config do vencedor anterior, então o runner o pula pelo
  hash: a referência no `compare.py --rodada` vem por `--incluir <vencedor anterior>`.
- O R2b complementa o R2: o R2 varia um fator por vez e o R2b testa a combinação dos 3
  melhores níveis de cada fator. O vencedor do R2b é o vencedor final do R2, então os
  `# ATUALIZAR: vencedor de R2` do R3 em diante passam a valer para o R2b. O prefixo é `R2b_`
  (o `--rodada R2_` não pega essas execuções).

## Restrições importantes

- **O conjunto de teste é isolado:** `load_test()` só é chamado pelo `final_test.py` e, depois
  dele, pelo `mlp_vs_baselines.py` (que confere se o teste final daquela execução existe). Não
  rode nenhum deles sem o usuário pedir.
- **Preparação dos dados (`src/make_split.py`):** descarta a coluna `Id_number` (o arquivo é
  ordenado por classe: o Id vaza o rótulo) e remove duplicatas ANTES de dividir, para a mesma
  amostra não cair no treino e no teste. Não regere `data/split/` depois de rodar experimentos
  (mudaria o sha256 e quebraria a comparação com as execuções antigas).
- **Padronização e reamostragem só dentro do pipeline:** nunca padronize nem reamostre a base
  inteira antes da validação cruzada (vazamento). O scaler aprende no treino de cada fold; o
  imblearn reamostra só o treino de cada fold.
- **`balancing` + `early_stopping: true` é recusado pelo builder:** a reamostragem duplica
  amostras antes de o MLP separar a validação interna, e a nota dessa validação ficaria otimista.
- **SMOTE e a classe 6:** a classe 6 tem 9 amostras (~5 no treino de cada fold). O SMOTE precisa
  de `k_neighbors + 1` amostras por classe; o padrão do projeto é `k_neighbors: 3`.
- **Outliers:** não remover de forma global. Na base, os "outliers" (ex.: Ba alto nos faróis)
  caracterizam as classes raras (ver `docs/baseline-e-plano-de-treinamento.md`).
- **Valores iniciais a confirmar antes da primeira rodada:** `TEST_SIZE = 0.2` e a CV 5x10
  (`N_SPLITS, N_REPEATS`) em `src/config.py`, e o ponto de partida da rede (1 camada de 16
  neurônios, `alpha 0.001`) nas rodadas. Foram propostos a partir do plano em
  `docs/baseline-e-plano-de-treinamento.md`; o usuário decide. Mudar qualquer um depois de
  rodar invalida as comparações.
- **CV fixo** em `src/config.py` (`RepeatedStratifiedKFold`, `SEED=42`). Componentes aleatórios
  devem receber `random_state=SEED`.
- **Paralelismo:** a validação cruzada roda com `parallel_config(backend="threading")`. O backend
  padrão do joblib (loky) quebra no Python 3.14 quando o caminho do projeto tem acento.
- **Classes ausentes e rótulos:** a classe 4 não existe na base. Os rótulos (1, 2, 3, 5, 6, 7)
  são usados como estão; não recodifique para 0..5 (os nomes em `CLASS_NAMES` dependem deles).
