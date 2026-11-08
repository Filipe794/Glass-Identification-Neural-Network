# Redes neurais para identificação de vidros (Glass Identification)

Trabalho da disciplina Aprendizado de Máquina, do PPGCC - UFPI, 2026.2.

**Autores:** Filipe Mendes, Bétanny Alexandra, Juciane Maciel · **Orientação/professor:** Prof. Dr. Vinicius Ponte Machado.

**Relatório do trabalho:** *Redes Neurais para Identificação de Tipos de Vidro utilizando a Base Glass Identification* (PDF — a ser incluído ao final do trabalho)

Descreverá a metodologia, os experimentos por rodada, o modelo escolhido, o resultado no conjunto de teste e a análise de robustez.

## Objetivo

Construir e avaliar uma rede neural do tipo perceptron multicamadas (MLP) que identifique o tipo de vidro a partir da sua composição química e do índice de refração, em um cenário de poucos dados e classes muito desbalanceadas. O trabalho compara, de forma controlada, o efeito da padronização das variáveis, do tamanho e da regularização da rede, do tratamento do desbalanceamento (com reamostragem feita apenas dentro de cada fold) e do agrupamento de classes, sempre com validação cruzada estratificada e comparação explícita com um baseline de classe majoritária.

A análise dos baselines publicados na página da UCI e o plano de treinamento estão em [`docs/baseline-e-plano-de-treinamento.md`](docs/baseline-e-plano-de-treinamento.md).

A pergunta de fundo é **como uma rede neural se comporta com pouquíssimos dados**: 213 amostras (depois de remover a duplicata), com classes de 9 a 76 exemplos. Por isso, além do desempenho, o trabalho avalia:

- **Convergência:** a rede termina o treino antes de `max_iter`? Quantas iterações usa?
- **Classes raras:** o ganho de uma técnica vem das classes grandes ou das pequenas (3, 5 e 6)? A matriz de confusão e o `balanced_accuracy` respondem; a acurácia sozinha não.
- **Robustez:** o resultado depende da inicialização aleatória dos pesos?
- **Estabilidade:** a importância das variáveis se mantém entre folds e sementes?

## Base de dados

**Glass Identification** (UCI Machine Learning Repository, id 42): 214 amostras de vidro descritas por 9 variáveis contínuas — índice de refração (RI) e percentual em peso dos óxidos de sódio (Na), magnésio (Mg), alumínio (Al), silício (Si), potássio (K), cálcio (Ca), bário (Ba) e ferro (Fe) —, além de uma coluna identificadora (`Id_number`) que não deve ser usada como variável. O alvo (`Type`) tem 7 categorias previstas, mas apenas 6 presentes:

| Classe | Descrição | Amostras |
|---|---|---|
| 1 | Janela de edificação, processo float | 70 |
| 2 | Janela de edificação, não float | 76 |
| 3 | Janela de veículo, processo float | 17 |
| 4 | Janela de veículo, não float | 0 (ausente) |
| 5 | Recipientes | 13 |
| 6 | Utensílios de mesa | 9 |
| 7 | Faróis | 29 |

A motivação original é forense: o vidro deixado na cena de um crime pode servir de evidência, se identificado corretamente.

**Artigo original:** Evett, I. W.; Spiehler, E. J. *Rule induction in forensic science*. Technical Report, Central Research Establishment, Home Office Forensic Science Service, Aldermaston, Reading, Berkshire, 1987.

**Citação da base:** German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P

**Disponível em:** https://archive.ics.uci.edu/dataset/42/glass+identification

O arquivo usado está em `data/raw/`, sem alterações (instruções em [`data/raw/LEIA-ME.md`](data/raw/LEIA-ME.md)). A remoção da coluna de identificação e de duplicatas é feita na etapa de preparação dos dados (`python -m src.make_split`), **antes** de dividir em treino e teste.

## Licença

MIT. A base de dados segue os termos da fonte original (CC BY 4.0).

---

# MLP em poucas palavras

Uma rede MLP é uma sequência de camadas de neurônios. Cada neurônio faz uma soma ponderada das
entradas e passa o resultado por uma função de ativação (`relu`, `tanh`...). Na saída, uma
função *softmax* transforma os valores em probabilidades das 6 classes, e o treino ajusta os
pesos para reduzir a entropia cruzada (*loss*), por retropropagação do erro. O que torna esta
base difícil para uma rede:

1. **Escala:** RI fica perto de 1,5 e Si perto de 72. Sem padronização, as variáveis de maior
   magnitude dominam o gradiente (por isso o R1).
2. **Poucos dados:** com ~170 amostras de treino, redes grandes decoram o treino (R2 e R3).
3. **Classes raras:** a classe 6 tem 9 amostras; com o treino do fold, ficam ~5. A acurácia
   esconde esse problema (R4 e a matriz de confusão).
4. **Aleatoriedade:** o resultado depende da inicialização dos pesos (R10).

Mais detalhes em [`docs/fundamentacao-mlp.md`](docs/fundamentacao-mlp.md).

---

# Como usar o projeto

O projeto é um sistema de experimentos em que cada método de tratamento é uma peça
intercambiável e cada experimento é só uma configuração no YAML. Nenhum resultado é
sobrescrito: cada execução ganha uma pasta própria. É o mesmo sistema dos projetos
DecisionTree e RandomForest.

## 1. Instalação

Requer **Python 3.14**. O `requirements.txt` é o mesmo do projeto RandomForest e fixa as
versões: o `pipeline.joblib` de cada execução só abre com as mesmas versões do scikit-learn e do
imbalanced-learn.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

**Dados:** coloque o `glass.data` em `data/raw/` (ver [`data/raw/LEIA-ME.md`](data/raw/LEIA-ME.md))
e prepare o split, uma única vez:

```bash
python -m src.make_split
```

O script remove o `Id_number` e as duplicatas, divide 80/20 de forma estratificada
(`TEST_SIZE`, `SEED = 42`) e imprime a quantidade de amostras por classe no treino e no teste.
**Não regere o split depois de rodar experimentos**: mudaria o sha256 do `train.csv` e as
execuções antigas deixariam de ser comparáveis.

**Paralelismo:** a validação cruzada roda em paralelo com threads
(`parallel_config(backend="threading")`), como nos projetos anteriores: o backend padrão do
joblib quebra no Python 3.14 quando o caminho tem acento ("Máquina").

No Colab, monte o Drive, entre na pasta do projeto e rode os mesmos comandos com `!`:

```python
%cd "/content/drive/MyDrive/Mestrado/Aprendizado de Máquina/Glass-Identification-Neural-Network"
!pip install -r requirements.txt
!python -m src.make_split
!python run_experiments.py experiments/r1_padronizacao.yaml
```

Todos os comandos devem ser executados **a partir da raiz do projeto**.

## 2. Rotina de uso

```bash
# 0. referências (uma vez)
python run_experiments.py experiments/r0_baselines.yaml

# 1. atualizar o baseline da rodada (linhas "# ATUALIZAR") e executar
python run_experiments.py experiments/r1_padronizacao.yaml

# 2. comparar as execuções
python summarize.py                                  # tabela geral, ordenada por f1_macro
python compare.py --rodada R1_ --salvar              # fecha a rodada (teste t corrigido)

# 3. no fim, avaliar a melhor no teste (uma vez)
python final_test.py --run <pasta da execução escolhida em results/runs/>

# 4. análises do vencedor (depois do teste final)
python mlp_vs_baselines.py --mlp <vencedor> --baselines <R0_dummy_majoritaria> <R0_logreg_padronizada>
python run_experiments.py experiments/r10_robustez.yaml                # outras sementes
python stability.py --run <vencedor> --seeds R10_
```

Opções do `run_experiments.py`:

| Opção | Efeito |
|---|---|
| `--only R2_solver` | roda só esses experimentos (o nome de um sweep roda todos os seus valores) |
| `--force` | roda de novo mesmo se já executado (cria pasta nova, não substitui) |
| `--n-jobs 1` | número de threads da validação cruzada (padrão: todos os núcleos) |

**Tempo:** cada execução treina 50 redes (5 folds × 10 repetições) mais 50 cálculos de
importância. Num ambiente de teste, 10 folds levaram cerca de 8 s com a rede de 16 neurônios,
então esperam-se algumas dezenas de segundos por execução no `adam`; o `lbfgs` é bem mais
rápido. As rodadas mais longas são a R2 (13 execuções) e a R3 (12 execuções).

## 3. Estrutura de pastas

```
Glass-Identification-Neural-Network/
├── data/
│   ├── raw/glass.data                    nunca alterar (ver data/raw/LEIA-ME.md)
│   └── split/train.csv, test.csv         gerados por src/make_split.py
├── src/
│   ├── config.py        caminhos, SEED, CV (5 folds x 10 repetições), métricas, ALPHA
│   ├── data.py          load_train(), load_test()
│   ├── make_split.py    prepara os dados e gera o split (descarta Id e duplicatas)
│   ├── steps/           um arquivo por etapa
│   │   ├── scalers.py       standard, robust, minmax
│   │   ├── selection.py     kbest (não usada nas rodadas)
│   │   ├── balancing.py     random_over, smote
│   │   ├── models.py        mlp, logreg, dummy
│   │   └── targets.py       six_classes, window_vs_other, float_vs_nonfloat
│   ├── registry.py      catálogo: nome no YAML -> função que constrói o passo
│   ├── builder.py       config -> Pipeline
│   ├── evaluate.py      métricas, relatório por classe e matriz de confusão
│   └── runner.py        validação cruzada + importâncias por fold + gravação dos resultados
├── experiments/
│   ├── r0_baselines.yaml ... r6_combinacao.yaml     uma rodada por arquivo (seção 7)
│   └── r10_robustez.yaml                            outras sementes (depois do teste)
├── docs/
│   ├── baseline-e-plano-de-treinamento.md   baselines da UCI e plano de treinamento
│   ├── fundamentacao-mlp.md                 a rede neural e as escolhas do projeto
│   └── anotacoes_artigo.md                  decisões, achados e limitações (para o relatório)
├── results/
│   ├── runs/             uma pasta por execução (só se acrescenta)
│   ├── final/            avaliações no teste
│   ├── comparacoes/      tabelas do compare.py --salvar
│   ├── mlp_vs_baselines/ rede x referências (R9)
│   ├── stability/        robustez e estabilidade (R10)
│   └── summary.csv       tabela gerada pelo summarize.py
├── run_experiments.py
├── summarize.py
├── compare.py
├── final_test.py
├── mlp_vs_baselines.py
├── stability.py
├── requirements.txt
└── CLAUDE.md          instruções para o Claude Code
```

No dia a dia só se mexe em dois lugares: os YAMLs das rodadas em `experiments/` (para usar
métodos) e `src/steps/` + `src/registry.py` (para criar métodos novos).

## 4. O pipeline

Toda execução passa pelas mesmas etapas, sempre nesta ordem. Cada etapa recebe uma opção,
e `none` pula a etapa.

| # | Etapa | Opções | Observação |
|---|---|---|---|
| 1 | `scaler` | `none`, `standard`, `robust`, `minmax` | aprende no treino de cada fold (anti-vazamento) |
| 2 | `selection` | `none`, `kbest` | não usada nas rodadas (são só 9 variáveis) |
| 3 | `balancing` | `none`, `random_over`, `smote` | reamostra só o treino do fold; `smote` tem `k_neighbors: 3` por padrão |
| 4 | `model` | `mlp`, `logreg`, `dummy` + hiperparâmetros | `random_state` fixado em 42 (o R10 troca) |
| — | `target` | `six_classes`, `window_vs_other`, `float_vs_nonfloat` | **opcional**: agrupamento de classes (R5) |

O pipeline é o do imblearn: a reamostragem só atua no `fit`, então o fold de validação nunca é
reamostrado. O `target` não é um passo do pipeline: define o problema e é aplicado a `X` e `y`
antes da validação cruzada (e ao teste, no `final_test.py`).

### Alvos (`target`)

| Opção | Problema | Amostras |
|---|---|---|
| `six_classes` (padrão) | as 6 classes originais | todas |
| `window_vs_other` | 1 = janela (classes 1, 2, 3) x 0 = outros (5, 6, 7) | todas |
| `float_vs_nonfloat` | 1 = float (classes 1 e 3) x 0 = não float (classe 2) | só janelas (descarta 5, 6 e 7); é a tarefa do estudo original citado na UCI |

As notas de alvos diferentes **não são comparáveis**: o `compare.py` recusa comparar execuções
com `target` diferente.

### Modelos

| Opção | O que é |
|---|---|
| `mlp` | `MLPClassifier` do scikit-learn: saída softmax, entropia cruzada, `max_iter: 2000` |
| `logreg` | `LogisticRegression` multinomial, o "modelo simples" de referência |
| `dummy` | `DummyClassifier`: `most_frequent` (piso do projeto) ou `stratified` |

Hiperparâmetros do `mlp` usados nas rodadas:

| Parâmetro | O que controla | Padrão do projeto |
|---|---|---|
| `hidden_layer_sizes` | neurônios por camada oculta (`[16]`, `[16, 8]`) | `[100]` (o ponto de partida das rodadas é `[16]`) |
| `activation` | `relu`, `tanh`, `logistic` | `relu` |
| `solver` | `adam` ou `lbfgs` | `adam` |
| `alpha` | regularização L2 | `0.0001` (o ponto de partida das rodadas é `0.001`) |
| `early_stopping` | separa 10% do treino para parar quando a validação para de melhorar | `false` |
| `learning_rate_init` | passo inicial do otimizador | `0.001` |
| `max_iter` | limite de épocas | `2000` |
| `random_state` | semente da rede | `42` |

> Com ~136 amostras no treino de cada fold, o `batch_size` padrão (`min(200, n)`) é o treino
> inteiro: cada época é **uma** atualização dos pesos. Por isso a rede precisa de muitas épocas
> (centenas a mais de mil) e o `max_iter` fica em 2000.

**Parâmetros iguais ao padrão saem da config.** `{hidden_layer_sizes: [100], max_iter: 2000}`
é gravado como `{name: mlp}`. Assim a mesma rede tem sempre o mesmo hash, escrita com ou sem os
padrões, e o runner não a executa duas vezes.

**Combinação recusada:** `balancing` diferente de `none` com `early_stopping: true`. A
reamostragem duplicaria amostras antes de o MLP separar a validação interna, e a nota dessa
validação ficaria otimista. O `run_experiments.py` falha antes de rodar qualquer coisa, com a
explicação.

## 5. Decisões do projeto

As justificativas e os achados de cada rodada vão em
[`docs/anotacoes_artigo.md`](docs/anotacoes_artigo.md).

- **Preparação antes da divisão:** `Id_number` descartado (o arquivo é ordenado por classe e o Id
  vazaria o rótulo) e duplicatas removidas, para a mesma amostra não cair no treino e no teste.
- **Padronização e reamostragem dentro do pipeline:** nunca na base inteira.
- **Métrica principal: `f1_macro`.** Todas as classes pesam igual, inclusive as de 9 a 17
  amostras. A acurácia não serve sozinha: prever sempre a classe 2 já acerta cerca de 35%.
  Acompanham `balanced_accuracy`, `precision_macro`, `recall_macro`, `roc_auc_ovr` (AUC
  um-contra-todos) e `log_loss` (menor é melhor).
- **Baseline:** classe majoritária, no mesmo protocolo (R0). Um modelo que não o supera em
  `balanced_accuracy` e `f1_macro` não aprendeu nada útil. A regressão logística padronizada
  entra como "modelo simples".
- **Regra de escolha:** entre as configs sem diferença significativa para a melhor
  (`compare.py`), fica a mais simples.
- **Sem remoção global de outliers:** nesta base eles caracterizam as classes raras (ex.: Ba alto
  nos faróis).
- **Valores iniciais a confirmar** antes da primeira rodada (em `src/config.py` e nos YAMLs):
  teste de 20% (`TEST_SIZE = 0.2`), CV de 5 folds × 10 repetições e rede de partida com 1
  camada de 16 neurônios e `alpha 0.001`. Mudar qualquer um depois de rodar invalida as
  comparações.

## 6. Escrevendo experimentos

O formato do YAML é o mesmo dos projetos anteriores; o `model` tem `name`, e o `hidden_layer_sizes`
é uma lista:

```yaml
baseline_name: R1_baseline
baseline:
  scaler: standard
  selection: none
  balancing: none
  model: {name: mlp, hidden_layer_sizes: [16], alpha: 0.001}

experiments:
  - name: R1_padronizacao
    sweep: {scaler.name: [none, standard, robust, minmax]}

  - name: R4_smote_k
    override: {balancing: smote}
    sweep: {balancing.k_neighbors: [2, 4]}

  - name: R5_janela_vs_outros
    override: {target: window_vs_other}

  - name: R3_regularizacao
    sweep:
      model.alpha: [0.001, 0.01, 0.1]
      model.early_stopping: [false, true]
```

- O `sweep` gera um experimento por valor. A chave é `etapa.parametro`; com mais de uma
  chave, roda todas as combinações. O nome mostra o parâmetro
  (`R3_regularizacao__alpha=0.1_early_stopping=True`) ou, quando o sweep troca a opção, a etapa
  (`R1_padronizacao__scaler=robust`). Valores em lista aparecem sem colchetes
  (`R2_neuronios__hidden_layer_sizes=16x8`).
- Configs já executadas (mesmo hash) são puladas.
- O `override` **substitui a etapa inteira**, inclusive o `model`: para mudar só `alpha`, repita
  `hidden_layer_sizes` e os demais vencedores.
- O baseline roda automaticamente com o nome de `baseline_name`. Do R2 em diante ele é igual ao
  vencedor da rodada anterior e é pulado pelo hash. Por isso o `compare.py --rodada` precisa de
  `--incluir <vencedor anterior>`.
- Do R2 em diante, os valores marcados com `# ATUALIZAR: vencedor de Rx` são provisórios:
  **não rode uma rodada antes de atualizar o baseline dela**.
- Antes de rodar qualquer coisa, o script valida todos os experimentos (nomes de opções,
  hiperparâmetros do modelo, combinações recusadas). Um erro de digitação para tudo com uma
  mensagem clara.

## 7. Rodadas

Cada rodada responde a uma pergunta e muda uma coisa só. O vencedor de uma vira o baseline
da seguinte. Cada YAML traz no cabeçalho a pergunta, o critério e o comando para fechar a
rodada.

| Rodada | Pergunta | O que varia | Critério |
|---|---|---|---|
| **R0** `r0_baselines` | O que se consegue sem a rede? | dummy (`most_frequent`, `stratified`), regressão logística (com e sem `class_weight`) | não há escolha; servem de referência |
| **R1** `r1_padronizacao` | Quanto a escala das variáveis atrapalha? | `scaler`: `none`, `standard`, `robust`, `minmax` | `f1_macro`; a mais simples sem diferença significativa |
| **R2** `r2_tamanho_rede` | Quantos neurônios? Que ativação e otimizador? | `hidden_layer_sizes` (4 a 64, e duas camadas), `activation`, `solver` | `f1_macro`; a **menor** rede sem diferença significativa |
| **R3** `r3_regularizacao` | Quanta regularização? O early stopping ajuda? | `alpha`: 1e-4 a 10 × `early_stopping` (com e sem) | `f1_macro`; o **maior** `alpha` sem diferença significativa |
| **R4** `r4_desbalanceamento` | Reamostrar as classes raras ajuda? | `balancing`: `none`, `random_over`, `smote`; `k_neighbors` 2 e 4 | `f1_macro` e `balanced_accuracy` |
| **R5** `r5_agrupamento` | E se o problema virar binário? | `target`: `window_vs_other`, `float_vs_nonfloat` | **exploratória**: não escolhe vencedor |
| **R6** `r6_combinacao` | Os vencedores funcionam juntos? | todos os vencedores + ablação (tirar um de cada vez) | o critério da rodada que decidiu cada ponto |

Depois do R6:

| Rodada | O que é |
|---|---|
| **R7** Escolha | registrar a config final e a justificativa de cada rodada em `docs/anotacoes_artigo.md` |
| **R8** Teste final | `final_test.py` no vencedor, **uma única vez** |
| **R9** Rede x referências | `mlp_vs_baselines.py`: CV fold a fold e teste, lado a lado, contra o dummy e a regressão logística |
| **R10** Robustez e estabilidade | `r10_robustez.yaml` (o vencedor com 5 outras sementes) e `stability.py` |

### Hipóteses a discutir no relatório

- **R1:** sem padronização, a rede tende a ter desempenho pior e treinar pior (mais iterações ou
  sem convergir), porque RI e Si estão em escalas muito diferentes. A UCI mostra baselines de rede
  abaixo da classe majoritária; o plano atribui isso à falta de padronização (hipótese, não
  fato documentado).
- **R2:** com ~170 amostras, redes menores devem empatar com as maiores; uma segunda camada não
  deve compensar os parâmetros a mais.
- **R3:** a regularização L2 deve ajudar, até certo ponto; o `early_stopping` pode atrapalhar
  porque tira amostras das classes raras do treino.
- **R4:** o ganho da reamostragem, se houver, deve aparecer nas classes 3, 5 e 6 (matriz de
  confusão e `balanced_accuracy`), não na acurácia.
- **R5:** o problema binário é mais fácil e menos desbalanceado (no `float_vs_nonfloat`, 87 x 76);
  contrastar mostra quanto do erro vem do número de classes e do tamanho delas.
- **R10:** a variação entre sementes mede o quanto da diferença entre configurações é só ruído
  de inicialização.

## 8. Onde ficam os resultados

Cada execução cria uma pasta nova em `results/runs/`:

```
results/runs/20261112_142558_0e1f8f_R2_neuronios__hidden_layer_sizes=16x8/
├── config.yaml                   config completa usada (padrões do modelo omitidos)
├── metrics.json                  média e desvio de cada métrica na validação cruzada
├── fold_scores.csv               as 50 notas individuais, + n_iter e convergiu de cada rede
├── meta.json                     data, hash, target, versões, hash do train.csv
├── pipeline.joblib               o pipeline treinado (padronização + rede)
├── network_stats.json            arquitetura, nº de parâmetros, iterações, convergência, loss final
├── importances_permutation.csv   importância por permutação: média e desvio nos 50 folds
├── fold_importances.csv          a importância em cada um dos 50 folds
├── importances.png               importância por permutação
└── loss_curve.png                curva de perda da rede final (não gerada com o solver lbfgs)
```

A validação cruzada treina 50 redes só para medir o desempenho, e depois uma **rede final** é
treinada com o `train.csv` inteiro. É ela que fica salva.

**Importância por permutação:** em cada fold, embaralha uma variável **original** no fold de
validação e mede quanto o `f1_macro` cai (5 repetições por variável). É medida em dados que a
rede não viu, e atravessa o pipeline inteiro.

**Convergência:** `convergiu` vale 1 quando o treino terminou antes de `max_iter`. O
`summarize.py` mostra a porcentagem de folds que convergiram e o `compare.py` avisa quando algum
fold não convergiu: nesse caso a nota daquela configuração está limitada pelo `max_iter`, e não
pela rede.

Para usar o pipeline num notebook:

```python
import joblib
import src                      # aplica a configuração do projeto antes de carregar
from src.builder import mlp_of
from src.data import load_train

pipe = joblib.load("results/runs/<pasta>/pipeline.joblib")
rede = mlp_of(pipe)              # o MLPClassifier (None nos baselines)
X, y = load_train()
pipe.predict_proba(X.head())
```

### Tabela comparativa

`python summarize.py` lê todas as pastas de `results/runs/` e gera `results/summary.csv`,
ordenado pela métrica escolhida (`--metric`; no `log_loss`, do menor para o maior). Além das
métricas, traz a porcentagem de folds que convergiram, o nº de parâmetros e o `target`. A coluna
`pipeline` resume a config, por exemplo
`standard > smote(k_neighbors=2) | mlp(hidden_layer_sizes=[16],alpha=0.01)`.

Essa tabela é sempre derivada das pastas: se algo der errado, apague-a e gere de novo.

### Garantias contra sobrescrita

- O runner só **cria** pastas. Ele nunca apaga nem edita nada em `results/runs/`.
- Se a mesma config, com os mesmos dados, já foi executada, ele avisa e pula. Com
  `--force`, cria uma pasta nova ao lado da antiga.
- Enquanto roda, a execução fica numa pasta `.tmp_...`. Ela só ganha o nome definitivo
  quando tudo foi gravado, então uma execução interrompida nunca conta como feita.
- Se você mudar o **código** de um método já existente (não a config), o hash não muda e o
  runner vai pular. Nesse caso, use `--force` (ou crie uma opção nova).

## 9. Comparando execuções

O `compare.py` compara as 50 notas fold a fold com o **teste t corrigido de Nadeau e Bengio
(2003)**, com α = 0,05.

```bash
python compare.py <run_A> <run_B>                                    # duas execuções
python compare.py --rodada R2_ --incluir <vencedor do R1> --salvar   # fecha uma rodada
python compare.py --rodada R4_ --metric balanced_accuracy --incluir <vencedor do R3>
python compare.py --rodada R3_ --metric log_loss --incluir <vencedor do R2>
```

- No modo `--rodada`, a referência é a execução de melhor média entre as do problema de 6
  classes (a maior; no `log_loss`, a menor), ou a passada em `--ref`. A tabela mostra também o
  **nº de parâmetros**, as **iterações médias** e a **% de folds que convergiram**.
- Só compara execuções do **mesmo split e do mesmo `target`**.

p ≥ α quer dizer "sem diferença significativa", não "equivalentes". Cada rodada faz várias
comparações sem correção para comparações múltiplas; p-valores perto de α devem ser lidos
com cautela.

## 10. Teste final

Só o `final_test.py` lê o `test.csv` durante o projeto. Ele carrega o `pipeline.joblib` da
execução indicada, avalia uma vez no teste e grava em `results/final/<run>/`:

| Arquivo | Conteúdo |
|---|---|
| `metrics.json` | todas as métricas no teste |
| `baseline_majoritario.json` | as mesmas métricas do classificador de classe majoritária, no mesmo teste |
| `classification_report.csv` | precisão, recall, F1 e nº de amostras de cada classe |
| `confusion_matrix.csv` / `.png` | matriz de confusão |
| `meta.json` | data, alvo e tamanhos |

Se a execução já foi avaliada, ele apenas mostra o resultado guardado. O teste é pequeno
(20% de 213 amostras; as classes 5 e 6 ficam com 3 e 2): leia as notas por classe como
indicativo.

## 11. Análises do vencedor

### Rede x referências (`mlp_vs_baselines.py`, R9)

```bash
python mlp_vs_baselines.py --mlp <vencedor> --baselines <R0_dummy_majoritaria> <R0_logreg_padronizada>
```

Exige o teste final da rede. Os baselines são execuções do R0 (já têm CV e pipeline) e precisam
ter o mesmo split e o mesmo `target`. Grava em `results/mlp_vs_baselines/<run>/`:

| Arquivo | O que mostra |
|---|---|
| `cv.csv` | as métricas da CV lado a lado, com o teste t corrigido fold a fold (rede − baseline) |
| `teste.csv` | as métricas no teste dos modelos salvos, com as mesmas funções |
| `f1_por_classe.csv` / `.png` | o F1 de cada classe no teste, rede x baselines |

### Robustez e estabilidade (`stability.py`, R10)

```bash
python run_experiments.py experiments/r10_robustez.yaml
python stability.py --run <vencedor> --seeds R10_
```

Antes, atualize o `r10_robustez.yaml` com a config do vencedor. Mede:

- o `f1_macro` da CV do vencedor e de cada semente, com a média, o desvio, o mínimo e o máximo
  entre elas;
- a **concordância do ranking** das variáveis entre as sementes (correlação de Spearman média e
  mínima entre os pares);
- a concordância do ranking entre os **folds** do vencedor e em quantos a variável mais importante
  é a mesma.

Grava em `results/stability/<run>/` as tabelas e um boxplot da importância por fold.

## 12. Criando um método novo

Exemplo: um scaler que aplica `log1p` antes de padronizar.

1. Em `src/steps/scalers.py`:
   ```python
   def log_standard():
       return make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
   ```
2. Em `src/registry.py`, na entrada `"scaler"`:
   ```python
   "log_standard": scalers.log_standard,
   ```
3. No YAML:
   ```yaml
   - name: R1_log
     override: {scaler: log_standard}
   ```

Transformadores que aprendem algo com os dados devem calcular esses valores no `fit` e só
aplicá-los no `transform`. É isso que impede a contaminação com os dados de validação.

## Referências

- Chawla, N. V.; Bowyer, K. W.; Hall, L. O.; Kegelmeyer, W. P. *SMOTE: Synthetic Minority
  Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357, 2002.
- Evett, I. W.; Spiehler, E. J. *Rule induction in forensic science*. Technical Report, Home Office
  Forensic Science Service, 1987.
- German, B. *Glass Identification* [Dataset]. UCI Machine Learning Repository, 1987.
  https://doi.org/10.24432/C5WW2P
- Kingma, D. P.; Ba, J. *Adam: A Method for Stochastic Optimization*. International Conference on
  Learning Representations (ICLR), 2015.
- Lemaître, G.; Nogueira, F.; Aridas, C. K. *Imbalanced-learn: A Python Toolbox to Tackle the Curse
  of Imbalanced Datasets in Machine Learning*. Journal of Machine Learning Research, 18(17), 1–5, 2017.
- Nadeau, C.; Bengio, Y. *Inference for the Generalization Error*. Machine Learning, 52,
  239–281, 2003.
- Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning
  Research, 12, 2825–2830, 2011.
- Rumelhart, D. E.; Hinton, G. E.; Williams, R. J. *Learning representations by back-propagating
  errors*. Nature, 323, 533–536, 1986.
