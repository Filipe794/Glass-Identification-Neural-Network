# Anotações para o relatório

Registro de decisões, achados e limitações, para alimentar o relatório. **Preencha cada
seção depois de fechar a rodada correspondente.** Nada aqui é resultado: os campos entre
`[colchetes]` são para preencher com o que os experimentos mostrarem.

## 0. Preparação dos dados

- Base: UCI Glass Identification (id 42), 214 linhas, 9 variáveis + `Id_number` + `Type`.
- `Id_number` descartado (o arquivo é ordenado por classe: o Id vaza o rótulo).
- Duplicatas removidas antes da divisão (a conhecida tem Id diferente, por isso o Id sai antes).
  Amostras após a remoção: 213.
- Classe 4 ausente; rótulos mantidos 1, 2, 3, 5, 6, 7.
- Divisão estratificada: teste 20%, semente 42. 
- Amostras por classe no treino/teste: 

| Classe | Total | Treino | Teste |
|:------:|------:|-------:|------:|
| 1      | 69    | 55     | 14    |
| 2      | 76    | 61     | 15    |
| 3      | 17    | 14     | 3     |
| 5      | 13    | 10     | 3     |
| 6      | 9     | 7      | 2     |
| 7      | 29    | 23     | 6     |
| **Total** | **213** | **170** | **43** |

- Validação cruzada: 5 folds × 10 repetições.
- Limitação: teste muito pequeno (classes 5 e 6 com poucas amostras); notas por classe são indicativas.

### R0 — Modelos de referência

Antes de treinar a rede, fixou-se o que se consegue sem ela, com o mesmo split e a mesma
validação cruzada (5 folds × 10 repetições, estratificada) das rodadas seguintes. Valores:
média ± desvio entre os folds.

| Modelo | Acurácia | Balanced accuracy | F1-macro |
|---|---:|---:|---:|
| Dummy `most_frequent` | 0,359 ± 0,012 | 0,167 ± 0,000 | 0,088 ± 0,002 |
| Dummy `stratified` | 0,265 ± 0,080 | 0,169 ± 0,067 | 0,164 ± 0,066 |
| Regressão logística padronizada | **0,614** ± 0,062 | 0,521 ± 0,088 | 0,513 ± 0,088 |
| Regressão logística padronizada, `class_weight="balanced"` | 0,568 ± 0,075 | **0,666** ± 0,081 | **0,616** ± 0,070 |

- **Piso:** o dummy `most_frequent` prevê sempre a classe 2. Por isso a acurácia (0,359) fica
  próxima da proporção dessa classe, a balanced accuracy é exatamente 1/6 (0,167) e o F1-macro
  é 0,088, valores que coincidem com os esperados. O dummy estratificado fica em ≈ 1/6 nas
  métricas por classe. Um modelo que não supera esses valores em **balanced accuracy e
  F1-macro** não aprendeu nada útil, mesmo com acurácia razoável.
- **Modelo simples:** a regressão logística, sem não linearidade, supera os dois dummies em
  todas as métricas (p < 0,001). É o valor que a rede precisa superar para justificar a
  complexidade adicional.
- **Efeito do peso de classe:** o `class_weight="balanced"` troca acurácia por desempenho nas
  classes raras. A acurácia cai de 0,614 para 0,568 (diferença não significativa, p = 0,157),
  enquanto a balanced accuracy sobe de 0,521 para 0,666 (+0,145; p = 0,004) e o F1-macro de
  0,513 para 0,616 (+0,102; p = 0,032). O MLP do scikit-learn não tem `class_weight`, o que
  motiva a rodada R4 (reamostragem).
- **Métrica de decisão:** como a acurácia favorece o modelo que ignora as classes raras, o
  critério de escolha ao longo do trabalho é o F1-macro, com a balanced accuracy como apoio.
- **Variabilidade:** os desvios da regressão logística (0,06–0,09) são grandes diante das
  diferenças entre as versões, pois as classes raras têm poucas amostras por fold. Isso pede
  cautela ao comparar modelos próximos.

Teste t pareado com correção para validação cruzada, comparando cada modelo com o de maior média em cada métrica.

### R1 — Padronização das variáveis

Variou-se só o scaler, com a rede inicial fixa (1 camada de 16 neurônios, `alpha = 0,001`, 262
parâmetros) e o scaler ajustado dentro do pipeline, no treino de cada fold. Valores: média ± desvio
entre os folds (5 × 10).

| Scaler | Acurácia | Balanced accuracy | F1-macro | p (F1-macro) | Convergiu (%) |
|---|---:|---:|---:|---:|---:|
| `standard` (referência) | 0,674 ± 0,076 | 0,614 ± 0,102 | 0,597 ± 0,110 | — | 20 |
| `robust` | 0,674 ± 0,071 | 0,607 ± 0,099 | 0,590 ± 0,097 | 0,880 | 26 |
| `minmax` | 0,673 ± 0,051 | 0,599 ± 0,081 | 0,577 ± 0,084 | 0,733 | 0 |
| `none` | 0,614 ± 0,080 | 0,499 ± 0,098 | 0,484 ± 0,103 | 0,143 | 20 |

- **Os três scalers são equivalentes.** `standard`, `robust` e `minmax` diferem em menos de 0,02
  de F1-macro (p ≥ 0,73).
- **Sem padronização a nota cai, mas não houve colapso.** O `none` perde 0,113 de F1-macro e
  0,115 de balanced accuracy em relação ao `standard`, consistente nas três métricas, mas a
  diferença não é significativa (p = 0,143 e 0,114). Ainda assim, o `none` fica bem acima dos
  dummies do R0 (F1-macro de 0,088 e 0,164), então a hipótese de colapso em poucas classes não se
  confirmou nesta base.
- **Convergência:** em todas as configurações, a maioria dos folds usou quase todas as 2000
  iterações (média de ≈ 1975–2000) e só 0–26% convergiram, com o `minmax` em 0%. A escala das
  variáveis, portanto, não explica a falta de convergência; o R2 (solver `lbfgs`) e o R3 (`alpha`)
  investigam isso.
- **Escolha: `standard`.** Pelo critério definido antes da rodada (entre as configurações sem
  diferença significativa para a melhor, a mais simples: `none` → `standard` → `minmax` →
  `robust`), o `none` seria o escolhido, já que nenhuma diferença foi significativa (p ≥ 0,11
  em todas as métricas). Optou-se, porém, pelo `standard`, por um desvio consciente do
  critério, pelos motivos abaixo:
  1. **Tamanho do efeito:** o `none` perde 0,113 de F1-macro e 0,115 de balanced accuracy, de
     forma consistente nas três métricas. É uma diferença da mesma ordem que a significativa
     no R0 (0,102; p = 0,032), mas com desvio entre folds de ≈ 0,10 e apenas 50 notas
     correlacionadas, o teste tem pouco poder para detectá-la.
  2. **Fundamento:** o MLP é treinado por gradiente e variáveis em escalas muito diferentes
     (RI ≈ 1,5; Si ≈ 72; Ba e Fe quase sempre 0) distorcem a contribuição de cada entrada e
     pioram a otimização (LeCun et al., 1998). A padronização é a prática recomendada para
     esse tipo de modelo.
  3. **Sem custo:** o `standard` é simples, não tem hiperparâmetros e é ajustado só no treino
     de cada fold, sem vazamento.

  Os scalers `robust` e `minmax` tiveram resultado equivalente ao `standard` (diferenças < 0,02
  de F1-macro; p ≥ 0,73), o que indica que, entre os que padronizam, a escolha não é crítica
  nesta base. O `standard` é o scaler fixo das rodadas seguintes. O desvio do critério fica
  registrado em `docs/anotacoes_artigo.md`.

## R2 — Tamanho da rede, ativação e otimizador

Três sweeps, cada um mudando um fator a partir do baseline (`standard`, 1 camada de 16 neurônios,
`relu`, `adam`, `alpha = 0,001`): `hidden_layer_sizes`, `activation` e `solver`. Valores: média
entre os folds (5 × 10); p do F1-macro contra a melhor média (`[32]`). Tabela gerada por
`python tabela_rodada.py --rodada R2_` (`docs/tabelas/R2.md`).

| Configuração | F1-macro | Bal. acc. | Acurácia | Parâmetros | Convergiu (%) | p (F1) |
|---|---:|---:|---:|---:|---:|---:|
| `[32]` | **0,626** | **0,634** | 0,684 | 518 | 98 | — |
| `[32, 16]` | 0,619 | 0,631 | 0,683 | 950 | 100 | 0,869 |
| `[16]`, `logistic` | 0,608 | 0,621 | **0,688** | 262 | 0 | 0,724 |
| `[64]` | 0,608 | 0,617 | 0,685 | 1030 | 100 | 0,711 |
| `[16]`, `tanh` | 0,606 | 0,620 | 0,672 | 262 | 4 | 0,689 |
| `[16]`, `relu`, `adam` (baseline = vencedor do R1) | 0,597 | 0,614 | 0,674 | 262 | 20 | 0,564 |
| `[16]`, `lbfgs` | 0,571 | 0,584 | 0,651 | 262 | 100 | 0,280 |
| `[16, 8]` | 0,569 | 0,585 | 0,661 | 350 | 52 | 0,290 |
| `[8]` | 0,568 | 0,586 | 0,661 | 134 | 0 | 0,366 |
| `[4]` | 0,513 | 0,537 | 0,628 | 70 | 46 | **0,047** |

- **Quase nada é distinguível.** De `[8]` a `[64]`, todas as configurações ficam a menos de 0,06
  de F1-macro da melhor, sem diferença significativa (p ≥ 0,28). Só a `[4]` é significativamente
  pior, e no limite: p = 0,047 no F1-macro, mas p = 0,065 na balanced accuracy.
- **Tamanho:** `[32]`, `[32, 16]` e `[64]` são as três primeiras entre os tamanhos nas três
  métricas, 0,01 a 0,03 acima do baseline `[16]`, sem significância. Duas camadas não ajudaram:
  `[32, 16]` empata com `[32]` com quase o dobro de parâmetros, e `[16, 8]` fica abaixo de `[16]`.
- **Ativação:** `logistic` e `tanh` ficam ≈ 0,01 acima da `relu` no F1-macro, sem significância. A
  `logistic` tem a maior acurácia da rodada, com 0% de convergência.
- **Otimizador:** o `lbfgs` converge em todos os folds (≈ 737 iterações), mas tem F1-macro 0,055
  abaixo do `adam` (p = 0,28). Convergir não garantiu generalizar melhor. Hipótese (não testada):
  o `lbfgs` se ajusta mais ao treino e sobreajusta.
- **Convergência e full-batch:** no `adam` do scikit-learn, `batch_size = min(200, n_amostras)`.
  Com ≈ 136 amostras de treino por fold, cada época é uma única atualização (full-batch), então
  2000 épocas são só 2000 passos. As redes com 32 neurônios ou mais convergem (98–100%, ≈
  1360–1750 iterações); as de 16 ou menos quase sempre chegam às 2000 iterações (0–52%). Parte da
  vantagem das redes maiores pode ser orçamento de treino, e não capacidade: as comparações de
  tamanho e ativação estão condicionadas a `max_iter = 2000` (ver Limitações).
- **Escolha:** o R2 não fixa o vencedor; ele escolhe os níveis do R2b (método: os 3 melhores de
  cada fator pelo F1-macro). Tamanhos: `[32]`, `[32, 16]` e `[64]`; ativação: as três; solver: os
  dois. Pela regra da rodada aplicada à risca (menor rede sem diferença significativa para a
  melhor), a escolha seria `[8]`, que fica 0,058 abaixo da melhor; o teste tem pouco poder aqui,
  como no R1.
- **Decisão registrada ANTES de ver os resultados do R2b (10/11/2026):**
  com o baseline `[16]` na comparação do R2b, a regra à risca tenderia a escolhê-lo de novo (pelo
  baixo poder do teste), o que anularia o grid. Critério adotado para o R2b:
  1. O vencedor sai **só das configurações da grade**. O baseline `[16]` entra na tabela apenas como
     referência.
  2. Dentro da grade vale a regra da rodada: entre as configurações sem diferença significativa para
     a melhor (F1-macro), fica a de **menos parâmetros**.
  3. Desempate entre as de mesmo nº de parâmetros (ativação e solver não mudam a contagem): maior
     F1-macro médio; persistindo o empate, maior balanced accuracy.
  4. A escolha é confirmada com outra semente de CV antes do R3 (ver R2b).

  Justificativa: (a) no R2, as redes com 32 neurônios ou mais ficaram no topo nas três métricas, não
  só numa; (b) elas convergem (98–100% dos folds) e o `[16]` não (20%), então o `[16]`
  provavelmente está limitado pelo orçamento de treino; (c) o R2b existe para escolher entre os
  melhores níveis. A diferença de médias sozinha (0,01–0,03, sem significância) NÃO justificaria a
  exclusão: é o conjunto (a)–(c) que justifica.

## R2b — Interação entre tamanho, ativação e solver

- Por que existe: o R2 varia um fator por vez e não enxerga interações (ex.: tanh em redes
  maiores, lbfgs em redes pequenas). O R2b faz um grid só com os 3 melhores níveis de cada
  fator do R2 (3 × 3 × 2 = 18 configurações). As 3 com `relu` e `adam` já existiam no R2 e foram
  puladas pelo hash (entram na comparação por `--incluir`, com o baseline `[16]` do R1).
- Níveis usados (3 melhores do R2 pelo F1-macro): `hidden_layer_sizes` `[32]`, `[32, 16]`, `[64]`;
  `activation` `relu`, `tanh`, `logistic` (os três); `solver` `adam`, `lbfgs` (os dois).
- Resultado (tabela completa: `python tabela_rodada.py --rodada R2b_`, em `docs/tabelas/R2b.md`;
  p do F1-macro contra a melhor média). As 5 primeiras e as referências:

| Configuração | F1-macro | Bal. acc. | Acurácia | Parâmetros | Convergiu (%) | p (F1) |
|---|---:|---:|---:|---:|---:|---:|
| `[64]`, logistic, adam (melhor média) | **0,651** | **0,664** | **0,699** | 1030 | 0 | — |
| `[32, 16]`, tanh, adam | 0,646 | 0,659 | 0,684 | 950 | 100 | 0,927 |
| `[64]`, tanh, adam | 0,643 | 0,661 | 0,690 | 1030 | 86 | 0,879 |
| **`[32]`, tanh, adam (vencedor)** | 0,641 | 0,656 | 0,688 | 518 | 50 | 0,845 |
| `[32]`, logistic, adam | 0,632 | 0,638 | 0,686 | 518 | 0 | 0,637 |
| `[32]`, relu, adam (melhor do R2) | 0,626 | 0,634 | 0,684 | 518 | 98 | 0,560 |
| `[16]`, relu, adam (baseline; só referência) | 0,597 | 0,614 | 0,674 | 262 | 20 | 0,302 |
| `[64]`, relu, lbfgs (pior) | 0,576 | 0,589 | 0,648 | 1030 | 100 | 0,130 |

- **Nenhuma diferença é significativa:** todas as configurações têm p ≥ 0,13 contra a melhor
  (F1-macro); a maior distância é 0,075.
- **Houve interação? Sim, entre ativação e tamanho:** com `adam` e 32 neurônios ou mais, o `tanh`
  supera o `relu` em todos os tamanhos: +0,015 com `[32]`, +0,027 com `[32, 16]` e +0,035 com
  `[64]`. No R2, com `[16]`, a vantagem era de só +0,009. Nenhuma dessas diferenças é significativa
  isoladamente; o que pesa é a consistência.
- **O `lbfgs` perdeu em todas as 9 comparações** com o `adam` de mesmo tamanho e ativação (F1-macro
  de 0,576 a 0,623). Nenhuma diferença é significativa isoladamente, mas o padrão é 9 de 9. Indício
  do motivo, pela saída do runner: o log loss do `lbfgs` fica entre 2,1 e 5,8, contra 0,9 a 1,4 no
  `adam` com `tanh` ou `logistic` (as `relu` + `adam` vieram do R2 e não aparecem nessa saída).
  Isso indica previsões confiantes demais e erradas, consistente com sobreajuste: o `lbfgs` converge
  rápido (≈ 110–560 iterações) e se ajusta mais ao treino. É um indício, não um teste.
- **A `logistic` é instável:** é a melhor com `[64]` (0,651), mas fraca com `[32, 16]` (0,606), e
  não converge em nenhum caso (0%). Esse vaivém sugere ruído, e não uma vantagem real.
- **Escolha: `[32]`, `tanh`, `adam`**, pelo critério registrado antes dos resultados (ver R2):
  1. nenhuma configuração da grade difere significativamente da melhor, então todas as 18 são
     candidatas;
  2. as de menos parâmetros são as seis variantes `[32]` (518);
  3. desempate pelo F1-macro: `tanh` + `adam` (0,641) > `logistic` + `adam` (0,632) > `relu` +
     `adam` (0,626).
  O vencedor fica 0,010 abaixo da melhor média (p = 0,85) com metade dos parâmetros (518 × 1030).
  É o vencedor final do R2 e o baseline do R3 (scaler `standard`, `alpha = 0,001` ainda como ponto
  de partida).
- **Convergência do vencedor:** converge em só 50% dos folds (≈ 1932 iterações de 2000). Limitação
  a registrar: o resultado está condicionado a `max_iter = 2000` (ver Limitações); o R3 (`alpha`)
  pode mudar a convergência.
- **Escolha final robusta (outra semente de CV, só nas finalistas):** semente usada: 7. Finalistas
  (o vencedor, a melhor média e as próximas na ordem): `[32]` tanh adam; `[64]` logistic adam;
  `[32, 16]` tanh adam; `[64]` tanh adam; `[32]` logistic adam. O `S7_R2b_baseline` entra só como
  referência.
  - **Critério de leitura (registrado antes de rodar a semente 7):** a escolha é confirmada se, com
    a semente 7, `[32]` tanh adam continua sem diferença significativa para a melhor das finalistas
    e continua acima de `[32]` logistic adam. Se a `[32]` logistic passar à frente, vale a maior
    média de F1-macro nas duas sementes.
  - **Resultado: a escolha se manteve (`[32]`, tanh, adam), pela regra de desempate.** Com a
    semente 7, a 1ª condição foi atendida (sem diferença significativa para a melhor, p = 0,976),
    mas a 2ª não: `[32]` logistic adam ficou 0,001 à frente (0,654 × 0,653). Pela regra registrada,
    vale a maior média de F1-macro nas duas sementes, e `[32]` tanh adam tem a maior das cinco
    finalistas. Tabela da semente 7: `docs/tabelas/S7_R2b.md`.

| Finalista | F1 (semente 42) | F1 (semente 7) | Média das duas | Bal. acc. (semente 7) |
|---|---:|---:|---:|---:|
| **`[32]`, tanh, adam** | 0,641 | 0,653 | **0,647** | **0,668** |
| `[64]`, logistic, adam | 0,651 | 0,642 | 0,646 | 0,653 |
| `[32, 16]`, tanh, adam | 0,646 | 0,642 | 0,644 | 0,653 |
| `[32]`, logistic, adam | 0,632 | 0,654 | 0,643 | 0,665 |
| `[64]`, tanh, adam | 0,643 | 0,626 | 0,635 | 0,642 |

  - **Ranking entre as sementes: praticamente invertido.** A melhor da semente 42 (`[64]` logistic)
    caiu para 4º e a 5ª (`[32]` logistic) subiu para 1º (Spearman ≈ −0,7 entre as duas ordens; com
    só 5 itens, número muito instável). Todas as finalistas ficam a menos de 0,03 de F1-macro umas
    das outras: entre elas, a ordem é ruído.
  - **Leitura:** isso reforça a regra da menor rede. Se a ordem entre as finalistas varia com a
    partição, escolher a mais simples (518 parâmetros) é mais robusto do que escolher a maior média
    de uma semente. A `logistic` oscilou de novo (`[32]`: 0,632 → 0,654; `[64]`: 0,651 → 0,642), com
    0% de convergência nas duas sementes. O vencedor ficou estável (0,641 → 0,653; convergência
    50% → 52%).
  - Pasta do vencedor (semente 42, para o `--incluir` do R3):
    `20261110_004521_04de17_R2b_interacao__hidden_layer_sizes=32_activation=tanh_solver=adam`.
  - **Ordem:** fazer isto DEPOIS de escolher o vencedor do R2b e ANTES de rodar o R3, porque o
    vencedor do R2b é o baseline do R3 e de todas as rodadas seguintes: uma escolha feita por
    sorte nos folds se propagaria até o modelo final.
  - **Como:** `cv_seed_check.py` (raiz do projeto) roda o YAML com outra semente de CV sem
    alterar nenhum arquivo: troca `SEED` e `CV` em memória, separa o hash e põe o prefixo
    `S<semente>_` no nome. Ex.: `python cv_seed_check.py 7 experiments/r2b_interacao.yaml
    --only R2b_baseline <finalistas>`; comparar com `compare.py --rodada S7_R2b_` (sem
    `--incluir` de execuções de outra semente: o `compare.py` não confere a semente).
  - **O que prova:** só a sensibilidade à partição dos folds (a inicialização da rede e a
    reamostragem seguem com a semente 42). Não prova generalização para dados novos: isso é o
    teste final (R8). Descrever como "sensibilidade à partição", não como validação independente.
- Comparações múltiplas: 18 configurações da grade (mais 4 referências) comparadas com a melhor, sem
  correção. Como nenhuma diferença foi significativa, a correção não mudaria a conclusão; mas
  qualquer "significativo" isolado em rodadas assim deveria ser lido com cautela.

## R3 — Regularização

Sweep de `alpha` (1e-4 a 10, escala logarítmica) × `early_stopping` (com e sem), na rede vencedora
do R2b (`[32]`, tanh, adam, scaler `standard`). Valores: média entre os folds (5 × 10); p do
F1-macro contra a melhor média. Tabela completa: `python tabela_rodada.py --rodada R3_`
(`docs/tabelas/R3.md`).

| `alpha` | early stopping | F1-macro | Bal. acc. | Acurácia | Convergiu (%) | Iterações | p (F1) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0,0001 | não | **0,641** | 0,656 | 0,688 | 48 | 1936 | — |
| 0,001 (vencedor do R2b) | não | **0,641** | 0,656 | 0,688 | 50 | 1932 | 1,000 |
| 0,01 | não | 0,633 | 0,650 | 0,686 | 62 | 1893 | 0,706 |
| 0,1 | não | 0,639 | **0,657** | **0,688** | 100 | 1526 | 0,956 |
| **1 (escolhido)** | não | 0,611 | 0,628 | 0,673 | 100 | 946 | 0,577 |
| 10 | não | 0,342 | 0,372 | 0,565 | 100 | 556 | **< 0,001** |
| 0,0001 a 10 | sim | 0,285–0,294 | 0,324–0,335 | 0,458–0,466 | 100* | ≈ 18 | **< 0,001** |

\* Com early stopping, "convergiu" engana: a coluna conta `n_iter < max_iter`, e aqui a rede parou
cedo, não convergiu.

- **O early stopping, como configurado, inviabiliza o treino.** Todas as 6 configurações com
  early stopping caem para F1-macro ≈ 0,29 (p < 0,001), com ≈ 18 iterações em média. Ele separa
  10% do treino de cada fold (≈ 14 exemplos) e para quando a acurácia nessa validação interna não
  melhora por 10 épocas seguidas (`n_iter_no_change = 10`). Com o `adam` em full-batch (um passo por
  época) e passos pequenos, essa nota estagna quase de imediato e o treino para com a rede
  praticamente sem treinar. Por isso o `alpha` não faz diferença com early stopping (resultados
  quase idênticos de 1e-4 a 10): a rede para antes de a regularização agir.
- **Early stopping tirou amostras das classes raras do treino?** Sim, por construção: os 10% de
  validação interna saem do treino do fold, que já tem só ≈ 5–6 exemplos da classe 6. Mas o
  mecanismo dominante foi a parada prematura (≈ 18 iterações), e este desenho não separa os dois
  efeitos. A conclusão vale para esta configuração (`validation_fraction = 0,1`,
  `n_iter_no_change = 10`) nesta base, não para o early stopping em geral.
- **`alpha` sem early stopping:** de 1e-4 a 1, nenhuma diferença é significativa (p ≥ 0,58 no
  F1-macro); `alpha = 10` subajusta e é significativamente pior (F1-macro 0,342). O `alpha` maior
  melhora a convergência: 48–62% até 0,01, e 100% a partir de 0,1, com cada vez menos iterações
  (1526 com 0,1; 946 com 1).
- **`alpha` 0,0001 e 0,001 têm notas idênticas** em todos os 50 folds e nas três métricas, mas
  iterações diferentes (1936 × 1932): não é erro, a diferença de regularização é pequena demais
  para mudar alguma previsão nos folds.
- **Escolha: `alpha = 1`, sem early stopping**, pela regra da rodada (o MAIOR `alpha` sem diferença
  significativa para a melhor): `alpha = 10` é significativamente pior, então fica o 1. Perda de
  0,031 no F1-macro, sem significância em nenhuma métrica (p = 0,58 no F1-macro, 0,62 na balanced
  accuracy e 0,67 na acurácia). Ponto de atenção: o `alpha = 1` fica vizinho de um penhasco (de 1
  para 10 o F1-macro cai de 0,611 para 0,342).
- **Checagem com a semente de CV 7 (critério registrado ANTES de rodar, 10/11/2026):** pela regra,
  a semente 42 escolhe `alpha = 1` (sem early stopping), vizinho de um penhasco (`alpha = 10` cai
  para F1-macro 0,342). Roda-se `alpha` 0,1 e 1 com a semente 7, com `alpha = 0,001` (vencedor do
  R2b, já rodado na semente 7) como referência. Leitura: aplica-se a mesma regra dentro da semente
  7. Se `alpha = 1` continuar sem diferença significativa para a melhor, a escolha se confirma; se
  ficar significativamente pior, vale o próximo maior `alpha` sem diferença significativa (0,1).
  - **Resultado: confirmado, `alpha = 1`.** Com a semente 7, ele continua sem diferença
    significativa para a melhor nas três métricas (p = 0,39 no F1-macro, 0,32 na balanced accuracy
    e 0,60 na acurácia). Tabela: `docs/tabelas/S7_R3.md`.

| `alpha` | F1 (semente 42) | F1 (semente 7) | p (F1), semente 7 | Convergiu (42 / 7) | Iterações (42 / 7) |
|---|---:|---:|---:|---:|---:|
| 0,001 (vencedor do R2b) | 0,641 | 0,653 | — | 50% / 52% | 1932 / 1919 |
| 0,1 | 0,639 | 0,649 | 0,911 | 100% / 100% | 1526 / 1538 |
| **1** | 0,611 | 0,616 | 0,395 | 100% / 100% | 946 / 973 |

  - **A ordem se repetiu entre as sementes** (diferente do R2b, em que o ranking se inverteu):
    0,001 ≈ 0,1 > 1 nas duas. O `alpha = 1` ficou abaixo da melhor nas duas, com perdas parecidas:
    −0,031 e −0,036 no F1-macro; −0,029 e −0,039 na balanced accuracy. Uma perda que se repete com
    partições diferentes provavelmente é real, mesmo que o teste não a detecte.
  - **Leitura para o relatório:** pela regra, o `alpha = 1` foi escolhido e confirmado, mas há
    indício consistente de um custo de ≈ 0,03 de F1-macro, pequeno demais para o teste detectar.
    Em troca, o modelo é mais regularizado e converge em 100% dos folds com metade das iterações,
    também de forma estável nas duas sementes.
  - Pasta do vencedor (semente 42, para o `--incluir` do R4):
    `20261110_011819_aec03d_R3_regularizacao__alpha=1_early_stopping=False`.

## R4 — Desbalanceamento

Reamostragem dentro do pipeline do `imblearn` (só o treino de cada fold é reamostrado; a
validação nunca é tocada), na rede vencedora do R3 (`[32]`, tanh, adam, `alpha = 1`, scaler
`standard`). Valores: média entre os folds (5 × 10); p contra a melhor média. Tabela completa:
`python tabela_rodada.py --rodada R4_` (`docs/tabelas/R4.md`).

| Reamostragem | F1-macro | Bal. acc. | Acurácia | Iterações | p (F1) | p (bal. acc.) |
|---|---:|---:|---:|---:|---:|---:|
| SMOTE (k = 3, padrão do projeto) | **0,679** | **0,723** | 0,658 | 883 | — | — |
| SMOTE, k = 4 | 0,666 | 0,705 | 0,653 | 865 | 0,588 | 0,440 |
| **`random_over` (escolhido)** | 0,663 | 0,706 | 0,644 | 923 | 0,538 | 0,474 |
| SMOTE, k = 2 | 0,659 | 0,700 | 0,642 | 882 | 0,445 | 0,420 |
| nenhuma (vencedor do R3) | 0,611 | 0,628 | **0,673** | 946 | 0,199 | 0,060 |

Todas as configurações convergem em 100% dos folds.

- **As quatro variantes de reamostragem superam "nenhuma"** nas duas métricas que olham as
  classes: +0,048 a +0,068 no F1-macro e +0,072 a +0,095 na balanced accuracy. Nenhuma diferença é
  significativa; a da balanced accuracy chega perto (p = 0,060, SMOTE × nenhuma).
- **A acurácia cai um pouco** (−0,015 a −0,031, sem significância): a mesma troca vista no R0 com o
  `class_weight` da regressão logística (perde-se nas classes grandes, ganha-se nas raras).
- **Tamanho do efeito comparável ao do R0**, em que o `class_weight` deu +0,102 no F1-macro
  (p = 0,032) e +0,145 na balanced accuracy (p = 0,004) e foi significativo.
- **Entre as variantes, diferenças pequenas** (até 0,020 no F1-macro, p ≥ 0,44); a ordem do
  `k_neighbors` do SMOTE (3 > 4 > 2) parece ruído.
- **A reamostragem compensou o custo do `alpha = 1`** (R3): o F1-macro foi de 0,611 para 0,66–0,68,
  acima dos 0,641 do vencedor do R2b.
- **Efeito nas classes 3, 5 e 6:** não medido na CV. O runner grava só as métricas agregadas por
  fold, não as previsões por classe; o efeito por classe aparece na matriz de confusão do teste
  final (R8), com a ressalva de o teste ser pequeno (2 a 3 exemplos das classes 3, 5 e 6).
- **Escolha: `random_over` (desvio da regra, decidido DEPOIS de ver os resultados).** Pela regra
  geral do projeto (entre as sem diferença significativa para a melhor, a mais simples), a escolha
  seria "nenhuma". Optou-se pela opção intermediária:
  1. **Desvio só na pergunta "reamostrar ou não".** Justificativa: direção consistente em 4 de 4
     variantes e nas duas métricas que olham as classes; efeito de 0,05–0,10, do tamanho do efeito
     significativo do `class_weight` no R0; p = 0,060 na balanced accuracy; e motivação prévia do
     plano (seção 4.5) e do R0.
  2. **Dentro da reamostragem, vale a regra:** `random_over` não difere do SMOTE (p = 0,54 no
     F1-macro; 0,47 na balanced accuracy) e é o mais simples (sem hiperparâmetro e sem amostras
     sintéticas). Isso também evita o cuidado do plano com o SMOTE na classe 6, que tem só ≈ 5–6
     exemplos no treino de cada fold.
  Como o desvio foi decidido depois dos resultados, ele é checado com a semente de CV 7 antes do R5.
- **Checagem com a semente de CV 7 (critério registrado ANTES de rodar, 10/11/2026):** roda-se
  `random_over` e SMOTE (k = 3) com a semente 7; "nenhuma" é a execução do R3 com `alpha = 1` já
  rodada na semente 7 (mesma config). Leitura:
  - o desvio se sustenta se, com a semente 7, `random_over` continuar acima de "nenhuma" no
    F1-macro **e** na balanced accuracy; se não, volta-se à regra ("nenhuma");
  - dentro da reamostragem, fica `random_over`, a menos que o SMOTE seja significativamente melhor
    com a semente 7 (aí fica o SMOTE).
  - Resultado: [ ].
- Cuidado registrado: SMOTE com a classe 6 (≈ 5–6 amostras por fold de treino); `k_neighbors` até
  4 funciona, o padrão do projeto é 3.

## R5 — Agrupamento (exploratória)

- janela × outros: [ ]; float × não float: [ ].
- Notas não comparáveis entre alvos. Usar só para discutir quanto do erro vem do número de classes.

## R6 — Combinação e ablação

- Combinação final: [ ] vs. ablações [ ].

## R7 — Config final

[Pipeline completo e a justificativa por rodada.]

## R8 — Teste final

- Métricas no teste: [ ]; baseline majoritário no mesmo teste: [ ].
- Matriz de confusão: classes mais confundidas [ ].

## R9 — Rede × referências

- Diferenças fold a fold e no teste: [ ].

## R10 — Robustez e estabilidade

- f1_macro por semente (média, desvio, mín., máx.): [ ].
- Concordância do ranking de importâncias (Spearman): [ ].

## Limitações a citar

- Base pequena (213 amostras) e classes raras; intervalos largos.
- Teste final pequeno e usado uma única vez.
- Várias comparações por rodada sem correção para comparações múltiplas.
- As interações entre tamanho, ativação e solver foram exploradas só entre os 3 melhores níveis
  de cada fator (R2b), e não em um grid completo; interações com os níveis descartados no R2
  não foram avaliadas. O `alpha` foi variado depois, só na rede escolhida (R3).
- Várias configurações não convergiram em `max_iter = 2000` (com o `adam` em full-batch, uma
  atualização por época); as comparações de tamanho e ativação valem para esse orçamento de treino.
  O modelo escolhido no R3 (`alpha = 1`) converge em 100% dos folds, nas duas sementes de CV.
- O early stopping foi avaliado só com os parâmetros padrão (`validation_fraction = 0,1`,
  `n_iter_no_change = 10`), que nesta base param o treino em ≈ 18 iterações; outras configurações
  dele não foram testadas.
- A regra "maior `alpha` sem diferença significativa" aceitou um custo consistente de ≈ 0,03 de
  F1-macro (repetido nas duas sementes), que o teste não tem poder para detectar.
- Dois desvios da regra "a mais simples entre as sem diferença significativa", ambos decididos depois
  de ver os resultados: R1 (`standard` em vez de `none`) e R4 (reamostrar com `random_over` em vez
  de não reamostrar). Os dois têm efeito de 0,07–0,11 e direção consistente, mas não significância;
  o do R4 foi checado com outra semente de CV.
- O efeito da reamostragem por classe (3, 5 e 6) não foi medido na CV, só no teste final, pequeno.
- Importância por permutação com variáveis correlacionadas pode se diluir.
