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
  - **Resultado: confirmado, `random_over`.** As duas partes do critério foram atendidas: (1)
    `random_over` continua acima de "nenhuma" no F1-macro (+0,048) e na balanced accuracy (+0,080);
    (2) o SMOTE não é significativamente melhor (−0,002 no F1-macro, +0,002 na balanced accuracy,
    p ≈ 0,93). Tabela: `docs/tabelas/S7_R4.md`.

| Reamostragem | F1 (semente 42) | F1 (semente 7) | Bal. acc. (42) | Bal. acc. (7) | Acurácia (42) | Acurácia (7) |
|---|---:|---:|---:|---:|---:|---:|
| **`random_over`** | 0,663 | **0,664** | 0,706 | 0,708 | 0,644 | 0,648 |
| SMOTE (k = 3) | **0,679** | 0,663 | **0,723** | **0,710** | 0,658 | 0,641 |
| nenhuma | 0,611 | 0,616 | 0,628 | 0,628 | **0,673** | **0,673** |

  - **O ganho da reamostragem se repetiu** entre as sementes, com tamanho parecido; a balanced
    accuracy de "nenhuma" contra a reamostragem ficou de novo perto da significância (p = 0,060 na
    semente 42; p = 0,052 na semente 7).
  - **A ordem entre `random_over` e SMOTE se inverteu** (SMOTE à frente na semente 42, atrás na 7):
    a diferença entre eles é ruído, o que apoia escolher o mais simples.
  - **A troca com a acurácia também se repetiu:** "nenhuma" tem a maior acurácia nas duas sementes;
    a reamostragem perde ≈ 0,025 nas classes grandes para ganhar nas raras.
  - Pasta do vencedor (semente 42, para o `--incluir` das próximas rodadas):
    `20261110_013202_485353_R4_reamostragem__balancing=random_over`.
- Cuidado registrado: SMOTE com a classe 6 (≈ 5–6 amostras por fold de treino); `k_neighbors` até
  4 funciona, o padrão do projeto é 3.

## R5 — Agrupamento (exploratória)

Rede: a vencedora do R4 (`standard`, `random_over`, `[32]`, tanh, adam, `alpha = 1`). Alvos:
`window_vs_other` (janela 1, 2, 3 × outros 5, 6, 7) e `float_vs_nonfloat` (só janelas: float 1 e
3 × não float 2; descarta as classes 5, 6 e 7; é a tarefa do estudo original da UCI).

- **Experimentos de apoio por alvo (acrescentados antes de rodar, 10/11/2026):** sem eles, os
  números binários não têm referência.
  - `*_dummy` (classe majoritária, `balancing: none`): o piso de cada problema. No treino, janela ×
    outros = 130 × 40 (piso de acurácia ≈ 76%); float × não float = 69 × 61 (piso ≈ 53%). O
    `balancing: none` é obrigatório: com `random_over` as classes empatam e o `most_frequent`
    deixaria de ser o piso.
  - `*_logreg` (regressão logística balanceada): a rede acrescenta algo no problema binário, ou um
    modelo linear já basta?
  - `*_sem_reamostragem` (a mesma rede sem `random_over`): o ganho da reamostragem depende do grau
    de desbalanceamento (76/24 no janela × outros; 53/47 no float × não float)?

Valores: média entre os folds (5 × 10); p do F1-macro contra a melhor média do próprio alvo.
Tabelas: `docs/tabelas/R5_janela.md` e `docs/tabelas/R5_float.md`. A rede binária tem 353
parâmetros (uma saída) e converge em 100% dos folds nos dois alvos.

**Janela × outros** (treino 130 × 40; piso de acurácia 76,5%)

| Modelo | F1-macro | Bal. acc. | Acurácia | p (F1) |
|---|---:|---:|---:|---:|
| Regressão logística balanceada | **0,902** | **0,905** | 0,930 | — |
| Rede sem reamostragem | 0,902 | 0,890 | **0,933** | 0,991 |
| Rede (com `random_over`) | 0,895 | 0,895 | 0,925 | 0,700 |
| Dummy (classe majoritária) | 0,433 | 0,500 | 0,765 | < 0,001 |

**Float × não float** (só janelas; treino 69 × 61; piso de acurácia 53,1%)

| Modelo | F1-macro | Bal. acc. | Acurácia | p (F1) |
|---|---:|---:|---:|---:|
| Rede (com `random_over`) | **0,748** | **0,750** | **0,754** | — |
| Rede sem reamostragem | 0,743 | 0,746 | 0,750 | 0,853 |
| Regressão logística balanceada | 0,689 | 0,692 | 0,694 | 0,162 |
| Dummy (classe majoritária) | 0,347 | 0,500 | 0,531 | < 0,001 |

- **Os dummies confirmam o protocolo:** acurácia igual às proporções do treino (0,765 e 0,531) e
  balanced accuracy 0,500.
- **Janela × outros é fácil:** ≈ 93% de acurácia contra piso de 76,5%. Rede e regressão logística
  empatam (p ≥ 0,47 entre os três modelos): um modelo linear basta; a não linearidade não acrescenta.
- **Float × não float é difícil:** ≈ 75% contra piso de 53%. É a tarefa do estudo original da UCI
  (Evett e Spiehler, 1987). A rede fica ≈ 0,06 acima da regressão logística (p = 0,16; abaixo do
  limite detectável de ≈ 0,10, ver `afirmacoes-chave.md`, entrada 4).
- **A rede supera a regressão logística em algum dos problemas binários?** Só no float × não float,
  e sem significância (+0,059 no F1-macro). No janela × outros, não (empate).
- **O ganho da reamostragem muda com o desbalanceamento?** Nos dois problemas binários a
  reamostragem não faz diferença (−0,007 a +0,005 no F1-macro). No float × não float, isso era
  esperado (53/47). No janela × outros, o desbalanceamento é forte (76/24) e mesmo assim não houve
  ganho: a minoria tem 40 exemplos e é bem separável. Isso sugere que o ganho de ≈ 0,05 da
  reamostragem no problema de 6 classes (R4) vem das classes MUITO pequenas (7 a 14 exemplos no
  treino), e não do desbalanceamento em si.
- **⚠ HIPÓTESE CENTRAL (H1) — leitura combinada (inferência, não medida direta):** se janela × não janela é quase resolvido e
  float × não float é o gargalo, boa parte do erro do problema de 6 classes deve vir da confusão
  entre as classes 1 e 2 (janelas de edificação float e não float), mais do que das classes raras
  isoladamente. A conferir na matriz de confusão do teste final (R8): ver "Teste da H1" na seção R8.
- Notas não comparáveis entre alvos (problemas e amostras diferentes). Ler cada uma contra o seu
  próprio piso e usar só para discutir quanto do erro vem do número de classes e do tamanho delas.
- Comparar por alvo: `compare.py --rodada R5_janela` e `--rodada R5_float` (o `compare.py` recusa
  misturar alvos); tabelas com `tabela_rodada.py`.

## R6 — Combinação e ablação

Combinação completa (baseline do R6 = vencedor do R4): `standard` → `random_over` → MLP `[32]`,
tanh, adam, `alpha = 1`, sem early stopping. Ablações, cada uma tirando um elemento (voltando ao
valor de antes da decisão, não ao padrão do scikit-learn): sem padronização (R1), sem
reamostragem (R4), `alpha` padrão 0,0001 (R3), 16 neurônios (R2b, tamanho), relu (R2b, ativação)
e rede inicial `[16]`, relu (R2b inteiro). Sem `hidden_layer_sizes`, o scikit-learn usaria 100
neurônios, configuração nunca testada; por isso a ablação do tamanho volta a `[16]`, o ponto de
partida do R2.

- **Critério de leitura (decidido ANTES de rodar, 10/11/2026): o R6 confirma, não redecide.**
  - Um elemento só sai se a ablação **não for pior em média nas duas métricas** (F1-macro e
    balanced accuracy) que a combinação completa (`--ref`). Se tirar o elemento piora a média em
    pelo menos uma delas, ele fica.
  - Motivo: o critério original ("se tirar não piora significativamente, sai") não funciona com o
    poder do teste desta base (diferença mínima detectável ≈ 0,08–0,10; ver
    `afirmacoes-chave.md`, entrada 4). Aplicado à risca, ele tiraria quase tudo, inclusive a
    padronização e a reamostragem, mantidas por desvios justificados no R1 e no R4: o R6 desfaria
    essas decisões pela mesma falta de poder que as motivou. O critério adotado usa o mesmo tipo de
    evidência dos desvios (direção consistente em duas métricas).
  - Os dois elementos do R2b (tamanho e ativação) são lidos um por vez; se os dois saírem, vale a
    ablação conjunta (`R6_rede_inicial`) como conferência.
  - Se mais de um elemento sair, a combinação reduzida é a configuração final do R7.
- **Resultado (semente 42):** combinação completa (`--ref`, vencedor do R4): F1-macro 0,663,
  balanced accuracy 0,706, acurácia 0,644. Tabela: `docs/tabelas/R6.md`.

| Ablação | Δ F1-macro | Δ Bal. acc. | Δ Acurácia | p (F1) | Decisão pelo critério |
|---|---:|---:|---:|---:|---|
| `alpha` padrão (0,0001) | **+0,015** | **+0,0001** | +0,045 | 0,727 | **sai o `alpha = 1`** |
| 16 neurônios | −0,007 | −0,001 | +0,001 | 0,829 | fica o `[32]` |
| relu | −0,011 | −0,030 | +0,014 | 0,796 | fica o tanh |
| rede inicial (`[16]`, relu) | −0,015 | −0,024 | +0,019 | 0,688 | confirma as duas acima |
| sem reamostragem | −0,052 | −0,077 | +0,029 | 0,295 | fica o `random_over` |
| sem padronização | −0,087 | −0,071 | −0,079 | 0,077 | fica o `standard` |

  Nenhuma ablação é significativamente diferente da combinação completa (p ≥ 0,077 no F1-macro).

  - **Sai um elemento, o `alpha = 1`.** Configuração resultante: `standard` → `random_over` → MLP
    `[32]`, tanh, adam, `alpha` padrão (0,0001). Ela já foi rodada: é a execução `R6_alpha_padrao`
    (F1-macro 0,678, balanced accuracy 0,706, acurácia 0,689, convergência 96%, ≈ 1672 iterações).
  - **A saída passou por um fio na balanced accuracy** (+0,00008, praticamente empate; pela regra,
    empate conta como "não pior"). O F1-macro (+0,015) e a acurácia (+0,045) apontam na mesma direção.
  - **Coerente com o R3:** lá a regra aceitou um custo de ≈ 0,03 do `alpha = 1`, consistente nas duas
    sementes; o R6 desfaz justamente essa escolha. Com a reamostragem, o `alpha` alto deixou de ser
    necessário, e a convergência continua alta (96%).
  - **`[32]` fica por margem mínima** (−0,007 e −0,001 ao voltar para `[16]`): pela regra é claro,
    mas o tamanho pouco importa nessa faixa, e o relatório deve dizer isso.
  - **Padronização e reamostragem se confirmaram**, com as maiores perdas ao serem tiradas
    (−0,05 a −0,09). Sem padronização, ficou perto da significância (p = 0,077 no F1-macro;
    p = 0,065 na acurácia), mais forte que no R1.
  - **Limitação:** as ablações de tamanho e ativação foram feitas com `alpha = 1`; não foram refeitas
    com o `alpha` padrão.
  - Pasta da configuração resultante (semente 42): `20261110_122254_0747b5_R6_alpha_padrao`.
- **Checagem com a semente de CV 7 (critério registrado ANTES de rodar, 10/11/2026):** a decisão
  muda a configuração por margem mínima numa das métricas, então roda-se `R6_alpha_padrao` com a
  semente 7 e compara-se com a combinação completa na semente 7 (`S7_R4_..._random_over`, `--ref`).
  - O `alpha = 1` sai de vez se, com a semente 7, o `R6_alpha_padrao` também não for pior em média
    no F1-macro e na balanced accuracy. Se for pior em alguma das duas, o `alpha = 1` volta e a
    configuração final é a combinação completa.
  - **Resultado: confirmado, o `alpha = 1` sai de vez.** Com a semente 7, o `R6_alpha_padrao` não
    foi pior em nenhuma das duas métricas: +0,019 no F1-macro (p = 0,61), +0,002 na balanced
    accuracy e +0,043 na acurácia (0,683 / 0,710 / 0,691 × 0,664 / 0,708 / 0,648). Tabela:
    `docs/tabelas/S7_R6.md`.

| `alpha` padrão × combinação completa | Δ F1-macro | Δ Bal. acc. | Δ Acurácia |
|---|---:|---:|---:|
| semente 42 | +0,015 | +0,0001 | +0,045 |
| semente 7 | +0,019 | +0,002 | +0,043 |

  - A direção se repetiu nas três métricas e nas duas sementes, com tamanhos parecidos. A balanced
    accuracy segue praticamente empatada; o ganho está no F1-macro e na acurácia.

## R7 — Config final

**Pipeline final:** `StandardScaler` → `RandomOverSampler` (só no treino de cada fold, dentro do
pipeline do `imblearn`) → `MLPClassifier(hidden_layer_sizes=(32,), activation="tanh",
solver="adam", alpha=0.0001, max_iter=2000, random_state=42)`, sem early stopping. 518 parâmetros.

**Desempenho na validação cruzada (treino, 5 × 10):** F1-macro 0,678 (semente 42) e 0,683 (semente
7); balanced accuracy 0,706 e 0,710; acurácia 0,689 e 0,691; convergência 96% nas duas.
Execução de referência (semente 42): `20261110_122254_0747b5_R6_alpha_padrao`.

**Justificativa por rodada:**

| Rodada | Decisão | Como foi decidida |
|---|---|---|
| R0 | pisos: dummy (F1-macro 0,088) e regressão logística (0,513; balanceada 0,616) | referência, sem escolha |
| R1 | scaler `standard` | **desvio da regra** (a regra escolheria `none`): efeito de 0,11 consistente nas três métricas e fundamento teórico; confirmado no R6 (sem padronização é a pior ablação) |
| R2 | níveis do R2b: `[32]`, `[32, 16]`, `[64]`; as três ativações; os dois solvers | os 3 melhores tamanhos pelo F1-macro (um fator por vez) |
| R2b | `[32]`, tanh, adam | regra registrada antes dos resultados (vencedor só da grade; menos parâmetros; desempate pelo F1-macro); confirmado com a semente 7 pela regra de desempate (maior média nas duas sementes) |
| R3 | `alpha = 1`, sem early stopping | regra da rodada (maior `alpha` sem diferença significativa); confirmado com a semente 7, mas com custo consistente de ≈ 0,03; o early stopping falhou (parada em ≈ 18 iterações) |
| R4 | `random_over` | **desvio da regra** (a regra escolheria "nenhuma"), só na pergunta "reamostrar ou não"; dentro da reamostragem, o mais simples; confirmado com a semente 7 |
| R5 | exploratória (não escolhe) | janela × outros fácil (≈ 93%, a logística basta); float × não float difícil (≈ 75%); hipótese central H1 a testar no R8 |
| R6 | sai o `alpha = 1` (volta ao padrão 0,0001); ficam `standard`, `random_over`, `[32]` e tanh | critério registrado antes de rodar (sai se não for pior em média no F1-macro e na balanced accuracy); confirmado com a semente 7 |

**Resumo para o relatório:** quase nenhuma diferença foi significativa (limite detectável de
≈ 0,08–0,10; `afirmacoes-chave.md`, entrada 4). As decisões se apoiaram em regras registradas antes
dos resultados e, nos dois desvios (R1 e R4), na direção consistente entre métricas e sementes de
CV. As escolhas com mais efeito foram a padronização e a reamostragem; tamanho, ativação e `alpha`
mudam pouco o resultado nesta base.

**Próximos passos:** R8 (teste final, uma única vez, nesta configuração), R9 (rede × referências)
e R10 (robustez à semente da rede, só depois do R8).

## R8 — Teste final

Rodado uma única vez (10/11/2026), na configuração do R7:
`python final_test.py --run 20261110_122254_0747b5_R6_alpha_padrao` (resultados em
`results/final/20261110_122254_0747b5_R6_alpha_padrao/`). Teste: 43 exemplos (14, 15, 3, 3, 2 e 6
nas classes 1, 2, 3, 5, 6 e 7).

| | Teste | CV, semente 42 | CV, semente 7 | Majoritário no teste* |
|---|---:|---:|---:|---:|
| F1-macro | **0,726** | 0,678 | 0,683 | ≈ 0,086 |
| Balanced accuracy | **0,762** | 0,706 | 0,710 | 0,167 |
| Acurácia | **0,698** | 0,689 | 0,691 | 0,349 |
| Precisão macro | 0,737 | | | |
| ROC AUC (um-contra-todos, macro) | 0,942 | | | |
| Log loss | 1,020 | | | |

\* Calculado (sempre a classe 2: 15 de 43); conferir com `baseline_majoritario.json`.

- **Teste ≈ 0,05 acima da CV no F1-macro:** dentro do ruído esperado (desvio entre folds ≈ 0,09; o
  teste tem o tamanho de pouco mais de um fold, e classes com 2–3 exemplos movem muito o
  F1-macro). Não ler como "generaliza melhor que o esperado"; ler como consistente com a CV.

**Por classe (teste):**

| Classe | Precisão | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| 1 construção float | 0,800 | 0,571 | 0,667 | 14 |
| 2 construção não float | 0,625 | 0,667 | 0,645 | 15 |
| 3 veículo float | 0,333 | 0,667 | 0,444 | 3 |
| 5 recipientes | 1,000 | 0,667 | 0,800 | 3 |
| 6 louça de mesa | 0,667 | 1,000 | 0,800 | 2 |
| 7 faróis | 1,000 | 1,000 | 1,000 | 6 |

**Matriz de confusão (linhas = real; colunas = previsto):**

| real \ previsto | 1 | 2 | 3 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 5 | 1 | 0 | 0 | 0 |
| 2 | 2 | 10 | 2 | 0 | 1 | 0 |
| 3 | 0 | 1 | 2 | 0 | 0 | 0 |
| 5 | 0 | 0 | 1 | 2 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 2 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 6 |

- **Classes mais confundidas: 1 e 2** (5 de real 1 → previsto 2; 2 de real 2 → previsto 1). 30 acertos
  e 13 erros no total.
- **11 dos 13 erros (85%) são entre janelas** (classes 1, 2 e 3); só 2 cruzam janela × não janela
  (real 2 → previsto 6; real 5 → previsto 3). Janela × não janela no teste: 41 de 43 (95%),
  coerente com o R5 (≈ 93% na CV).
- **A classe 3 atrai erros:** recebeu 6 previsões, só 2 certas (precisão 0,33). Hipótese (não
  testada): efeito do `random_over`, que reforça as classes pequenas e empurra previsões para elas.
- **Classes raras:** faróis 6/6, louça de mesa 2/2, recipientes 2/3. Com 2–3 exemplos, indicativos.
- **Teste da H1 (hipótese central do R5; critério registrado ANTES do teste final, 10/11/2026):**
  H1 = boa parte do erro do problema de 6 classes vem da confusão entre as classes 1 e 2 (janelas
  de edificação float × não float), mais do que das classes raras.
  - **Confirmada** se os erros 1↔2 (real 1 previsto 2, mais real 2 previsto 1) forem a maior parte
    dos erros da matriz de confusão do teste (≥ 50% dos erros).
  - **Refutada** se a maior parte dos erros envolver as classes 3, 5, 6 ou 7.
  - **Ressalva:** o teste tem 43 exemplos (14 da classe 1, 15 da classe 2), então poucos erros mudam
    a proporção; ler como indício. Evidência mais forte exigiria as previsões da validação cruzada
    por classe (`cross_val_predict` no treino), que o runner não grava.
  - **Resultado: erros 1↔2 = 7 de 13 erros (54%); H1 confirmada pelo critério**, mas por margem
    mínima: um único erro 1↔2 a menos (6 de 12 = 50%) ainda confirmaria, dois a menos não. Pela
    ressalva registrada, ler como indício. Reforço: 11 dos 13 erros (85%) estão entre janelas
    (1, 2 e 3), e só 2 cruzam janela × não janela.

## R9 — Rede × referências

Comando: `python mlp_vs_baselines.py --mlp 20261110_122254_0747b5_R6_alpha_padrao --baselines
20261109_231353_e6648f_R0_dummy_majoritaria 20261109_231410_91dd22_R0_logreg_padronizada
20261109_231420_cf92ae_R0_logreg_balanceada` (resultados em `results/mlp_vs_baselines/`).

**Validação cruzada (fold a fold, teste t corrigido; * = p < 0,05):**

| Rede × | Δ F1-macro | Δ Bal. acc. | Δ Acurácia | Δ ROC AUC | Δ Log loss |
|---|---:|---:|---:|---:|---:|
| Dummy majoritário | +0,590* | +0,539* | +0,330* | +0,412* | −21,96* |
| Logística padronizada | **+0,165*** (p = 0,006) | **+0,185*** (p = 0,002) | +0,075 (p = 0,07) | **+0,073*** (p = 0,001) | +0,095 (pior; p = 0,47) |
| Logística balanceada | +0,063 (p = 0,22) | +0,040 (p = 0,47) | **+0,121*** (p = 0,013) | **+0,068*** (p = 0,004) | −0,023 (p = 0,87) |

- **Primeiras diferenças significativas fora do dummy:** contra a logística padronizada, a rede
  ganha onde a diferença passa de ≈ 0,1 (coerente com o limite detectável, `afirmacoes-chave.md`,
  entrada 4).
- **Contra a logística balanceada, a vantagem no F1-macro e na balanced accuracy some:** grande
  parte do ganho da rede sobre o modelo linear vem do tratamento do desbalanceamento, não da não
  linearidade.
- **O que a não linearidade parece acrescentar:** a rede equilibra as classes sem pagar em
  acurácia (a logística balanceada perde 0,12) e tem ROC AUC significativamente maior que as duas
  logísticas. Como a AUC mede a ordenação das probabilidades, independentemente de limiar e de peso
  de classe, é o sinal mais limpo de que a rede discrimina melhor.
- **Cuidado: a comparação na CV favorece a rede.** Ela foi escolhida comparando dezenas de
  configurações nos mesmos folds (efeito de seleção), e as logísticas ficaram com os parâmetros
  padrão (`C = 1`, sem ajuste). Parte da vantagem na CV pode ser otimista; o teste, usado uma
  única vez, não tem esse viés.

**Teste (43 exemplos):**

| | F1-macro | Bal. acc. | Acurácia | ROC AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| **Rede** | **0,726** | **0,762** | **0,698** | **0,942** | 1,020 |
| Logística padronizada | 0,672 | 0,718 | **0,698** | 0,905 | **0,822** |
| Logística balanceada | 0,607 | 0,739 | 0,558 | 0,898 | 1,063 |
| Dummy majoritário | 0,086 | 0,167 | 0,349 | 0,500 | 23,47 |

**F1 por classe no teste** (`f1_por_classe.csv` / `.png`):

| Classe (suporte) | Rede | Logística padronizada | Logística balanceada | Dummy |
|---|---:|---:|---:|---:|
| 1 construção float (14) | 0,667 | 0,667 | 0,583 | 0 |
| 2 construção não float (15) | 0,645 | 0,645 | 0,364 | 0,517 |
| 3 veículo float (3) | **0,444** | 0 | 0,308 | 0 |
| 5 recipientes (3) | 0,800 | **1,000** | 0,667 | 0 |
| 6 louça de mesa (2) | 0,800 | 0,800 | 0,800 | 0 |
| 7 faróis (6) | **1,000** | 0,923 | 0,923 | 0 |

- **No teste, a rede lidera no F1-macro, na balanced accuracy e na AUC** e empata em acurácia com a
  logística padronizada: a mesma ordem da CV.
- **A vantagem da rede sobre a logística padronizada no teste vem quase toda da classe 3** (2 de 3
  acertos × nenhum). Nas demais classes elas empatam ou trocam por pouco (a logística vai melhor
  em recipientes, a rede em faróis). Com 3 exemplos na classe 3, frágil, mas coerente com o papel
  da reamostragem.
- **Calibração:** a logística padronizada tem log loss melhor (0,82 × 1,02 no teste; 1,05 × 1,15 na
  CV, sem significância): as probabilidades da rede são menos calibradas. Hipótese (não testada): o
  `random_over` muda as proporções das classes no treino e desloca as probabilidades. Importa se as
  probabilidades forem usadas, não só a classe prevista.
- Para o relatório: a legenda do `f1_por_classe.png` usa os nomes das pastas; trocar por rótulos
  legíveis antes de usar a figura.

## R10 — Robustez e estabilidade

Configuração final (R7) com 5 sementes de inicialização da rede (`random_state` 1, 7, 123, 2024 e
31415), além da 42 já rodada (`R6_alpha_padrao`). Folds da CV e `random_over` seguem com a semente
42: só a inicialização muda (o `adam` é full-batch nesta base, então a ordem dos mini-lotes não
varia). Não muda a escolha do modelo; rodado depois do R8.

Comandos:
`python run_experiments.py experiments/r10_robustez.yaml` e
`python stability.py --run 20261110_122254_0747b5_R6_alpha_padrao --seeds R10_`.
Opcional: `compare.py --rodada R10_ --ref 20261110_122254_0747b5_R6_alpha_padrao` (diferenças
pareadas contra a semente 42) e `tabela_rodada.py --rodada R10_`.

- **Como ler (definido antes de rodar, 10/11/2026):** comparar a variação do F1-macro entre
  sementes com (a) a diferença entre as finalistas das rodadas (≈ 0,01–0,03 de F1-macro no R2b,
  R3 e R6) e (b) o desvio entre folds (≈ 0,09). Se a variação entre sementes for da ordem de (a),
  parte das diferenças entre configurações é ruído de inicialização, e não só de partição dos
  folds (complementa a checagem com a semente de CV 7, que mediu a partição).
- f1_macro por semente (média, desvio, mín., máx.): [ ].
- Balanced accuracy e acurácia por semente: [ ].
- Concordância do ranking de importâncias (Spearman): [ ]; variável mais importante em cada semente [ ].
- A variação entre sementes é da ordem das diferenças entre finalistas? [ ].

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
- O critério do R6 foi trocado antes de rodar ("sai se não piorar significativamente" → "sai se
  não piorar em média no F1-macro e na balanced accuracy"), pelo baixo poder do teste.
- As regressões logísticas de referência não tiveram hiperparâmetros ajustados (`C = 1`), enquanto a
  rede passou por várias rodadas de seleção nos mesmos folds: a comparação do R9 na CV favorece a
  rede; o teste não tem esse viés, mas é pequeno.
- Importância por permutação com variáveis correlacionadas pode se diluir.
