# Anotações para o relatório

Registro de decisões, achados e limitações, para alimentar o relatório. **Preencha cada
seção depois de fechar a rodada correspondente.** Nada aqui é resultado: os campos entre
`[colchetes]` são para preencher com o que os experimentos mostrarem.

## 0. Preparação dos dados

- Base: UCI Glass Identification (id 42), 214 linhas, 9 variáveis + `Id_number` + `Type`.
- `Id_number` descartado (o arquivo é ordenado por classe: o Id vaza o rótulo).
- Duplicatas removidas antes da divisão (a conhecida tem Id diferente, por isso o Id sai antes).
  Amostras após a remoção: [conferir a saída do `make_split`; esperado 213].
- Classe 4 ausente; rótulos mantidos 1, 2, 3, 5, 6, 7.
- Divisão estratificada: teste [20%?], semente 42. Amostras por classe no treino/teste: [colar a tabela do `make_split`].
- Validação cruzada: [5 folds × 10 repetições?]. Valores a confirmar: `TEST_SIZE`, CV, rede inicial.
- Limitação: teste muito pequeno (classes 5 e 6 com poucas amostras); notas por classe são indicativas.

## R0 — Referências

- Dummy `most_frequent`: acc [ ], balanced_acc [ ], f1_macro [ ].
- Dummy `stratified`: [ ].
- Regressão logística (padronizada / balanced): [ ].
- Leitura: [o piso e o modelo simples que a rede precisa superar].

## R1 — Padronização

- Pergunta: quanto a escala das variáveis atrapalha?
- Resultado (f1_macro, iterações, % convergiu): none [ ] · standard [ ] · robust [ ] · minmax [ ].
- Teste t corrigido (p): [ ].
- Escolha: [ ] porque [ ]. A hipótese (sem padronização, pior) se confirmou? [ ].

## R2 — Tamanho da rede, ativação e otimizador

- Resultado: [tabela de neurônios, 1 vs 2 camadas, relu/tanh/logistic, adam/lbfgs].
- Escolha (menor rede sem diferença significativa): [ ].
- Observação sobre convergência e full-batch: [ ].

## R3 — Regularização

- alpha × early_stopping: [ ].
- Early stopping tirou amostras das classes raras do treino? [ ].
- Escolha (maior alpha sem diferença significativa): [ ].

## R4 — Desbalanceamento

- none / random_over / smote (k 2, 3, 4): f1_macro [ ], balanced_accuracy [ ].
- Efeito nas classes 3, 5 e 6: [ ].
- Escolha: [ ].
- Cuidado: SMOTE com a classe 6 (~5 amostras por fold de treino).

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
- Importância por permutação com variáveis correlacionadas pode se diluir.
