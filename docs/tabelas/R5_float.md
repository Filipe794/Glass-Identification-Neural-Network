## Rodada R5_float

| Configuração | F1-macro | Bal. acc. | Acurácia | Parâmetros | Convergiu (%) | p (F1) |
|---|---:|---:|---:|---:|---:|---:|
| R5_float_vs_nao_float | **0,748** | **0,750** | **0,754** | 353 | 100 | — |
| R5_float_sem_reamostragem | 0,743 | 0,746 | 0,750 | 353 | 100 | 0,853 |
| R5_float_logreg | 0,689 | 0,692 | 0,694 | — | — | 0,162 |
| R5_float_dummy | 0,347 | 0,500 | 0,531 | — | — | **< 0,001** |

Média entre os folds. Ordenado pelo F1-macro. p: teste t corrigido (Nadeau e Bengio), bilateral, no F1-macro, contra a melhor média (R5_float_vs_nao_float); em negrito, p < α. Negrito nas métricas: melhor média da coluna.

Gerado por `tabela_rodada.py --rodada R5_float` a partir de `results/comparacoes/R5_float_<métrica>.csv`.
