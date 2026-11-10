## Rodada R5_janela

| Configuração | F1-macro | Bal. acc. | Acurácia | Parâmetros | Convergiu (%) | p (F1) |
|---|---:|---:|---:|---:|---:|---:|
| R5_janela_logreg | **0,902** | **0,905** | 0,930 | — | — | — |
| R5_janela_sem_reamostragem | 0,902 | 0,890 | **0,933** | 353 | 100 | 0,991 |
| R5_janela_vs_outros | 0,895 | 0,895 | 0,925 | 353 | 100 | 0,700 |
| R5_janela_dummy | 0,433 | 0,500 | 0,765 | — | — | **< 0,001** |

Média entre os folds. Ordenado pelo F1-macro. p: teste t corrigido (Nadeau e Bengio), bilateral, no F1-macro, contra a melhor média (R5_janela_logreg); em negrito, p < α. Negrito nas métricas: melhor média da coluna.

Gerado por `tabela_rodada.py --rodada R5_janela` a partir de `results/comparacoes/R5_janela_<métrica>.csv`.
