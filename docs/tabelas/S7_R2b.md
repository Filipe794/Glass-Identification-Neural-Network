## Rodada S7_R2b

| Configuração | F1-macro | Bal. acc. | Acurácia | Parâmetros | Convergiu (%) | p (F1) |
|---|---:|---:|---:|---:|---:|---:|
| [32], logistic, adam | **0,654** | 0,665 | 0,690 | 518 | 0 | — |
| [32], tanh, adam | 0,653 | **0,668** | 0,681 | 518 | 52 | 0,976 |
| [32, 16], tanh, adam | 0,642 | 0,653 | 0,675 | 950 | 100 | 0,782 |
| [64], logistic, adam | 0,642 | 0,653 | **0,694** | 1030 | 0 | 0,758 |
| [64], tanh, adam | 0,626 | 0,642 | 0,675 | 1030 | 82 | 0,495 |
| S7_R2b_baseline | 0,616 | 0,631 | 0,671 | 262 | 20 | 0,450 |

Média entre os folds. Ordenado pelo F1-macro. p: teste t corrigido (Nadeau e Bengio), bilateral, no F1-macro, contra a melhor média ([32], logistic, adam); em negrito, p < α. Negrito nas métricas: melhor média da coluna.

Gerado por `tabela_rodada.py --rodada S7_R2b_` a partir de `results/comparacoes/S7_R2b_<métrica>.csv`.
