## Rodada R2b

| Configuração | F1-macro | Bal. acc. | Acurácia | Parâmetros | Convergiu (%) | p (F1) |
|---|---:|---:|---:|---:|---:|---:|
| [64], logistic, adam | **0,651** | **0,664** | **0,699** | 1030 | 0 | — |
| [32, 16], tanh, adam | 0,646 | 0,659 | 0,684 | 950 | 100 | 0,927 |
| [64], tanh, adam | 0,643 | 0,661 | 0,690 | 1030 | 86 | 0,879 |
| [32], tanh, adam | 0,641 | 0,656 | 0,688 | 518 | 50 | 0,845 |
| [32], logistic, adam | 0,632 | 0,638 | 0,686 | 518 | 0 | 0,637 |
| [32] | 0,626 | 0,634 | 0,684 | 518 | 98 | 0,560 |
| [32, 16], tanh, lbfgs | 0,623 | 0,631 | 0,672 | 950 | 100 | 0,576 |
| [32, 16] | 0,619 | 0,631 | 0,683 | 950 | 100 | 0,489 |
| [64], tanh, lbfgs | 0,615 | 0,633 | 0,665 | 1030 | 100 | 0,438 |
| [32], logistic, lbfgs | 0,614 | 0,625 | 0,679 | 518 | 100 | 0,510 |
| [64], logistic, lbfgs | 0,614 | 0,624 | 0,670 | 1030 | 100 | 0,478 |
| [64] | 0,608 | 0,617 | 0,685 | 1030 | 100 | 0,426 |
| [32, 16], logistic, adam | 0,606 | 0,616 | 0,681 | 950 | 0 | 0,350 |
| [32], tanh, lbfgs | 0,605 | 0,611 | 0,675 | 518 | 100 | 0,410 |
| R1_baseline | 0,597 | 0,614 | 0,674 | 262 | 20 | 0,302 |
| [32], relu, lbfgs | 0,594 | 0,603 | 0,667 | 518 | 98 | 0,322 |
| [32, 16], logistic, lbfgs | 0,588 | 0,604 | 0,657 | 950 | 100 | 0,207 |
| [32, 16], relu, lbfgs | 0,584 | 0,599 | 0,661 | 950 | 100 | 0,225 |
| [64], relu, lbfgs | 0,576 | 0,589 | 0,648 | 1030 | 100 | 0,130 |

Média entre os folds. Ordenado pelo F1-macro. p: teste t corrigido (Nadeau e Bengio), bilateral, no F1-macro, contra a melhor média ([64], logistic, adam); em negrito, p < α. Negrito nas métricas: melhor média da coluna.

Gerado por `tabela_rodada.py --rodada R2b_` a partir de `results/comparacoes/R2b_<métrica>.csv`.
