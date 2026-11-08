# Fundamentação: a rede MLP e as escolhas do projeto

## O modelo

Um perceptron multicamadas (MLP) é uma sequência de camadas densas. Cada neurônio calcula
`a = f(w·x + b)`. Na saída multiclasse, a função *softmax* converte os valores em
probabilidades, e o treino minimiza a **entropia cruzada** (a `log_loss`) por retropropagação do
erro (Rumelhart et al., 1986). Com `MLPClassifier` do scikit-learn (Pedregosa et al., 2011).

## Escolhas que o projeto testa

| Tema | Por que importa nesta base |
|---|---|
| **Padronização** | RI ≈ 1,5 e Si ≈ 72: sem padronizar, as variáveis de maior magnitude dominam o gradiente e a otimização fica lenta ou instável. O scaler aprende no treino de cada fold. |
| **Ativação** (`relu`, `tanh`, `logistic`) | Define a forma da função aprendida e a facilidade de otimização. |
| **Otimizador** (`adam`, `lbfgs`) | `adam` (Kingma e Ba, 2015) é estocástico; `lbfgs` é de segunda ordem e costuma convergir bem em bases pequenas. O `lbfgs` não guarda a curva de perda. |
| **Tamanho** | Com ~170 amostras de treino, redes grandes memorizam; redes pequenas tendem a empatar. |
| **Regularização L2** (`alpha`) | Penaliza pesos grandes e reduz o sobreajuste. |
| **Early stopping** | Separa 10% do treino para validar e parar. Em bases pequenas tira amostras de classes raras do treino. |
| **Reamostragem** | `random_over` duplica amostras; o SMOTE (Chawla et al., 2002) interpola vizinhos (precisa de `k_neighbors + 1` amostras por classe). Aplicadas só ao treino de cada fold, com o pipeline do imbalanced-learn (Lemaître et al., 2017). |

## Treino com poucos dados

Como o `batch_size` padrão é `min(200, n)` e o treino de cada fold tem menos de 200 amostras, o
treino é *full-batch*: cada época é uma atualização dos pesos. Daí o `max_iter` alto (2000) e o
acompanhamento da convergência (`n_iter`, `convergiu`). Se uma rede não converge, sua nota
está limitada pelo `max_iter`, não pela arquitetura.

## Avaliação

- Validação cruzada estratificada repetida (5×10) e métrica principal `f1_macro`.
- Comparação fold a fold com o teste t corrigido de Nadeau e Bengio (2003), que leva em conta a
  dependência entre folds.
- Importância por permutação em dados de validação (variáveis originais).
- Teste isolado, usado uma única vez.

## Referências

- Chawla, N. V. et al. *SMOTE*. JAIR, 16, 321–357, 2002.
- Kingma, D. P.; Ba, J. *Adam*. ICLR, 2015.
- Lemaître, G.; Nogueira, F.; Aridas, C. K. *Imbalanced-learn*. JMLR, 18(17), 2017.
- Nadeau, C.; Bengio, Y. *Inference for the Generalization Error*. Machine Learning, 52, 2003.
- Pedregosa, F. et al. *Scikit-learn*. JMLR, 12, 2011.
- Rumelhart, D. E.; Hinton, G. E.; Williams, R. J. *Learning representations by back-propagating errors*. Nature, 323, 1986.
