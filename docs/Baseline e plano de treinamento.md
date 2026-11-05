# Baselines da UCI e plano de treinamento

## 1. Por que os baselines de redes neurais da UCI são tão baixos

A página da base na UCI mostra, na seção *Baseline Model Performance*, uma acurácia média de cerca de 22% para a rede neural e de cerca de 40% para o SVC, contra 83–85% para Random Forest e XGBoost. A acurácia da rede está abaixo do que se obtém respondendo sempre a classe mais frequente (classe 2, com 76 de 214 amostras, ou 35,5%), o que indica falha de treinamento, e não um limite real do modelo. A precisão reforça isso: se ela for macro (média entre as 6 classes), um classificador que prevê apenas uma classe tem precisão em torno de 0,355 / 6 ≈ 5,9%, e é exatamente nessa faixa que ficam a rede (~6%) e o SVC (~7%). Os dois modelos estão colapsando em uma ou poucas classes. A causa mais provável é a ausência de padronização: as variáveis estão em escalas muito diferentes (RI em torno de 1,51–1,53; Si em torno de 70–75; Ba e Fe quase sempre zero). Isso prejudica fortemente modelos que dependem de gradientes ou distâncias, como MLP e SVM com kernel RBF, mas não afeta árvores, que são invariantes à escala. A esse quadro somam-se hiperparâmetros padrão (no scikit-learn, 100 neurônios e `max_iter=200`, que em geral não convergem nessa base), apenas 214 amostras e classes muito pequenas.

As barras de erro largas também são esperadas: em folds de cerca de 40 exemplos, a classe 6 aparece uma ou duas vezes, e cada acerto ou erro muda muito a métrica. A UCI não documenta como esses baselines são gerados (pré-processamento, hiperparâmetros, protocolo de avaliação, tipo de média da precisão), então essa explicação é uma inferência a partir dos números, e não uma descrição do procedimento oficial.

## 2. Referências

**Base e artigo original**

- German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P — página: https://archive.ics.uci.edu/dataset/42/glass+identification
- Evett, I. W.; Spiehler, E. J. *Rule induction in forensic science*. Technical Report, Central Research Establishment, Home Office Forensic Science Service, 1987.

**Referências de magnitude (desempenho esperado)**

| Referência | Protocolo | Resultado |
|---|---|---|
| Adamiak, K.; Duch, P.; Ślot, K. *Object Classification Using Support Vector Machines with Kernel-based Data Preprocessing*. Image Processing & Communications, v. 21, n. 3, p. 45–54, 2016. DOI: [10.1515/ipc-2016-0015](https://doi.org/10.1515/ipc-2016-0015). Texto: https://www.academia.edu/117807902/Object_Classification_Using_Support_Vector_Machines_with_Kernel_based_Data_Preprocessing | SVM com C e parâmetro do kernel ajustados por *grid search*; validação cruzada de 5 folds repetida 20 vezes | Glass foi a base mais difícil das três avaliadas; sem ajuste fino dos parâmetros, o SVM em dados brutos perdeu entre 4% e 8% de desempenho |
| *Glass Classification based on Machine Learning Algorithms*. https://www.academia.edu/98505253/Glass_Classification_based_on_Machine_Learning_Algorithms | Comparação de vários algoritmos | Random Forest foi o melhor, com 79,62% de acurácia |
| McLeod, A. I. Notas de aula *Forensic Glass* (Western University, 2019). https://fisher.stats.uwo.ca/faculty/aim/2019/9850/RNotebooks/20_01_ForensicGlass_Apr5.html | Validação cruzada de 10 folds repetida 7 vezes; inclui MLPs | Acurácias em torno de 65–71%; o autor observa que a validação honesta reduz a acurácia aparente |
| Projeto *Glass Project* (GitHub). https://github.com/Quentin-dev937/Glass-Project | RobustScaler + reamostragem; validação em um único split de 43 amostras | 91% — **otimista**, pela validação em um único conjunto pequeno; não usar como alvo |

Faixa realista para um MLP bem configurado: **cerca de 65–75% de acurácia**, comparável à regressão logística e um pouco abaixo dos ensembles de árvores.

## 3. Baseline para o treinamento

O baseline de referência é um classificador de **classe majoritária** (`DummyClassifier(strategy="most_frequent")`), avaliado com o mesmo protocolo de validação cruzada dos demais modelos. Na base completa (214 amostras), seus valores esperados são:

| Métrica | Classe majoritária (sempre classe 2) | Aleatório estratificado (`strategy="stratified"`) |
|---|---|---|
| Acurácia | 35,5% (76/214) | ≈ 26,3% (Σ pᵢ²) |
| Balanced accuracy | 16,7% (1/6) | ≈ 16,7% |
| Precisão macro | ≈ 5,9% | ≈ 16,7% |
| F1-macro | ≈ 8,7% | ≈ 16,7% |

Regras:

- Todo modelo deve ser comparado com esse baseline nas mesmas métricas e nos mesmos folds. Um modelo que não supera a classe majoritária em **balanced accuracy e F1-macro** não aprendeu nada útil, mesmo com acurácia razoável.
- Os valores devem ser recalculados após a limpeza dos dados (por exemplo, a remoção de duplicatas altera ligeiramente as proporções).
- Como referência adicional de "modelo simples", recomenda-se reportar também uma **regressão logística com padronização**, no mesmo protocolo. Ela indica o que é alcançável sem não linearidade.
- Os baselines da UCI (RF ≈ 83%, XGBoost ≈ 85%, regressão logística ≈ 65%) servem como contexto, mas não como comparação direta, já que o protocolo deles não é documentado.

## 4. Sugestões para o treinamento

### 4.1 Preparação dos dados

- **Remover a coluna `Id_number`.** Na página da UCI ela tem papel de ID, e o `ucimlrepo` já a exclui; mas, ao ler o `glass.data` diretamente, ela vem junto. Como o arquivo está ordenado por classe, o Id vaza o rótulo.
- **Remover duplicatas.** Verificar com `df.duplicated()` e remover antes da divisão em folds, para que a mesma amostra não apareça no treino e na validação.
- **Outliers com cautela.** Nessa base, vários "outliers" são justamente o que caracteriza as classes raras (por exemplo, Ba elevado em faróis). Não remover outliers de forma global. Se algum tratamento for testado, ele deve ser feito apenas no treino de cada fold e comparado com a versão sem tratamento. Uma alternativa mais segura é o `RobustScaler`.

### 4.2 Padronização dentro de um Pipeline

- `StandardScaler` → `MLPClassifier` em um `Pipeline` do scikit-learn (ou `imblearn.pipeline.Pipeline`, quando houver reamostragem).
- O scaler é ajustado **somente no treino de cada fold**, evitando vazamento de informação da validação.

### 4.3 Rede pequena e regularizada

- 1 camada oculta com **8–32 neurônios** (com 214 amostras, redes maiores tendem a sobreajustar).
- Ajustar `alpha` (regularização L2), por exemplo em escala logarítmica de 1e-4 a 1e1.
- `max_iter` alto (por exemplo, 2000) e `early_stopping=True`. Observar que o `early_stopping` separa uma fração do treino para validação, o que reduz ainda mais as classes raras; comparar com e sem.
- Fixar `random_state` e reportar a variação entre sementes.

### 4.4 Avaliação

- `RepeatedStratifiedKFold` (por exemplo, 5 folds × 10 repetições).
- Reportar **F1-macro** e **balanced accuracy**, além da acurácia, com média e desvio padrão.
- Mostrar a **matriz de confusão** agregada.
- Comparar sempre com o baseline de classe majoritária (35,5% de acurácia; seção 3).
- Ajuste de hiperparâmetros com validação cruzada aninhada (ou um conjunto de teste estratificado separado no início e usado só no fim), para não superestimar o desempenho.

### 4.5 Desbalanceamento

- O `MLPClassifier` não tem `class_weight`, então usar **oversampling dentro do fold** (`imblearn`: `RandomOverSampler` ou `SMOTE`), sempre dentro do pipeline.
- Cuidado com SMOTE na **classe 6, que tem só 9 exemplos**: em um fold de treino ela tem cerca de 7, então ajustar `k_neighbors` (por exemplo, 3) para não gerar erro nem amostras sintéticas pouco representativas.
- Comparar: sem tratamento, oversampling aleatório e SMOTE.

### 4.6 Experimento extra (opcional): agrupamento de classes

- Agrupar classes em tarefas binárias, como **janela vs. não janela** (classes 1, 2, 3 vs. 5, 6, 7) ou **float vs. não float** (classes 1, 3 vs. 2), esta última sendo a tarefa do estudo original citado na página da UCI.
- Comparar o desempenho da rede no problema agrupado com o problema de 6 classes, para mostrar como o desbalanceamento e o tamanho das classes afetam a rede.