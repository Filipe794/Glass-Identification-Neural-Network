# Afirmações-chave para o relatório e a apresentação

Registro das afirmações que o grupo considera importantes. Cada entrada traz o texto original,
uma formulação mais precisa para usar no relatório, a evidência do próprio trabalho, a
referência (a conferir antes de citar) e onde a afirmação será usada.

Status: `a conferir` = ainda não verificado na fonte; `ok` = verificado.

---

## 1. Escala das variáveis e treinamento do MLP

**Texto original:**
> O MLP aprende por gradiente. Variáveis com valores grandes dominam o cálculo, enquanto as de
> escala pequena quase não influenciam os pesos.

**Formulação sugerida para o relatório:**
O MLP é treinado por descida de gradiente. Na primeira camada, o gradiente em relação a um peso é
proporcional ao valor da variável de entrada correspondente; assim, variáveis de escala grande
tendem a dominar a ativação dos neurônios e a atualização dos pesos, enquanto as de escala
pequena contribuem pouco. Variáveis em escalas muito diferentes também pioram o
condicionamento do problema de otimização, o que torna o treinamento mais lento e instável.
A padronização das entradas atenua esse efeito.

**Por que vale a pena precisar a frase:** "quase não influenciam os pesos" é uma simplificação.
O mais correto é dizer que a escala distorce a contribuição relativa das variáveis e a
otimização, não que as variáveis pequenas sejam ignoradas.

**Evidência no trabalho (R1):**
- Sem padronização (`none`) o F1-macro foi 0,484, contra 0,597 do `standard` (−0,113), e a
  balanced accuracy foi 0,499 contra 0,614 (−0,115). A queda aparece nas três métricas.
- A diferença **não foi estatisticamente significativa** (p = 0,143 no F1-macro; p = 0,114 na
  balanced accuracy). Não houve colapso: o `none` ficou bem acima dos dummies do R0.
- `standard`, `robust` e `minmax` ficaram equivalentes (diferenças < 0,02 de F1-macro).
- Cuidado ao redigir: dizer que a evidência é **consistente com** a afirmação, e não que a
  confirma de forma conclusiva.

**Referência (a conferir):** LeCun, Y.; Bottou, L.; Orr, G. B.; Müller, K.-R. *Efficient
BackProp*. In: Neural Networks: Tricks of the Trade, 1998 (recomenda normalizar as entradas).
Status: `a conferir`.

**Onde usar:** relatório (metodologia e resultados do R1) e slide de padronização.

---

## 2. Validação da escolha com outra semente de CV: só no R2b

**Texto original:**
> O R2b é a rodada com mais configurações (até 18 comparações com a melhor), então é onde o
> risco de escolher por acaso é maior. As outras rodadas têm poucas comparações: o R1 tem 4, o
> R4 tem 5. Repetir a validação em todas daria muito trabalho para pouco ganho.

**Formulação sugerida para o relatório:**
Quanto mais configurações são comparadas com a melhor, maior a chance de a escolhida ser
resultado do acaso nos folds (comparações múltiplas e "maldição do vencedor"). O R2b é a rodada
com mais configurações (até 18) e o seu vencedor é o ponto de partida de todas as rodadas
seguintes; por isso, a escolha foi confirmada com outra semente de CV, aplicada apenas às
configurações finalistas.

**Correção em relação ao texto original:** a frase "as outras rodadas têm poucas comparações" é
imprecisa. O R2 e o R3 também têm cerca de 12 configurações cada (o R2 tem 7 + 3 + 2 com o
baseline, o R3 tem 6 × 2). O que distingue o R2b é ser o **maior** grid e o que **propaga** a
escolha para as demais; R1 (4) e R4 (5) são, de fato, pequenas. Se a justificativa for usada no
relatório, escrever "o maior número de configurações", e não "as outras têm poucas".

**Evidência no trabalho (R2b, semente de CV 7 nas 5 finalistas):**
- A escolha (`[32]`, tanh, adam) se manteve, mas pela regra de desempate: com a semente 7 ela
  continuou sem diferença significativa para a melhor (p = 0,976), porém ficou 0,001 atrás de
  `[32]` logistic adam. Pela maior média de F1-macro nas duas sementes (0,647), segue vencedora.
- O ranking das finalistas praticamente se inverteu entre as sementes (Spearman ≈ −0,7, com só 5
  itens): a melhor da semente 42 caiu para 4º. Isso confirma, no próprio trabalho, o risco que a
  afirmação descreve: entre configurações a menos de 0,03 de F1-macro, escolher pela maior média de
  uma única partição seria escolher por acaso.
- A checagem mede só a sensibilidade à partição dos folds, não a generalização (ver R8).

**Limitação associada:** as demais rodadas (R2, R3, R4) não foram repetidas com outra semente de
CV e têm comparações múltiplas sem correção (ver "Limitações a citar" em
`docs/anotacoes_artigo.md`). Uma checagem final, mais barata, pode ser feita no R7.

**Referência:** a conferir (comparações múltiplas e seleção de modelos por validação cruzada,
por exemplo Cawley e Talbot, 2010, *On Over-fitting in Model Selection and Subsequent Selection
Bias in Performance Evaluation*, JMLR 11). Status: `a conferir`.

**Onde usar:** relatório (metodologia, seleção de modelo) e slide de rigor experimental.

---

## 3. Origem do ruído na comparação de configurações

**Texto original:**
> O ruído decorre de diferenças entre configurações (≈ 0,01–0,03 de F1-macro) muito menores que a
> variação entre folds (≈ 0,10). Essa variação é causada sobretudo pelo tamanho da base e pelas
> classes raras, que têm 1 ou 2 exemplos por fold de validação.

**Formulação sugerida para o relatório:**
As configurações finalistas diferem entre si em cerca de 0,01 a 0,03 de F1-macro, enquanto o
desvio padrão entre folds é de cerca de 0,10. Essa variação vem principalmente do tamanho da base:
com 170 amostras de treino, cada fold de validação tem cerca de 34 exemplos, e as classes raras
aparecem com 1 ou 2 exemplos por fold (classe 6: 7 no treino; classe 5: 10). Como o F1-macro dá
peso 1/6 a cada classe, acertar ou errar um único exemplo de uma classe rara pode mover o F1-macro
do fold em cerca de 0,1 a 0,17. Nessas condições, a ordem entre configurações próximas é dominada
pelo acaso da partição.

**Ressalva importante (não atribuir tudo ao tamanho da base):** há três fatores além dele:
1. **Configurações de fato equivalentes:** se duas configurações são realmente equivalentes, a ordem
   entre elas é sorteio com qualquer tamanho de base; mais dados só revelariam uma diferença real,
   se ela existir.
2. **Efeito de seleção:** escolher a melhor entre muitas (18 no R2b) tende a pegar a que teve mais
   sorte nos folds (ver entrada 2).
3. **Treino não convergido:** parte das redes para em `max_iter = 2000`, o que adiciona variação que
   vem do treino, e não dos dados.

**Evidência no trabalho:**
- Desvio do F1-macro entre folds de ≈ 0,08 a 0,11 em todas as rodadas (R0 a R2b).
- R2b: as 5 finalistas a menos de 0,03 de F1-macro umas das outras; com a semente de CV 7, o
  ranking praticamente se inverteu (Spearman ≈ −0,7).
- Split: treino com 55, 61, 14, 10, 7 e 23 amostras nas classes 1, 2, 3, 5, 6 e 7; em 5 folds
  estratificados, isso dá ≈ 34 exemplos por fold de validação, ≈ 1,4 da classe 6 e ≈ 2 da classe 5
  (cálculo a partir da tabela do split; contagem exata por fold não conferida).
- O impacto de 0,1 a 0,17 por exemplo é ilustrativo: depende também dos falsos positivos da classe.

**Referência:** não é necessária para o argumento (é uma consequência direta do cálculo do
F1-macro e do tamanho dos folds); pode-se citar a seção 1 de `docs/baseline-e-plano-de-treinamento.md`,
que já observa a mesma instabilidade nos baselines da UCI.

**Onde usar:** relatório (discussão dos resultados e limitações) e slide sobre por que quase nada é
significativo.

---

## 4. Diferença mínima detectável pelo teste (poder do teste)

**Texto original:**
> Nesta base, só diferenças da ordem de 0,08–0,10 de F1-macro ou balanced accuracy podem ser
> detectadas como significativas. Efeitos menores, mesmo consistentes entre sementes (como os
> ≈ 0,03 do `alpha = 1` e os ≈ 0,05–0,08 da reamostragem), ficam abaixo do poder do teste.

**Formulação sugerida para o relatório:**
As comparações usam o teste t corrigido de Nadeau e Bengio sobre as 50 notas da validação
cruzada (5 folds × 10 repetições): t = média(d) / √((1/50 + 1/4) · var(d)), em que d é a diferença
entre os dois modelos em cada fold e 1/4 = n_validação / n_treino (34/136). Com 49 graus de
liberdade e α = 0,05, a diferença média só é significativa quando supera ≈ 1,04 vezes o desvio
padrão das diferenças por fold (2,01 × 0,52). Nesta base, esse desvio fica em ≈ 0,08–0,10, o que
coloca a diferença mínima detectável em ≈ 0,08–0,10 de F1-macro ou de balanced accuracy. Efeitos
menores, ainda que consistentes entre partições, não são detectados; por isso a consistência entre
sementes de CV foi usada como evidência complementar.

**Por que a correção torna o teste exigente:** sem o termo 1/4, o limite seria ≈ 0,02–0,03, quase
quatro vezes menor. A correção existe porque as 50 notas não são independentes: os treinos de
folds diferentes compartilham ≈ 75% das amostras, e as 10 repetições reusam os mesmos dados. O
teste t comum trataria as notas como independentes e daria falsos positivos demais; o corrigido é
mais honesto, mas tem menos poder.

**Evidência no trabalho (R4; limites calculados a partir do t e da diferença de cada CSV):**

| Comparação | Diferença observada | Desvio das diferenças | Limite para p < 0,05 | p |
|---|---:|---:|---:|---:|
| F1, `random_over` × nenhuma (semente 7) | 0,048 | 0,093 | 0,097 | 0,324 |
| Bal. acc., SMOTE × nenhuma (semente 7) | 0,082 | 0,079 | 0,082 | 0,052 |
| F1, SMOTE × nenhuma (semente 42) | 0,068 | 0,100 | 0,105 | 0,199 |
| Bal. acc., SMOTE × nenhuma (semente 42) | 0,095 | 0,095 | 0,099 | 0,060 |

- A balanced accuracy na semente 7 ficou praticamente no limite (0,0818 × 0,0824), daí o p = 0,052.
- Efeitos consistentes entre sementes, mas abaixo do limite: custo do `alpha = 1` (≈ 0,03 nas
  duas sementes, R3) e ganho da reamostragem (≈ 0,05–0,08, R4).
- O t crítico 2,01 (49 graus de liberdade) é o da tabela t; a fórmula foi conferida em
  `compare.py` (`corrected_ttest`, com `np.var(d, ddof=1)`).

**Ligação com as outras entradas:** é a razão quantitativa da entrada 3 (o ruído entre folds
domina diferenças pequenas) e o motivo de as escolhas terem sido confirmadas pela consistência
entre sementes (entrada 2), e não só pelo p-valor.

**Referência (a conferir):** Nadeau, C.; Bengio, Y. *Inference for the Generalization Error*.
Machine Learning, v. 52, p. 239–281, 2003. Status: `a conferir`.

**Onde usar:** relatório (metodologia: teste estatístico; discussão e limitações) e slide sobre
por que quase nada é significativo.

---