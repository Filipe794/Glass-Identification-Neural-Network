# Redes neurais para identificação de vidros (Glass Identification)

Trabalho da disciplina Aprendizado de Máquina, do PPGCC - UFPI, 2026.2.

**Autores:** Filipe Mendes, Bétanny Alexandra, Juciane Maciel · **Orientação/professor:** Prof. Dr. Vinicius Ponte Machado.

**Relatório do trabalho:** *Redes Neurais para Identificação de Tipos de Vidro utilizando a Base Glass Identification* (PDF — a ser incluído ao final do trabalho)

Descreverá a metodologia, os experimentos por rodada, o modelo escolhido, o resultado no conjunto de teste e a análise de robustez.

## Objetivo

Construir e avaliar uma rede neural do tipo perceptron multicamadas (MLP) que identifique o tipo de vidro a partir da sua composição química e do índice de refração, em um cenário de poucos dados e classes muito desbalanceadas. O trabalho compara, de forma controlada, o efeito da padronização das variáveis, do tamanho e da regularização da rede, do tratamento do desbalanceamento (com reamostragem feita apenas dentro de cada fold) e do agrupamento de classes, sempre com validação cruzada estratificada e comparação explícita com um baseline de classe majoritária.

A análise dos baselines publicados na página da UCI e o plano de treinamento estão em [`docs/baseline-e-plano-de-treinamento.md`](docs/baseline-e-plano-de-treinamento.md).

## Base de dados

**Glass Identification** (UCI Machine Learning Repository, id 42): 214 amostras de vidro descritas por 9 variáveis contínuas — índice de refração (RI) e percentual em peso dos óxidos de sódio (Na), magnésio (Mg), alumínio (Al), silício (Si), potássio (K), cálcio (Ca), bário (Ba) e ferro (Fe) —, além de uma coluna identificadora (`Id_number`) que não deve ser usada como variável. O alvo (`Type`) tem 7 categorias previstas, mas apenas 6 presentes:

| Classe | Descrição | Amostras |
|---|---|---|
| 1 | Janela de edificação, processo float | 70 |
| 2 | Janela de edificação, não float | 76 |
| 3 | Janela de veículo, processo float | 17 |
| 4 | Janela de veículo, não float | 0 (ausente) |
| 5 | Recipientes | 13 |
| 6 | Utensílios de mesa | 9 |
| 7 | Faróis | 29 |

A motivação original é forense: o vidro deixado na cena de um crime pode servir de evidência, se identificado corretamente.

**Artigo original:** Evett, I. W.; Spiehler, E. J. *Rule induction in forensic science*. Technical Report, Central Research Establishment, Home Office Forensic Science Service, Aldermaston, Reading, Berkshire, 1987.

**Citação da base:** German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WW2P

**Disponível em:** https://archive.ics.uci.edu/dataset/42/glass+identification

O arquivo usado está em `data/raw/`, sem alterações. A remoção da coluna de identificação e de duplicatas é feita na etapa de preparação dos dados.

## Licença

MIT. A base de dados segue os termos da fonte original (CC BY 4.0).