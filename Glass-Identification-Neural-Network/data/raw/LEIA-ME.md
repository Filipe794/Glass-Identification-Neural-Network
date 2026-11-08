# data/raw

Coloque aqui o arquivo **`glass.data`**, exatamente como vem da UCI (sem cabeçalho, sem alterações).
Esta pasta nunca é modificada pelo projeto: a preparação (descartar o `Id_number`, remover
duplicatas e dividir em treino e teste) é feita por `python -m src.make_split`, que grava em
`data/split/`.

## Como obter

1. Abra <https://archive.ics.uci.edu/dataset/42/glass+identification> e clique em **Download**.
2. Extraia o zip e copie o `glass.data` para esta pasta.

O arquivo tem 214 linhas e 11 colunas separadas por vírgula: `Id_number, RI, Na, Mg, Al, Si, K,
Ca, Ba, Fe, Type`.

## Licença e citação

Base distribuída sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).

German, B. (1987). *Glass Identification* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5WW2P
