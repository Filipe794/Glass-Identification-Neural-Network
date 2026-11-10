"""Junta as comparações salvas de uma rodada (F1-macro, balanced accuracy e acurácia) numa
tabela Markdown, pronta para o relatório, e grava em docs/tabelas/<rodada>.md.

Lê os CSVs que o compare.py grava com --salvar, então rode antes, para as três métricas:
  python compare.py --rodada R2_ --incluir <...> --salvar
  python compare.py --rodada R2_ --metric balanced_accuracy --incluir <...> --salvar
  python compare.py --rodada R2_ --metric accuracy --incluir <...> --salvar

Uso:
  python tabela_rodada.py --rodada R2_
  python tabela_rodada.py --rodada R2b_ --desvio        (mostra média ± desvio)
  python tabela_rodada.py --rodada S7_R2b_ --saida docs/tabelas

A tabela é ordenada pelo F1-macro (métrica principal). O p-valor é o do F1-macro, contra a
execução de maior média nessa métrica (a referência do compare.py); a melhor média de cada
métrica fica em negrito. Só usa a biblioteca padrão.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
COMPARACOES_DIR = BASE / "results/comparacoes"
METRICAS = [("f1_macro", "F1-macro"), ("balanced_accuracy", "Bal. acc."), ("accuracy", "Acurácia")]
PRINCIPAL = "f1_macro"

# Parâmetros cujo valor sozinho já diz o que é (scaler=robust -> robust)
SO_VALOR = {"activation", "solver", "scaler", "balancing", "target", "model"}


def ler_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {linha["execução"]: linha for linha in csv.DictReader(f)}


def num(texto):
    try:
        v = float(texto)
    except (TypeError, ValueError):
        return math.nan
    return v


def fmt(v, casas=3):
    """0.6257 -> 0,626 (vírgula decimal); NaN -> —."""
    return "—" if math.isnan(v) else f"{v:.{casas}f}".replace(".", ",")


def fmt_p(v):
    if math.isnan(v):
        return "—"
    return "< 0,001" if v < 0.001 else fmt(v)


def rotulo(execucao):
    """20261110_002629_78b8fb_R2_neuronios__hidden_layer_sizes=32x16 -> [32, 16].
    Sem sweep (ex.: R1_baseline), devolve o nome do experimento."""
    nome = execucao.split("_", 3)[3] if execucao.count("_") >= 3 else execucao
    if "__" not in nome:
        return nome
    sufixo = nome.split("__", 1)[1]
    # pares chave=valor separados por "_"; a chave pode ter "_" (hidden_layer_sizes)
    pares = re.findall(r"(?:^|_)([a-z][a-z_]*?)=(.+?)(?=_[a-z][a-z_]*=|$)", sufixo)
    if not pares:
        return sufixo
    partes = []
    for chave, valor in pares:
        if chave == "hidden_layer_sizes":
            partes.append("[" + ", ".join(valor.split("x")) + "]")
        elif chave in SO_VALOR:
            partes.append(valor)
        else:
            partes.append(f"{chave}={valor}")
    return ", ".join(partes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rodada", required=True, metavar="PREFIXO", help="o mesmo prefixo do compare.py (ex.: R2_)")
    ap.add_argument("--saida", default="docs/tabelas", help="pasta de saída (padrão: docs/tabelas)")
    ap.add_argument("--desvio", action="store_true", help="mostra média ± desvio")
    args = ap.parse_args()

    prefixo = args.rodada.rstrip("_")
    dados = {}
    for metrica, _ in METRICAS:
        path = COMPARACOES_DIR / f"{prefixo}_{metrica}.csv"
        if not path.exists():
            sys.exit(f"Não encontrei {path.relative_to(BASE)}. Rode antes: "
                     f"python compare.py --rodada {args.rodada} --metric {metrica} [--incluir ...] --salvar")
        dados[metrica] = ler_csv(path)

    principal = dados[PRINCIPAL]
    execucoes = sorted(principal, key=lambda e: num(principal[e]["média"]), reverse=True)
    faltando = {m: [e for e in execucoes if e not in dados[m]] for m, _ in METRICAS}
    for m, lista in faltando.items():
        if lista:
            print(f"Aviso: {len(lista)} execução(ões) sem {m} (aparece —): {', '.join(lista)}")

    # melhor média de cada métrica (negrito)
    melhor = {m: max((num(dados[m][e]["média"]) for e in execucoes if e in dados[m]), default=math.nan)
              for m, _ in METRICAS}
    ref = next((e for e in execucoes if principal[e]["significativo (p < α)"] == "referência"), execucoes[0])

    cab = ["Configuração"] + [n for _, n in METRICAS] + ["Parâmetros", "Convergiu (%)", "p (F1)"]
    linhas = ["| " + " | ".join(cab) + " |",
              "|---|" + "---:|" * (len(cab) - 1)]
    for e in execucoes:
        celulas = [rotulo(e)]
        for m, _ in METRICAS:
            linha = dados[m].get(e)
            if linha is None:
                celulas.append("—")
                continue
            media, desvio = num(linha["média"]), num(linha["desvio"])
            texto = fmt(media) + (f" ± {fmt(desvio)}" if args.desvio else "")
            celulas.append(f"**{texto}**" if media == melhor[m] else texto)
        p = principal[e]
        params = num(p["parâmetros"])
        celulas.append("—" if math.isnan(params) else str(int(params)))
        conv = num(p["convergiu (%)"])
        celulas.append("—" if math.isnan(conv) else fmt(conv, 0))
        pv = num(p["p"])
        sig = p["significativo (p < α)"] == "sim"
        celulas.append("—" if e == ref else (f"**{fmt_p(pv)}**" if sig else fmt_p(pv)))
        linhas.append("| " + " | ".join(celulas) + " |")

    nota = (f"Média{' ± desvio' if args.desvio else ''} entre os folds. Ordenado pelo F1-macro. "
            f"p: teste t corrigido (Nadeau e Bengio), bilateral, no F1-macro, contra a melhor média "
            f"({rotulo(ref)}); em negrito, p < α. Negrito nas métricas: melhor média da coluna.")
    texto = (f"## Rodada {prefixo}\n\n" + "\n".join(linhas) + f"\n\n{nota}\n\n"
             f"Gerado por `tabela_rodada.py --rodada {args.rodada}` a partir de "
             f"`results/comparacoes/{prefixo}_<métrica>.csv`.\n")

    saida = BASE / args.saida
    saida.mkdir(parents=True, exist_ok=True)
    arquivo = saida / f"{prefixo}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Gravado em {arquivo.relative_to(BASE)}")


if __name__ == "__main__":
    main()
