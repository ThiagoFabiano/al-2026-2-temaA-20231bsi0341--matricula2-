# [IA] Claude Code - 2026-11-19 - download da lista de municípios do ES (IBGE) com registro no manifesto
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
MANIFESTO = RAIZ / "data" / "MANIFEST.txt"

URL_MUNICIPIOS = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/32/municipios"
TOTAL_MUNICIPIOS_ES = 78
GEOCODE_SERRA = 3205002


def baixar(url, timeout=60):
    try:
        resposta = requests.get(url, timeout=timeout)
        resposta.raise_for_status()
    except requests.exceptions.RequestException as erro:
        raise RuntimeError(f"falha ao acessar {url}: {erro}") from erro
    return resposta.content


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def registrar_manifesto(manifesto, data_hora, url, registros, arquivo):
    novo = not manifesto.exists()
    manifesto.parent.mkdir(parents=True, exist_ok=True)
    with open(manifesto, "a", encoding="utf-8") as f:
        if novo:
            f.write("data_hora\turl\tregistros\tsha256\tarquivo\n")
        relativo = os.path.relpath(arquivo, manifesto.parent.parent)
        f.write(f"{data_hora}\t{url}\t{registros}\t{sha256(arquivo)}\t{relativo}\n")


def extrair_municipios(dados):
    return [(str(m["id"]), m["nome"]) for m in dados]


def validar_municipios(municipios):
    if len(municipios) != TOTAL_MUNICIPIOS_ES:
        raise ValueError(f"esperados {TOTAL_MUNICIPIOS_ES} municípios, vieram {len(municipios)}")
    codigos = [codigo for codigo, _ in municipios]
    invalidos = [c for c in codigos if len(c) != 7 or not c.isdigit()]
    if invalidos:
        raise ValueError(f"códigos IBGE que não têm 7 dígitos: {invalidos}")
    if str(GEOCODE_SERRA) not in codigos:
        raise ValueError(f"código de Serra ({GEOCODE_SERRA}) não está na lista")


def baixar_municipios(dir_raw=DIR_RAW, manifesto=MANIFESTO):
    dir_raw = Path(dir_raw)
    manifesto = Path(manifesto)
    data_hora = datetime.now().astimezone().isoformat(timespec="seconds")
    conteudo = baixar(URL_MUNICIPIOS)

    dir_raw.mkdir(parents=True, exist_ok=True)
    arquivo = dir_raw / "municipios_es.json"
    arquivo.write_bytes(conteudo)

    try:
        dados = json.loads(conteudo)
    except json.JSONDecodeError as erro:
        raise ValueError(f"resposta do IBGE não é um JSON válido: {erro}") from erro

    registrar_manifesto(manifesto, data_hora, URL_MUNICIPIOS, len(dados), arquivo)

    municipios = extrair_municipios(dados)
    validar_municipios(municipios)
    return municipios


def main():
    try:
        municipios = baixar_municipios()
    except (RuntimeError, ValueError) as erro:
        sys.exit(f"ERRO: {erro}")
    print(f"{len(municipios)} municípios do ES obtidos (Serra = {GEOCODE_SERRA} presente)")


if __name__ == "__main__":
    main()
# [/IA]
