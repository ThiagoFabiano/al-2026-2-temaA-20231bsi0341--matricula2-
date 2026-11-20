# Álgebra Linear 2026/2 — Trabalho Prático, Tema A

**Vigilância em estados: cadeias de Markov e a dinâmica dos alertas de dengue no Espírito Santo**

IFES Campus Serra — Bacharelado em Sistemas de Informação — Prof. Fidelis Zanetti de Castro

## Integrantes

| Nome | Matrícula |
|---|---|
| Thiago Fabiano | 20231BSI0341 |
| Gabriel Oliveira | _(a preencher)_ |

## Objetivo

Modelar a sequência semanal dos níveis de alerta do InfoDengue (1 verde, 2 amarelo, 3 laranja, 4 vermelho) como cadeia de Markov e usar a matriz de transição para responder perguntas sobre a dinâmica dos alertas em Serra, em um segundo município e no ES agregado.

**Município escolhido:** _(a definir na Etapa 1, com justificativa)_

## Ambiente

- Python 3.12.7 (mínimo exigido: 3.10)
- Dependências com versões fixadas em `requirements.txt`

## Como instalar, executar e testar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py   # regenera results/ a partir de data/raw/, sem internet
pytest
```

Tempo aproximado de execução de `run_all.py`: _(a medir)_.

`src/download.py` foi executado uma única vez para gerar `data/raw/` e `data/MANIFEST.txt` e **não** faz parte de `run_all.py`.

## Organização

```
data/raw/            CSV brutos congelados do InfoDengue
data/MANIFEST.txt    data/hora, URL, linhas e SHA-256 de cada arquivo
src/download.py      download (executado uma única vez)
src/markov.py        funções gerais de cadeias de Markov
src/etapa1.py ... src/etapa8.py
tests/               testes automatizados (pytest)
results/tabelas/     tabelas (CSV) usadas no relatório
results/figuras/     figuras usadas no relatório
relatorio/           fonte do relatório e PDF
```

## Figuras e tabelas do relatório

| Item do relatório | Arquivo | Gerado por |
|---|---|---|
| _(a preencher)_ | | |
