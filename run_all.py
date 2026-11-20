import importlib

ETAPAS = [f"src.etapa{i}" for i in range(1, 9)]


def main():
    for nome in ETAPAS:
        print(f"== {nome}")
        importlib.import_module(nome).main()


if __name__ == "__main__":
    main()
