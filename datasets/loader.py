def carregar_entrada_txt(arquivo):

    with open(arquivo, "r") as f:

        conteudo = f.read()

    valores = conteudo.split(",")

    entrada = [
        0 if int(v.strip()) == -1 else 1
        for v in valores
    ]

    if len(entrada) != 120:

        raise ValueError(
            f"Esperado 120 valores, recebido {len(entrada)}"
        )

    return entrada