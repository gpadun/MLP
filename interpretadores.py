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



def letra_para_one_hot(letra):

    alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # inicia tudo com -1
    vetor = [-1] * 26

    indice = alfabeto.index(letra.upper())

    # posição correta recebe 1
    vetor[indice] = 1

    return vetor


def carregar_saidas_one_hot(arquivo):

    saidas = []

    with open(arquivo, "r") as f:

        for linha in f:

            letra = linha.strip()

            # ignora linhas vazias
            if letra != "":

                saidas.append(
                    letra_para_one_hot(letra)
                )

    return saidas