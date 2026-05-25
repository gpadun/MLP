def carregar_entradas_txt(arquivo):
    dados_totais = []

    with open(arquivo, "r") as f:
        for linha in f:
            valores = linha.split(",")

            entrada = [
                0 if int(v.strip()) == -1 else 1 for v in valores if v.strip() != ""
            ]

            if len(entrada) == 120:
                dados_totais.append(entrada)
            elif len(entrada) > 0:
                print(f"Aviso: Linha ignorada. Esperado 120, recebido {len(entrada)}")

    return dados_totais


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

                saidas.append(letra_para_one_hot(letra))

    return saidas
