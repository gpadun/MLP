import csv
import os
from rede.camada import Camada
from rede.rede_neural import RedeNeural
from rede.func_ativ import Sigmoide


def carregar_csv_logico(caminho_arquivo):
    entradas = []
    saidas_one_hot = []

    with open(caminho_arquivo, "r") as f:
        linhas = f.readlines()

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        valores = linha.split(",")
        if len(valores) >= 3:
            x1 = 0 if int(valores[0]) == -1 else 1
            x2 = 0 if int(valores[1]) == -1 else 1
            y = int(valores[2])

            entradas.append([x1, x2])

            if y == 1:
                saidas_one_hot.append([0, 1])
            else:
                saidas_one_hot.append([1, 0])

    return entradas, saidas_one_hot


arquivos_teste = ["problemAND.csv", "problemOR.csv", "problemXOR.csv"]

for arquivo in arquivos_teste:
    print("\n=======================================")
    print(f" TESTANDO: {arquivo}")
    print("=======================================")

    if not os.path.exists(arquivo):
        print(f"Arquivo '{arquivo}' não encontrado. Pulando...")
        continue

    x_treino, y_treino = carregar_csv_logico(arquivo)

    rede = RedeNeural()

    rede.adicionar_camada(Camada(num_neuronios=4, num_entradas=2, ativacao=Sigmoide))
    rede.adicionar_camada(Camada(num_neuronios=2, num_entradas=4, ativacao=Sigmoide))

    print("Treinando a rede...")
    rede.treinar(x_treino, y_treino, epochs=2000, taxa_aprendizado=0.5)

    print(f"\nResultados")
    acertos = 0
    for entrada, esperado_one_hot in zip(x_treino, y_treino):

        gabarito = esperado_one_hot.index(1)
        previsao = rede.prever(entrada)

        if previsao == gabarito:
            acertos += 1
            status = "ACERTOU"
        else:
            status = "ERROU"

        print(
            f"Entrada: {entrada} | Esperado: {gabarito} | Previsão: {previsao} - {status}"
        )

    acuracia = (acertos / len(x_treino)) * 100
    print(f"Acurácia final: {acuracia:.2f}%")
