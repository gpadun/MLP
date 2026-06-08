import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "CARACTERES COMPLETO"
ARTEFATOS_DIR = BASE_DIR / "artefatos" / "cnn"


class CNNCaracteres:
    """CNN pequena implementada apenas com NumPy para o extra do EP."""

    def __init__(self, num_filtros=8, tamanho_filtro=3, num_classes=26, seed=42):
        rng = np.random.default_rng(seed)
        self.num_filtros = num_filtros
        self.tamanho_filtro = tamanho_filtro
        self.num_classes = num_classes

        escala_conv = np.sqrt(2 / (tamanho_filtro * tamanho_filtro))
        self.filtros = rng.normal(
            0, escala_conv, (num_filtros, tamanho_filtro, tamanho_filtro)
        )
        self.bias_conv = np.zeros(num_filtros)

        # Entrada 10x12 -> convolucao 8x10 -> max-pooling 4x5.
        tamanho_flatten = num_filtros * 4 * 5
        escala_dense = np.sqrt(2 / tamanho_flatten)
        self.pesos_dense = rng.normal(0, escala_dense, (tamanho_flatten, num_classes))
        self.bias_dense = np.zeros(num_classes)

    def convolucao(self, imagem):
        k = self.tamanho_filtro
        janelas = sliding_window_view(imagem, (k, k))
        return np.einsum("ijab,fab->fij", janelas, self.filtros) + self.bias_conv[
            :, None, None
        ]

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(logits):
        ajustado = logits - np.max(logits)
        exp = np.exp(ajustado)
        return exp / np.sum(exp)

    @staticmethod
    def max_pooling(x, tamanho=2):
        filtros, altura, largura = x.shape
        blocos = x.reshape(filtros, altura // tamanho, tamanho, largura // tamanho, tamanho)
        saida = blocos.max(axis=(2, 4))
        mascara = np.zeros_like(x)

        for f in range(filtros):
            for i in range(saida.shape[1]):
                for j in range(saida.shape[2]):
                    linha = i * tamanho
                    coluna = j * tamanho
                    regiao = x[f, linha : linha + tamanho, coluna : coluna + tamanho]
                    posicao = np.unravel_index(np.argmax(regiao), regiao.shape)
                    saida[f, i, j] = regiao[posicao]
                    mascara[f, linha + posicao[0], coluna + posicao[1]] = 1

        return saida, mascara

    def forward(self, imagem):
        conv = self.convolucao(imagem)
        ativado = self.relu(conv)
        pool, mascara_pool = self.max_pooling(ativado)
        flatten = pool.reshape(-1)
        logits = flatten @ self.pesos_dense + self.bias_dense
        probabilidades = self.softmax(logits)

        cache = {
            "imagem": imagem,
            "conv": conv,
            "pool": pool,
            "mascara_pool": mascara_pool,
            "flatten": flatten,
            "probabilidades": probabilidades,
        }
        return probabilidades, cache

    def treinar_amostra(self, imagem, rotulo_one_hot, taxa_aprendizado):
        probabilidades, cache = self.forward(imagem)
        perda = -np.sum(rotulo_one_hot * np.log(probabilidades + 1e-12))

        dlogits = probabilidades - rotulo_one_hot
        dpesos_dense = np.outer(cache["flatten"], dlogits)
        dbias_dense = dlogits

        dflatten = self.pesos_dense @ dlogits
        dpool = dflatten.reshape(cache["pool"].shape)

        dativado = np.zeros_like(cache["conv"])
        for f in range(self.num_filtros):
            for i in range(dpool.shape[1]):
                for j in range(dpool.shape[2]):
                    linha = i * 2
                    coluna = j * 2
                    dativado[
                        f, linha : linha + 2, coluna : coluna + 2
                    ] += dpool[f, i, j] * cache["mascara_pool"][
                        f, linha : linha + 2, coluna : coluna + 2
                    ]

        dconv = dativado * (cache["conv"] > 0)
        janelas = sliding_window_view(imagem, (self.tamanho_filtro, self.tamanho_filtro))
        dfiltros = np.einsum("fij,ijab->fab", dconv, janelas)
        dbias_conv = dconv.sum(axis=(1, 2))

        self.pesos_dense -= taxa_aprendizado * dpesos_dense
        self.bias_dense -= taxa_aprendizado * dbias_dense
        self.filtros -= taxa_aprendizado * dfiltros
        self.bias_conv -= taxa_aprendizado * dbias_conv

        return perda

    def prever(self, imagem):
        probabilidades, _ = self.forward(imagem)
        return int(np.argmax(probabilidades))

    def salvar_pesos(self, caminho):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        dados = {
            "filtros": self.filtros.tolist(),
            "bias_conv": self.bias_conv.tolist(),
            "pesos_dense": self.pesos_dense.tolist(),
            "bias_dense": self.bias_dense.tolist(),
        }
        with caminho.open("w") as arquivo:
            json.dump(dados, arquivo)


def carregar_dados():
    x = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "Y_classe.npy")

    # O dataset original usa -1 e 1. Para a CNN, deixamos pixels em 0 e 1.
    x = ((x + 1) / 2).astype(float)
    x = x.reshape((x.shape[0], 10, 12))
    y = y.astype(float)
    return x, y


def dividir_holdout(x, y):
    x_treino_total = x[:-130]
    y_treino_total = y[:-130]
    x_teste = x[-130:]
    y_teste = y[-130:]

    corte_val = int(len(x_treino_total) * 0.8)
    return (
        x_treino_total[:corte_val],
        y_treino_total[:corte_val],
        x_treino_total[corte_val:],
        y_treino_total[corte_val:],
        x_teste,
        y_teste,
    )


def avaliar(modelo, x, y):
    matriz = np.zeros((26, 26), dtype=int)
    acertos = 0

    for imagem, esperado_one_hot in zip(x, y):
        esperado = int(np.argmax(esperado_one_hot))
        previsto = modelo.prever(imagem)
        matriz[esperado, previsto] += 1
        if previsto == esperado:
            acertos += 1

    acuracia = acertos / len(x) if len(x) else 0
    return acuracia, matriz


def metricas_macro(matriz):
    precisao_total = 0
    recall_total = 0
    f1_total = 0
    classes_presentes = 0

    for classe in range(matriz.shape[0]):
        tp = matriz[classe, classe]
        fp = np.sum(matriz[:, classe]) - tp
        fn = np.sum(matriz[classe, :]) - tp

        precisao = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0

        if tp + fn > 0:
            precisao_total += precisao
            recall_total += recall
            f1_total += f1
            classes_presentes += 1

    if classes_presentes == 0:
        return 0, 0, 0

    return (
        precisao_total / classes_presentes,
        recall_total / classes_presentes,
        f1_total / classes_presentes,
    )


def salvar_historico(historico):
    ARTEFATOS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTEFATOS_DIR / "historico_cnn.csv").open("w", newline="") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["Epoca", "Perda_Treino", "Acuracia_Validacao"])
        writer.writerows(historico)


def salvar_matriz(matriz):
    ARTEFATOS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTEFATOS_DIR / "matriz_confusao_cnn.csv").open("w", newline="") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerows(matriz.tolist())


def main():
    print("=== Extra opcional: CNN em NumPy ===")
    print("Este script nao altera o MLP principal do EP.\n")

    x, y = carregar_dados()
    x_treino, y_treino, x_val, y_val, x_teste, y_teste = dividir_holdout(x, y)

    print(f"Treino:    {len(x_treino)} amostras")
    print(f"Validacao: {len(x_val)} amostras")
    print(f"Teste:     {len(x_teste)} amostras\n")

    modelo = CNNCaracteres(num_filtros=8, tamanho_filtro=3, seed=42)
    epocas = 10
    taxa_aprendizado = 0.02
    historico = []
    indices = np.arange(len(x_treino))
    rng = np.random.default_rng(42)

    for epoca in range(1, epocas + 1):
        rng.shuffle(indices)
        perda_total = 0

        for indice in indices:
            perda_total += modelo.treinar_amostra(
                x_treino[indice], y_treino[indice], taxa_aprendizado
            )

        perda_media = perda_total / len(x_treino)
        acuracia_val, _ = avaliar(modelo, x_val, y_val)
        historico.append([epoca, perda_media, acuracia_val])
        print(
            f"Epoca {epoca:02d} | Perda treino: {perda_media:.4f} | "
            f"Acuracia validacao: {acuracia_val * 100:.2f}%"
        )

    acuracia_teste, matriz = avaliar(modelo, x_teste, y_teste)
    precisao, recall, f1 = metricas_macro(matriz)

    salvar_historico(historico)
    salvar_matriz(matriz)
    modelo.salvar_pesos(ARTEFATOS_DIR / "pesos_cnn.json")

    print("\n--- Resultado final no teste cego ---")
    print(f"Acuracia: {acuracia_teste * 100:.2f}%")
    print(f"Precisao macro: {precisao:.4f}")
    print(f"Recall macro:   {recall:.4f}")
    print(f"F1 macro:       {f1:.4f}")
    print(f"\nArtefatos salvos em: {ARTEFATOS_DIR}")


if __name__ == "__main__":
    main()
