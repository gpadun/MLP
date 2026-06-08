import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "CARACTERES COMPLETO"
ARTEFATOS_DIR = BASE_DIR / "artefatos" / "cnn"

class CNNCaracteres:
    """CNN pequena implementada apenas com NumPy para o extra do EP.

    A rede recebe a imagem no formato 10x12, em vez de achatar tudo em um vetor.
    Isso permite que a convolucao explore a vizinhanca entre pixels.
    """

    def __init__(self, num_filtros=8, tamanho_filtro=3, num_classes=26, seed=42):
        rng = np.random.default_rng(seed)
        self.num_filtros = num_filtros
        self.tamanho_filtro = tamanho_filtro
        self.num_classes = num_classes

        # Inicializacao de He: ajuda camadas com ReLU a comecarem com variancias
        # razoaveis, evitando que os sinais fiquem grandes ou pequenos demais
        escala_conv = np.sqrt(2 / (tamanho_filtro * tamanho_filtro))
        self.filtros = rng.normal(
            0, escala_conv, (num_filtros, tamanho_filtro, tamanho_filtro)
        )
        self.bias_conv = np.zeros(num_filtros)

        # Fluxo das dimensoes:
        # entrada 10x12 -> convolucao 3x3 sem padding gera mapas 8x10
        # -> max-pooling 2x2 reduz para 4x5 -> flatten alimenta a camada densa
        tamanho_flatten = num_filtros * 4 * 5
        escala_dense = np.sqrt(2 / tamanho_flatten)
        self.pesos_dense = rng.normal(0, escala_dense, (tamanho_flatten, num_classes))
        self.bias_dense = np.zeros(num_classes)

    def convolucao(self, imagem):
        # A convolucao desliza filtros 3x3 pela imagem. Cada filtro aprende um
        # tipo de padrao local, como tracos verticais, horizontais ou diagonais
        k = self.tamanho_filtro
        janelas = sliding_window_view(imagem, (k, k))
        return np.einsum("ijab,fab->fij", janelas, self.filtros) + self.bias_conv[
            :, None, None
        ]

    @staticmethod
    def relu(x):
        # ReLU introduz nao linearidade e zera ativacoes negativas
        return np.maximum(0, x)

    @staticmethod
    def softmax(logits):
        # Softmax transforma as 26 saidas em uma distribuicao de probabilidades.
        # Subtrair o maximo melhora a estabilidade numerica da exponencial
        ajustado = logits - np.max(logits)
        exp = np.exp(ajustado)
        return exp / np.sum(exp)

    @staticmethod
    def max_pooling(x, tamanho=2):
        # Max-pooling reduz a resolucao mantendo a ativacao mais forte de cada
        # bloco 2x2. A mascara guarda onde estava o maximo para o backpropagation
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
        # Passagem direta: extrai caracteristicas locais, reduz dimensao e
        # calcula probabilidades finais para as 26 letras
        conv = self.convolucao(imagem)
        ativado = self.relu(conv)
        pool, mascara_pool = self.max_pooling(ativado)
        flatten = pool.reshape(-1)
        logits = flatten @ self.pesos_dense + self.bias_dense
        probabilidades = self.softmax(logits)

        # O cache guarda valores intermediarios necessarios para calcular os
        # gradientes na volta do backpropagation
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

        # Entropia cruzada categorica: penaliza previsoes com baixa probabilidade
        # para a classe correta. Combina naturalmente com a saida softmax
        perda = -np.sum(rotulo_one_hot * np.log(probabilidades + 1e-12))

        # Para softmax + entropia cruzada, o gradiente dos logits simplifica para
        # probabilidade prevista menos o vetor one-hot esperado
        dlogits = probabilidades - rotulo_one_hot
        dpesos_dense = np.outer(cache["flatten"], dlogits)
        dbias_dense = dlogits

        # Propaga o erro da camada densa de volta para o formato dos mapas
        # gerados pelo pooling
        dflatten = self.pesos_dense @ dlogits
        dpool = dflatten.reshape(cache["pool"].shape)

        # No max-pooling, apenas o pixel que foi escolhido como maximo recebe o
        # gradiente. Os demais pixels do bloco 2x2 recebem zero
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

        # Derivada da ReLU: o gradiente passa onde a ativacao antes da ReLU era
        # positiva e e bloqueado onde era negativa
        dconv = dativado * (cache["conv"] > 0)

        # Gradiente dos filtros: soma, para cada posicao, a regiao da imagem
        # multiplicada pelo erro correspondente naquele mapa convolucional
        janelas = sliding_window_view(imagem, (self.tamanho_filtro, self.tamanho_filtro))
        dfiltros = np.einsum("fij,ijab->fab", dconv, janelas)
        dbias_conv = dconv.sum(axis=(1, 2))

        # Gradiente descendente estocastico: atualiza os pesos apos cada amostra
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

    # O dataset original usa pixels -1 e 1. Para a CNN, convertemos para 0 e 1,
    # uma escala mais comum para imagens binarias
    x = ((x + 1) / 2).astype(float)

    # Mantemos a estrutura espacial 10x12. Esta e a diferenca central em relacao
    # a MLP, que trabalha com a imagem achatada em 120 entradas
    x = x.reshape((x.shape[0], 10, 12))
    y = y.astype(float)
    return x, y


def dividir_holdout(x, y):
    # Mantem a mesma ideia do EP principal: as ultimas 130 amostras ficam como
    # teste cego, sem participar do treinamento
    x_treino_total = x[:-130]
    y_treino_total = y[:-130]
    x_teste = x[-130:]
    y_teste = y[-130:]

    # O restante e dividido em treino e validacao para acompanhar generalizacao
    # durante as epocas
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
    # A matriz de confusao registra, para cada classe real, qual classe foi
    # prevista. A diagonal principal representa os acertos
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
    # Macro media calcula a metrica por classe e depois tira a media. Assim,
    # cada letra tem o mesmo peso na avaliacao final
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

    # Hiperparametros escolhidos por teste simples no proprio extra, mantendo a
    # arquitetura pequena para nao desviar do foco do EP
    epocas = 10
    taxa_aprendizado = 0.02
    historico = []
    indices = np.arange(len(x_treino))
    rng = np.random.default_rng(42)

    for epoca in range(1, epocas + 1):
        # Embaralhar a ordem evita que a rede veja sempre as classes na mesma
        # sequencia durante o gradiente descendente estocastico
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
