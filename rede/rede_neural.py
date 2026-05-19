import json

from rede.func_ativ import derivada_sigmoid


class RedeNeural:

    def __init__(self):

        self.camadas = []

    def adicionar_camada(self, camada):
        self.camadas.append(camada)

    # ======================================
    # FORWARD
    # ======================================

    def forward(self, entradas):

        for camada in self.camadas:
            entradas = camada.forward(entradas)

        return entradas

    # ======================================
    # PREVER
    # ======================================

    def prever(self, entradas):

        saida = self.forward(entradas)

        return saida.index(max(saida))

    # ======================================
    # BACKPROPAGATION
    # ======================================

    def backpropagation(self, esperado, taxa_aprendizado):

        camada_saida = self.camadas[-1]

        # camada de saída
        for i, neuronio in enumerate(camada_saida.neuronios):

            erro = esperado[i] - neuronio.saida

            neuronio.delta = (
                erro * derivada_sigmoid(neuronio.z)
            )

        # camadas ocultas
        for i in reversed(range(len(self.camadas) - 1)):

            camada_atual = self.camadas[i]
            proxima_camada = self.camadas[i + 1]

            for j, neuronio in enumerate(camada_atual.neuronios):

                erro = 0

                for prox_neuronio in proxima_camada.neuronios:

                    erro += (
                        prox_neuronio.pesos[j]
                        * prox_neuronio.delta
                    )

                neuronio.delta = (
                    erro * derivada_sigmoid(neuronio.z)
                )

        # atualização pesos
        for camada in self.camadas:

            for neuronio in camada.neuronios:

                for i in range(len(neuronio.pesos)):

                    neuronio.pesos[i] += (
                        taxa_aprendizado
                        * neuronio.delta
                        * neuronio.entradas[i]
                    )

                neuronio.bias += (
                    taxa_aprendizado
                    * neuronio.delta
                )

    # ======================================
    # TREINAMENTO
    # ======================================

    def treinar(
        self,
        dados,
        saidas,
        epochs=1000,
        taxa_aprendizado=0.1
    ):

        for epoch in range(epochs):

            erro_total = 0

            for entrada, esperado in zip(dados, saidas):

                resultado = self.forward(entrada)

                erro_total += sum(
                    (e - r) ** 2
                    for e, r in zip(esperado, resultado)
                )

                self.backpropagation(
                    esperado,
                    taxa_aprendizado
                )

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch} | Erro: {erro_total:.4f}"
                )

    # ======================================
    # SALVAR PESOS
    # ======================================

    def salvar_pesos(self, arquivo):

        dados = []

        for camada in self.camadas:

            camada_dados = []

            for neuronio in camada.neuronios:

                camada_dados.append({
                    "pesos": neuronio.pesos,
                    "bias": neuronio.bias
                })

            dados.append(camada_dados)

        with open(arquivo, "w") as f:
            json.dump(dados, f)

        print(f"Pesos salvos em {arquivo}")

    # ======================================
    # CARREGAR PESOS
    # ======================================

    def carregar_pesos(self, arquivo):

        with open(arquivo, "r") as f:
            dados = json.load(f)

        for camada, camada_dados in zip(
            self.camadas,
            dados
        ):

            for neuronio, neuronio_dados in zip(
                camada.neuronios,
                camada_dados
            ):

                neuronio.pesos = neuronio_dados["pesos"]

                neuronio.bias = neuronio_dados["bias"]

        print(f"Pesos carregados de {arquivo}")