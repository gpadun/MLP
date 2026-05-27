import json
import os


class RedeNeural:

    def __init__(self):
        self.camadas = []
        self.historico_erros = []

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

            neuronio.delta = erro * neuronio.ativacao.df(neuronio.z)

        # camadas ocultas
        for i in reversed(range(len(self.camadas) - 1)):

            camada_atual = self.camadas[i]
            proxima_camada = self.camadas[i + 1]

            for j, neuronio in enumerate(camada_atual.neuronios):

                erro = 0

                for prox_neuronio in proxima_camada.neuronios:

                    erro += prox_neuronio.pesos[j] * prox_neuronio.delta

                neuronio.delta = erro * neuronio.ativacao.df(neuronio.z)

        # atualização pesos
        for camada in self.camadas:
            for neuronio in camada.neuronios:

                neuronio.pesos += taxa_aprendizado * neuronio.delta * neuronio.entradas

                neuronio.bias += taxa_aprendizado * neuronio.delta

    # ======================================
    # TREINAMENTO
    # ======================================

    def treinar(self, dados, saidas, epochs=1000, taxa_aprendizado=0.1):
        self.historico_erros = []

        for epoch in range(epochs):
            erro_total = 0
            for entrada, esperado in zip(dados, saidas):
                resultado = self.forward(entrada)
                erro_total += sum(
                    (e - r) ** 2 for e, r in zip(esperado, resultado)
                ) / len(esperado)
                self.backpropagation(esperado, taxa_aprendizado)

            self.historico_erros.append(erro_total)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Erro: {erro_total:.4f}")

    # ======================================
    # SALVAR PESOS
    # ======================================

    def salvar_pesos(self, arquivo):

        dados = []
        for camada in self.camadas:
            camada_dados = []
            for neuronio in camada.neuronios:
                camada_dados.append(
                    {
                        # converte o array do NumPy para uma lista normal do python
                        "pesos": neuronio.pesos.tolist(),
                        "bias": float(neuronio.bias),
                    }
                )
            dados.append(camada_dados)

        diretorio = os.path.dirname(arquivo)
        if diretorio:
            os.makedirs(diretorio, exist_ok=True)

        with open(arquivo, "w") as f:
            json.dump(dados, f)

        print(f"Pesos salvos em {arquivo}")

    # ======================================
    # SALVAR HISTÓRICO DE ERROS
    # ======================================
    def salvar_historico_erros(self, arquivo):
        import csv

        diretorio = os.path.dirname(arquivo)
        if diretorio:
            os.makedirs(diretorio, exist_ok=True)

        with open(arquivo, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Erro_MSE"])

            for epoch, erro in enumerate(self.historico_erros):
                writer.writerow([epoch, erro])

        print(f"Histórico de erros salvo em {arquivo}")

    # ======================================
    # SALVAR HIPERPARAMETROS
    # ======================================
    def salvar_hiperparametros(self, arquivo, taxa_aprendizado, epochs):
        diretorio = os.path.dirname(arquivo)
        if diretorio:
            os.makedirs(diretorio, exist_ok=True)

        with open(arquivo, "w") as f:
            f.write("=== Hiperparametros do Modelo ===\n")
            f.write(f"Total de Camadas: {len(self.camadas)}\n")

            for i, camada in enumerate(self.camadas):
                num_neuronios = len(camada.neuronios)
                func_nome = camada.neuronios[0].ativacao.__name__
                f.write(
                    f"  Camada {i+1}: {num_neuronios} neuronios | Ativacao: {func_nome}\n"
                )

            f.write(f"Taxa de Aprendizado: {taxa_aprendizado}\n")
            f.write(f"Epoch de Treinamento: {epochs}\n")

        print(f"Hiperparâmetros salvos em {arquivo}")

    # ======================================
    # CARREGAR PESOS
    # ======================================

    def carregar_pesos(self, arquivo):

        with open(arquivo, "r") as f:
            dados = json.load(f)

        for camada, camada_dados in zip(self.camadas, dados):

            for neuronio, neuronio_dados in zip(camada.neuronios, camada_dados):

                neuronio.pesos = neuronio_dados["pesos"]

                neuronio.bias = neuronio_dados["bias"]

        print(f"Pesos carregados de {arquivo}")
