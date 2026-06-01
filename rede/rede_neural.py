import json
import os
import copy
import csv


class RedeNeural:
    def __init__(self):
        self.camadas = []
        self.historico_erros = []  # Agora vai armazenar tuplas: (erro_treino, erro_val)

    def adicionar_camada(self, camada):
        self.camadas.append(camada)

    # ======================================
    # FEEDFORWARD
    # ======================================
    # Propaga o sinal de entrada pela rede, camada por camada.
    # É o processo de decisão da rede, usado no treino e no teste.
    def forward(self, entradas):
        for camada in self.camadas:
            entradas = camada.forward(entradas)
        return entradas

    # ======================================
    # PREVER
    # ======================================
    # Passa o dado pela rede e retorna o índice do neurônio com maior ativação.
    # Como as letras estão em One-Hot, o índice é a própria resposta.
    def prever(self, entradas):
        saida = self.forward(entradas)
        return saida.index(max(saida))

    # ======================================
    # BACKPROPAGATION
    # ======================================
    # Algoritmo de retropropagação do erro para ajuste dos pesos sinápticos.
    # Usa a regra da cadeia para calcular o gradiente do erro em relação a cada peso.
    def backpropagation(self, esperado, taxa_aprendizado):
        camada_saida = self.camadas[-1]

        # Cálculo do erro na camada de saída
        # delta = (esperado - calculado) * derivada_da_funcao_de_ativacao(z)
        for i, neuronio in enumerate(camada_saida.neuronios):
            erro = esperado[i] - neuronio.saida
            neuronio.delta = erro * neuronio.ativacao.df(neuronio.z)

        # Retropropagação para as camadas ocultas
        # O erro de um neurônio oculto é a soma ponderada dos deltas da camada seguinte.
        for i in reversed(range(len(self.camadas) - 1)):
            camada_atual = self.camadas[i]
            proxima_camada = self.camadas[i + 1]

            for j, neuronio in enumerate(camada_atual.neuronios):
                erro = 0
                for prox_neuronio in proxima_camada.neuronios:
                    erro += prox_neuronio.pesos[j] * prox_neuronio.delta
                neuronio.delta = erro * neuronio.ativacao.df(neuronio.z)

        # Atualização dos pesos (Gradiente Descendente)
        # novo_peso = peso_antigo + (taxa_aprendizado * delta * entrada)
        for camada in self.camadas:
            for neuronio in camada.neuronios:
                neuronio.pesos += taxa_aprendizado * neuronio.delta * neuronio.entradas
                neuronio.bias += taxa_aprendizado * neuronio.delta

    # ======================================
    # TREINAMENTO COM PARADA ANTECIPADA
    # ======================================
    def treinar(
        self,
        x_treino,
        y_treino,
        x_val=None,
        y_val=None,
        epochs=1000,
        taxa_aprendizado=0.1,
        paciencia=20,
    ):
        self.historico_erros = []
        melhor_erro_val = float("inf")
        epocas_sem_melhoria = 0
        melhores_pesos = None

        for epoch in range(epochs):
            erro_treino = 0

            # Feedforward e Backpropagation para o conjunto de treino
            for entrada, esperado in zip(x_treino, y_treino):
                resultado = self.forward(entrada)
                # Cálculo do Erro Quadrático Médio (MSE)
                erro_treino += sum(
                    (e - r) ** 2 for e, r in zip(esperado, resultado)
                ) / len(esperado)
                self.backpropagation(esperado, taxa_aprendizado)

            erro_treino /= len(x_treino)  # MSE real da época

            # Lógica do conjunto de validação e parada antecipada
            if x_val and y_val:
                erro_val = 0
                # Apenas Feedforward para avaliar a generalização
                for entrada, esperado in zip(x_val, y_val):
                    resultado = self.forward(entrada)
                    erro_val += sum(
                        (e - r) ** 2 for e, r in zip(esperado, resultado)
                    ) / len(esperado)

                erro_val /= len(x_val)
                self.historico_erros.append((erro_treino, erro_val))

                # Verifica se houve melhoria na generalização da rede
                if erro_val < melhor_erro_val:
                    melhor_erro_val = erro_val
                    epocas_sem_melhoria = 0
                    # Deepcopy congela o estado exato dos objetos na memória
                    melhores_pesos = copy.deepcopy(self.camadas)
                else:
                    epocas_sem_melhoria += 1

                if epoch % 50 == 0:
                    print(
                        f"Epoch {epoch} | Erro Treino: {erro_treino:.4f} | Erro Validação: {erro_val:.4f}"
                    )

                # Critério de parada: erro de validação parou de cair
                if epocas_sem_melhoria >= paciencia:
                    print(f"\n[!] Parada antecipada acionada na época {epoch}!")
                    print(
                        f"[!] Restaurando os melhores pesos (Erro Validação: {melhor_erro_val:.4f})"
                    )
                    self.camadas = melhores_pesos
                    break
            else:
                self.historico_erros.append((erro_treino, 0))
                if epoch % 50 == 0:
                    print(f"Epoch {epoch} | Erro Treino: {erro_treino:.4f}")

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
        diretorio = os.path.dirname(arquivo)
        if diretorio:
            os.makedirs(diretorio, exist_ok=True)

        with open(arquivo, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Erro_Treino", "Erro_Validacao"])

            for epoch, (erro_t, erro_v) in enumerate(self.historico_erros):
                writer.writerow([epoch, erro_t, erro_v])
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
