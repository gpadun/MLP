import random
import numpy as np


# Classe Neuronio: recebe como parâmetro o número de entradas e a função de ativação.
# Inicializa os pesos e o bias com valores aleatórios pequenos com a função np.random.uniform para os pesos e random.uniform para o bias.
class Neuronio:
    def __init__(self, num_entradas, ativacao):
        self.pesos = np.random.uniform(-0.1, 0.1, num_entradas)
        self.bias = random.uniform(-0.1, 0.1)

        self.ativacao = ativacao
        self.entradas = None
        self.z = 0
        self.saida = 0
        self.delta = 0

    # Método que executa o feedforward, calculando a coma ponderada (entradas * pesos + bias) e aplicando a função de ativação para obter a saída do neurônio.
    def forward(self, entradas):
        self.entradas = np.array(entradas)
        self.z = np.dot(self.entradas, self.pesos) + self.bias
        self.saida = self.ativacao.f(self.z)

        return self.saida
