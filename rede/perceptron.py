import random
import numpy as np


class Neuronio:

    def __init__(self, num_entradas, ativacao):
        # inicia os pesos como um array do numpy
        self.pesos = np.random.uniform(-0.1, 0.1, num_entradas)
        self.bias = random.uniform(-0.1, 0.1)

        self.ativacao = ativacao
        self.entradas = None
        self.z = 0
        self.saida = 0
        self.delta = 0

    def forward(self, entradas):
        self.entradas = np.array(entradas)
        self.z = np.dot(self.entradas, self.pesos) + self.bias
        self.saida = self.ativacao.f(self.z)

        return self.saida
