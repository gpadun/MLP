import random

from rede.func_ativ import sigmoid


class Neuronio:

    def __init__(self, num_entradas):

        self.pesos = [
            random.uniform(-1.0, 1.0)
            for _ in range(num_entradas)
        ]

        self.bias = random.uniform(-1.0, 1.0)

        self.entradas = []
        self.z = 0
        self.saida = 0
        self.delta = 0

    def forward(self, entradas):

        self.entradas = entradas

        self.z = sum(
            e * p for e, p in zip(entradas, self.pesos)
        ) + self.bias

        self.saida = sigmoid(self.z)

        return self.saida