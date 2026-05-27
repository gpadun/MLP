from rede.perceptron import Neuronio


class Camada:
    def __init__(self, num_neuronios, num_entradas, ativacao):
        self.neuronios = [
            Neuronio(num_entradas, ativacao) for _ in range(num_neuronios)
        ]

    def forward(self, entradas):
        return [neuronio.forward(entradas) for neuronio in self.neuronios]
