from rede.perceptron import Neuronio


# Os neurônios de uma mesma camada vão operar em paralelo, eles não se comunicam entre si, mas todos recebem simultaneamente o mesmo vetor de entrada da camada anterior.
class Camada:
    # Inicializa a camada instanciando 'num_neuronios' independentes.
    # Todos os neurônios desta camada compartilharão a mesma função de ativação e terão o mesmo num_entradas
    def __init__(self, num_neuronios, num_entradas, ativacao):
        self.neuronios = [
            Neuronio(num_entradas, ativacao) for _ in range(num_neuronios)
        ]

    # O processamento da camada (feedforward) é a execução paralela do cálculo matemático (soma ponderada + ativação) de cada neurônio individual.
    # Retorna um vetor contendo as saídas de todos os neurônios desta camada que servirá como o vetor de entrada para a camada seguinte.
    def forward(self, entradas):
        return [neuronio.forward(entradas) for neuronio in self.neuronios]
