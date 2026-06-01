import math

# Funções de ativação: Sigmoide e Tangente Hiperbólica, cada uma com seus métodos de função e derivada.


# Função Sigmoide: f(x) = 1 / (1 + exp(-x)) e sua derivada df(x) = s * (1 - s), onde s é a saída da função sigmoide.
# Mapeia os valores para o intervalo (0,1), útil para problemas de classificação binária.
class Sigmoide:
    @staticmethod
    def f(x):
        # para evitar overflow, limitamos o valor de x
        x = max(-700, min(700, x))
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def df(x):
        # a derivada da sigmoide usa a própria função sigmoide para calcular o valor
        s = Sigmoide.f(x)
        return s * (1 - s)


# Tangente Hiperbólica: f(x) = tanh(x) e sua derivada df(x) = 1 - tanh(x)^2.
# Mapeia os valores para o intervalo (-1,1), com centro em 0, o que geralmente ajuda a convergir mais rápido durante o treinamento.
class Tanh:
    @staticmethod
    def f(x):
        return math.tanh(x)

    @staticmethod
    def df(x):
        return 1 - math.tanh(x) ** 2
