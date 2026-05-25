import math


class Sigmoide:
    @staticmethod
    def f(x):
        x = max(-700, min(700, x))
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def df(x):
        s = Sigmoide.f(x)
        return s * (1 - s)


class Tanh:
    @staticmethod
    def f(x):
        return math.tanh(x)

    @staticmethod
    def df(x):
        return 1 - math.tanh(x) ** 2
