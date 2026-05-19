import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivada_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)