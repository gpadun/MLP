from rede.camada import Camada
from rede.rede_neural import RedeNeural
from rede.func_ativ import Sigmoide

print("========================================")
print(" INICIANDO TESTE DE MESA ")
print("========================================\n")

# 1. Criando a rede do teste: 2 entradas -> 3 ocultos -> 2 saídas
rede = RedeNeural()
camada_oculta = Camada(num_neuronios=3, num_entradas=2, ativacao=Sigmoide)
camada_saida = Camada(num_neuronios=2, num_entradas=3, ativacao=Sigmoide)

# 2. Forçando os pesos (V) da Camada Oculta conforme o PDF
# O primeiro valor do PDF é o Bias, os seguintes são os pesos
camada_oculta.neuronios[0].bias = -0.1
camada_oculta.neuronios[0].pesos = [0.1, -0.1]

camada_oculta.neuronios[1].bias = -0.1
camada_oculta.neuronios[1].pesos = [0.1, 0.1]

camada_oculta.neuronios[2].bias = 0.1
camada_oculta.neuronios[2].pesos = [-0.1, -0.1]

# 3. Forçando os pesos (W) da Camada de Saída conforme o PDF
camada_saida.neuronios[0].bias = -0.1
camada_saida.neuronios[0].pesos = [0.1, 0.0, 0.1]

camada_saida.neuronios[1].bias = 0.1
camada_saida.neuronios[1].pesos = [-0.1, 0.1, -0.1]

rede.adicionar_camada(camada_oculta)
rede.adicionar_camada(camada_saida)

# 4. Dados do problema
x = [1, 1]
gabarito = [1, 0]
taxa_aprendizado = 0.5

# ======================================
# FORWARD PASS (Cálculo da Saída)
# ======================================
print("--- FASE 1: FORWARD ---")
resultado = rede.forward(x)
print(f"Saída da Rede (y1, y2): {resultado[0]:.4f}, {resultado[1]:.4f}")
print("Gabarito do PDF:        0.4988, 0.5144\n")

# ======================================
# BACKWARD PASS (Ajuste de Pesos)
# ======================================
print("--- FASE 2: BACKPROPAGATION ---")
# Executa a retropropagação na mão para 1 iteração
rede.backpropagation(gabarito, taxa_aprendizado)

print("Novos Pesos - Oculta 1 (Bias, w1, w2):")
print(
    f"Rede: {camada_oculta.neuronios[0].bias:.4f}, {camada_oculta.neuronios[0].pesos[0]:.4f}, {camada_oculta.neuronios[0].pesos[1]:.4f}"
)
print("PDF:  -0.0968, 0.1032, -0.0968\n")

print("Novos Pesos - Saída 1 (Bias, w1, w2, w3):")
print(
    f"Rede: {camada_saida.neuronios[0].bias:.4f}, {camada_saida.neuronios[0].pesos[0]:.4f}, {camada_saida.neuronios[0].pesos[1]:.4f}, {camada_saida.neuronios[0].pesos[2]:.4f}"
)
print("PDF:  -0.0373, 0.1298, 0.0329, 0.1298")
print("========================================")
