from rede.camada import Camada
from rede.rede_neural import RedeNeural

from datasets.loader import carregar_entrada_txt


# ======================================
# CRIA REDE
# ======================================

rede = RedeNeural()

# entrada: 120
# oculta: 64
rede.adicionar_camada(
    Camada(
        num_neuronios=60,
        num_entradas=120
    )
)

# saída: 26 letras
rede.adicionar_camada(
    Camada(
        num_neuronios=26,
        num_entradas=64
    )
)


# ======================================
# EXEMPLO DE TREINAMENTO
# ======================================

dados = []
saidas = []

# exemplo:
# entrada_A = carregar_entrada_txt("dados/A/a1.txt")
# dados.append(entrada_A)
#
# saidas.append([
#     1,0,0,0,0,0,0,0,0,0,
#     0,0,0,0,0,0,0,0,0,0,
#     0,0,0,0,0,0
# ])


# ======================================
# TREINO
# ======================================

if len(dados) > 0:

    rede.treinar(
        dados,
        saidas,
        epochs=1000,
        taxa_aprendizado=0.1
    )

    rede.salvar_pesos(
        "modelos/pesos.json"
    )


# ======================================
# TESTE
# ======================================

# rede.carregar_pesos(
#     "modelos/pesos.json"
# )
#
# entrada = carregar_entrada_txt(
#     "dados/teste.txt"
# )
#
# resultado = rede.prever(entrada)
#
# print("Classe prevista:", resultado)