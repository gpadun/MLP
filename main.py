from rede.camada import Camada
from rede.rede_neural import RedeNeural

from interpretadores import carregar_entrada_txt


# ======================================
# CRIA REDE
# ======================================

rede = RedeNeural()

# entrada: 120, as imagens tem 10 de altura e 12 de largura
# oculta: 60
rede.adicionar_camada(
    Camada(
        num_neuronios=60,
        num_entradas=120
    )
)

# saída: 26 letras, 26 neurônios para possibilita em one hot
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
# entrada_A = carregar_entrada_txt("CARACTERES COMPLETO/X.txt")
# dados.append(entrada_A)
#saida_A=letra_para_one_hot()
# saidas.append(saida_A)


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
# entrada = carregar_entrada_txt(datasets.CARACTERES COMPLETO)
#
# resultado = rede.prever(entrada)
#
# print("Classe prevista:", resultado)