from rede.camada import Camada
from rede.rede_neural import RedeNeural
from rede.func_ativ import Sigmoide, Tanh
from interpretadores import carregar_entradas_txt, carregar_saidas_one_hot

# ======================================
# CRIA REDE
# ======================================
rede = RedeNeural()

rede.adicionar_camada(Camada(num_neuronios=60, num_entradas=120, ativacao=Tanh))

rede.adicionar_camada(Camada(num_neuronios=26, num_entradas=60, ativacao=Tanh))

# ======================================
# CARREGAMENTO E DIVISÃO DOS DADOS
# ======================================
dados = carregar_entradas_txt("CARACTERES COMPLETO/X.txt")
saidas = carregar_saidas_one_hot("CARACTERES COMPLETO/Y_letra.txt")

# divide em 80% treino e 20% teste
corte = int(len(dados) * 0.8)
x_treino = dados[:corte]
y_treino = saidas[:corte]

x_teste = dados[corte:]
y_teste = saidas[corte:]

# ======================================
# TREINO
# ======================================
if len(x_treino) > 0:

    epocas_treino = 1000
    lr_treino = 0.01
    print("Salvando estado inicial e hiperparâmetros...")
    rede.salvar_pesos("modelos/pesos_iniciais.json")
    rede.salvar_hiperparametros("modelos/hiperparametros.txt", lr_treino, epocas_treino)
    print("Iniciando treinamento...")
    rede.treinar(x_treino, y_treino, epochs=epocas_treino, taxa_aprendizado=lr_treino)
    rede.salvar_pesos("modelos/pesos_finais.json")
    rede.salvar_historico_erros("modelos/historico_erros.csv")

# ======================================
# TESTE
# ======================================
print("\nIniciando Teste...")
rede.carregar_pesos("modelos/pesos_finais.json")

acertos = 0
total_teste = len(x_teste)

registro_saidas = []

for i in range(total_teste):
    entrada = x_teste[i]
    esperado_one_hot = y_teste[i]

    indice_esperado = esperado_one_hot.index(1)
    resultado_previsto = rede.prever(entrada)

    registro_saidas.append([indice_esperado, resultado_previsto])

    if resultado_previsto == indice_esperado:
        acertos += 1

acuracia = (acertos / total_teste) * 100
print(f"Acurácia no conjunto de teste: {acuracia:.2f}%")

import csv

with open("modelos/saidas_teste.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Esperado (Gabarito)", "Previsto (Rede)"])
    writer.writerows(registro_saidas)

print("Saídas do teste salvas em modelos/saidas_teste.csv")
