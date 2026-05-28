from rede.camada import Camada
from rede.rede_neural import RedeNeural
from rede.func_ativ import Tanh
from interpretadores import carregar_entradas_txt, carregar_saidas_one_hot
import csv

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

# Conforme especificação: os últimos 130 exemplos são para teste
x_treino = dados[:-130]
y_treino = saidas[:-130]

x_teste = dados[-130:]
y_teste = saidas[-130:]

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

with open("modelos/saidas_teste.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Esperado (Gabarito)", "Previsto (Rede)"])
    writer.writerows(registro_saidas)

print("Saídas do teste salvas em modelos/saidas_teste.csv")

# ======================================
# MATRIZ DE CONFUSÃO E MÉTRICAS
# ======================================
print("\n" + "=" * 40)
print(" Matriz de confusão e métricas")
print("=" * 40)

num_classes = 26
matriz = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

for esperado, previsto in registro_saidas:
    matriz[esperado][previsto] += 1

print("\nMatriz de Confusão (Linha=Esperado, Coluna=Previsto):")
for linha in matriz:
    print(" ".join([f"{val:2d}" for val in linha]))

soma_precisao = 0
soma_recall = 0
soma_f1 = 0
classes_presentes = 0

for c in range(num_classes):
    tp = matriz[c][c]
    fp = sum(matriz[i][c] for i in range(num_classes)) - tp
    fn = sum(matriz[c][i] for i in range(num_classes)) - tp

    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

    if (tp + fn) > 0:
        soma_precisao += precisao
        soma_recall += recall
        soma_f1 += f1
        classes_presentes += 1

if classes_presentes > 0:
    macro_precisao = soma_precisao / classes_presentes
    macro_recall = soma_recall / classes_presentes
    macro_f1 = soma_f1 / classes_presentes
else:
    macro_precisao = macro_recall = macro_f1 = 0

print("\n--- Métricas Gerais (Macro Média) ---")
print(f"Precisão: {macro_precisao:.4f}")
print(f"Recall:   {macro_recall:.4f}")
print(f"F1-Score: {macro_f1:.4f}")
print("=======================================\n")
