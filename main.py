from rede.camada import Camada
from rede.rede_neural import RedeNeural
from rede.func_ativ import Tanh
from interpretadores import carregar_entradas_txt, carregar_saidas_one_hot
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# ======================================
# 1. CRIAÇÃO DA REDE NEURAL (ARQUITETURA)
# ======================================
print(" [1/4] INICIALIZANDO A REDE NEURAL")

# Instancia a rede e define a topologia baseada na dimensionalidade do problema.
rede = RedeNeural()

# Camada Oculta: Recebe os 120 pixels da imagem. O número de 60 neurônios foi
# escolhido empiricamente para extrair características suficientes sem onerar o processamento.
rede.adicionar_camada(Camada(num_neuronios=60, num_entradas=120, ativacao=Tanh))

# Camada de Saída: Recebe os 60 sinais da camada oculta e mapeia para 26 neurônios,
# representando o alfabeto (One-Hot Encoding). A função Tanh (-1 a 1) é usada para
# estabilizar o gradiente em conjunto com os rótulos do dataset.
rede.adicionar_camada(Camada(num_neuronios=26, num_entradas=60, ativacao=Tanh))
print("-> Topologia: Entrada (120) -> Oculta (60) -> Saída (26)\n")

# ======================================
# 2. PREPARAÇÃO DOS DADOS (HOLD-OUT)
# ======================================
print(" [2/4] CARREGANDO E DIVIDINDO DADOS")
dados = carregar_entradas_txt("CARACTERES COMPLETO/X.txt")
saidas = carregar_saidas_one_hot("CARACTERES COMPLETO/Y_letra.txt")

# Separação cega (Blind Test): Conforme especificação, os últimos 130 exemplos
# são isolados. A rede nunca verá esses dados durante o ajuste de pesos.
x_treino_total = dados[:-130]
y_treino_total = saidas[:-130]

x_teste = dados[-130:]
y_teste = saidas[-130:]

# Estratégia de Hold-out para Parada Antecipada: Dividimos o restante dos dados
# em 80% para Treinamento (ajuste do gradiente) e 20% para Validação (monitoramento de overfitting).
corte_val = int(len(x_treino_total) * 0.8)

x_treino = x_treino_total[:corte_val]
y_treino = y_treino_total[:corte_val]

x_val = x_treino_total[corte_val:]
y_val = y_treino_total[corte_val:]

print(f"-> Conjunto de Treinamento: {len(x_treino)} amostras")
print(f"-> Conjunto de Validação:   {len(x_val)} amostras")
print(f"-> Conjunto de Teste:       {len(x_teste)} amostras\n")

# ======================================
# 3. TREINAMENTO E PARADA ANTECIPADA (EARLY STOPPING)
# ======================================
print(" [3/4] TREINAMENTO COM PARADA ANTECIPADA")
if len(x_treino) > 0:
    # Hiperparâmetros de controle de convergência
    epocas_treino = 2000
    lr_treino = 0.01

    print("-> Salvando pesos iniciais e hiperparâmetros...")
    rede.salvar_pesos("artefatos/pesos_iniciais.json")
    rede.salvar_hiperparametros(
        "artefatos/hiperparametros.txt", lr_treino, epocas_treino
    )

    print(f"-> Iniciando treinamento (Máx Épocas: {epocas_treino} | LR: {lr_treino})")

    # O treinamento é executado avaliando o conjunto de validação ao fim de cada época.
    # Se o erro de validação subir consistentemente (paciencia=30), o treino é abortado
    # para evitar que a rede "decore" o dataset (perda de generalização).
    rede.treinar(
        x_treino,
        y_treino,
        x_val,
        y_val,
        epochs=epocas_treino,
        taxa_aprendizado=lr_treino,
        paciencia=30,
    )

    # Salva o estado final da rede após o Early Stopping restaurar os melhores pesos
    rede.salvar_pesos("artefatos/pesos_finais.json")
    rede.salvar_historico_erros("artefatos/historico_erros.csv")
    print("-> Treinamento concluído. Artefatos salvos na pasta 'artefatos/'.\n")

# ======================================
# 4. AVALIAÇÃO (INFERÊNCIA) E MÉTRICAS
# ======================================
print(" [4/4] TESTE FINAL E MÉTRICAS")
print("-> Carregando os melhores pesos para o teste...")
# Garante que estamos testando a versão da rede que melhor generalizou
rede.carregar_pesos("artefatos/pesos_finais.json")

acertos = 0
total_teste = len(x_teste)
registro_saidas = []

for i in range(total_teste):
    entrada = x_teste[i]
    esperado_one_hot = y_teste[i]

    # Extrai o índice da letra correta a partir do vetor One-Hot
    indice_esperado = esperado_one_hot.index(1)

    # Faz o Forward Pass e pega o neurônio de saída com maior ativação
    resultado_previsto = rede.prever(entrada)

    registro_saidas.append([indice_esperado, resultado_previsto])

    if resultado_previsto == indice_esperado:
        acertos += 1

acuracia = (acertos / total_teste) * 100
print(f"-> Acurácia no conjunto de teste: {acuracia:.2f}%\n")

with open("artefatos/saidas_teste.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["esperado (gabarito)", "previsto (rede)"])
    writer.writerows(registro_saidas)

# ======================================
# 5. MATRIZ DE CONFUSÃO E MÉTRICAS (Versão Visual)
# ======================================
print("\n" + "=" * 40)
print(" Matriz de confusão e métricas")
print("=" * 40)

num_classes = 26
matriz = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

# Preenche a matriz com os resultados do teste
for esperado, previsto in registro_saidas:
    matriz[esperado][previsto] += 1

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

# ======================================
# 6. PLOTAGEM DA MATRIZ
# ======================================
matriz_np = np.array(matriz)
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz_np,
    display_labels=[chr(i) for i in range(ord('A'), ord('Z') + 1)] # Transforma índices em A-Z
)

fig, ax = plt.subplots(figsize=(14, 14))

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=90
)

plt.title("Matriz de Confusão do Teste (MLP)")
plt.show()
