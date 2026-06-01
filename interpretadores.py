# Módulo de Pré-processamento de Dados, normalizando e codificando as entradas e saídas para a rede neural.


def carregar_entradas_txt(arquivo):
    dados_totais = []

    with open(arquivo, "r") as f:
        for linha in f:
            valores = linha.split(",")

            # Normalização de Dados: converte os valores -1 do dataset original para 0.
            # Redes neurais convergem melhor quando as entradas estão em escalas padronizadas (como 0 e 1), evitando problemas de explosão de gradiente.
            entrada = [
                0 if int(v.strip()) == -1 else 1 for v in valores if v.strip() != ""
            ]

            # Trava de segurança: garante que a dimensionalidade da entrada coincida com a topologia da primeira camada da rede (120 pixels).
            if len(entrada) == 120:
                dados_totais.append(entrada)
            elif len(entrada) > 0:
                print(f"Aviso: Linha ignorada. Esperado 120, recebido {len(entrada)}")

    return dados_totais


# Implementa a técnica de One-Hot Encoding para classificação multiclasse.
# Para 26 letras, criamos um vetor de 26 dimensões onde apenas o índice correspondente à classe correta recebe um sinal alto.
def letra_para_one_hot(letra):
    alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Inicializa o vetor de classes com o valor mínimo de ativação.
    # O uso do -1 está diretamente arquitetado para trabalhar em conjunto com a função Tanh (que varia de -1 a 1) utilizada na camada de saída.
    vetor = [-1] * 26

    indice = alfabeto.index(letra.upper())

    # A posição correta (o gabarito) recebe a ativação máxima (1).
    vetor[indice] = 1

    return vetor


# Processa o arquivo de labels, transformando a sequência de letras brutas em matrizes de vetores One-Hot, que serão essenciais para o cálculo do MSE durante o algoritmo
# do Backpropagation.
def carregar_saidas_one_hot(arquivo):
    saidas = []

    with open(arquivo, "r") as f:
        for linha in f:
            letra = linha.strip()

            # ignora linhas vazias
            if letra != "":
                saidas.append(letra_para_one_hot(letra))

    return saidas
