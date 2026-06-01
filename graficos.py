import csv
import matplotlib.pyplot as plt


def plotar_historico_erros(arquivo_csv):
    epocas = []
    erros_treino = []
    erros_val = []

    try:
        with open(arquivo_csv, "r") as f:
            leitor = csv.reader(f)
            next(leitor)

            for linha in leitor:
                epocas.append(int(linha[0]))
                erros_treino.append(float(linha[1]))
                erros_val.append(float(linha[2]))

    except FileNotFoundError:
        print(f"Erro: Arquivo {arquivo_csv} não encontrado.")
        print("Execute o main.py primeiro para gerar o histórico de erros.")
        return

    # Criação do Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(epocas, erros_treino, label="Erro de Treino", color="blue", linewidth=2)
    plt.plot(epocas, erros_val, label="Erro de Validação", color="orange", linewidth=2)

    # Estilização
    plt.title("Convergência do Erro (MSE) ao longo das Épocas", fontsize=14, pad=15)
    plt.xlabel("Épocas", fontsize=12)
    plt.ylabel("Erro Quadrático Médio (MSE)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Destaca o ponto de parada (onde o erro de validação foi mínimo)
    melhor_epoca = erros_val.index(min(erros_val))
    melhor_erro = min(erros_val)
    plt.axvline(
        x=melhor_epoca, color="red", linestyle="--", alpha=0.5, label="Melhor Validação"
    )
    plt.scatter(melhor_epoca, melhor_erro, color="red", zorder=5)

    # Salva o gráfico na pasta artefatos
    caminho_salvamento = "artefatos/grafico_convergencia.png"
    plt.savefig(caminho_salvamento, dpi=300, bbox_inches="tight")
    print(f"Gráfico gerado com sucesso e salvo em: {caminho_salvamento}")

    # Exibe o gráfico na tela
    plt.show()


if __name__ == "__main__":
    arquivo_historico = "artefatos/historico_erros.csv"
    plotar_historico_erros(arquivo_historico)
