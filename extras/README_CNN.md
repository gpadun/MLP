# Extra opcional: CNN em NumPy

Este extra implementa uma rede neural convolucional simples para o mesmo problema
de reconhecimento de caracteres do EP. Ele fica separado do codigo principal:
`main.py`, `rede/` e os artefatos originais da MLP nao sao alterados.

## Como executar

Na raiz do projeto:

```bash
python extras/cnn_numpy.py
```

O script usa os arquivos:

- `CARACTERES COMPLETO/X.npy`
- `CARACTERES COMPLETO/Y_classe.npy`

E salva os resultados em:

- `artefatos/cnn/historico_cnn.csv`
- `artefatos/cnn/matriz_confusao_cnn.csv`
- `artefatos/cnn/pesos_cnn.json`
- `artefatos/cnn/grafico_treino_cnn.png`
- `artefatos/cnn/exemplos_predicoes_cnn.png`

## O que foi implementado

- Camada convolucional 2D com 8 filtros 3x3.
- Funcao de ativacao ReLU.
- Max-pooling 2x2.
- Camada densa final com softmax.
- Funcao de perda de entropia cruzada categorica.
- Treinamento por gradiente descendente estocastico.
- Separacao hold-out: ultimas 130 amostras para teste cego e 20% do restante para validacao.
- Avaliacao com acuracia, matriz de confusao, precisao macro, recall macro e F1
  macro.
- Grafico da perda de treino e acuracia de validacao.
- Grade com exemplos de predicoes no teste cego.

## Resultados obtidos

Com 10 epocas e taxa de aprendizado 0.02, o teste cego gerou:

- Acuracia: 80.00%
- Precisao macro: 0.8533
- Recall macro: 0.8000
- F1 macro: 0.7958

## Itens opcionais contemplados

- Implementacao da CNN.
- Outra funcao de perda: entropia cruzada categorica.
- Uso de hold-out no experimento da CNN.

## Por que isso conta como extra

A MLP do EP recebe a imagem achatada como um vetor de 120 entradas. A CNN trata a
entrada como uma imagem 10x12, preservando a vizinhanca espacial dos pixels. Os
filtros convolucionais aprendem padroes locais, como tracos horizontais,
verticais e diagonais, antes da classificacao final.
