# projeto_rede_neural

## Introdução

O presente trabalho apresenta o desenvolvimento e a implementação de um modelo de rede neural do tipo MultiLayer Perceptron (MLP), treinado via algoritmo de retropopagação do erro (BackPropagation). A proposta tem como referência o script exemplo4.py, apresentado na disciplina de Matemática para Ciência de Dados, promovendo alterações estruturais na topologia do modelo e na metodologia de avaliação, além da alteração da base de dados.

O conjunto de dados foi modificado substituindo a base no formato de meia-luas para agrupamentos gaussianos (blobs binarizados), nos quais foram introduzidos um elevado grau de sobreposição entre os grupos e um desbalanceamento amostral severo entre as classes. Além disso, a arquitetura de rede também foi estendida para um modelo mais profundo (*Deep Feedforward*), composto por duas camadas ocultas com três neurônios artificiais, mantendo a camada de saída com um neurônio. Além da função *Sigmoid*, a função *ReLU* (*Rectified Linear Unit*) foi introduzida para análise comparativa nas camadas ocultas. A otimização dos parâmetros foi realizada via gradiente descendente com a função de perda da Entropia de Cruzada Binária (*Binary Cross Entropy - BSE*).

| | `exemplo4.py` | Aqui |
|---|---|---|
| Base de dados | make_moons | make_blobs (desbalanceada de propósito) |
| Camadas ocultas | 1 | 3 |
| Neurônios | 2 → 1 | 8 → 6 → 4 → 1 |
| Ativação das ocultas | Sigmoide | ReLU |
| Ativação da saída | Sigmoide | Sigmoide |
| Função de perda | Erro quadrático | Entropia cruzada |