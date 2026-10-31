# Pipeline de Similaridade Textual (PLN)

A estrutura do programa segue as seguintes premissas:
 - Converter os textos em vetores TF-IDF
 - Carregar textos em um DataFrame (pandas)
 - Calcular a similaridade de cosseno entre os pares do documento
 - Visualizar o resultado, com bibliotecas gráficas (matplotliv/seaborn)

Essa estrutura simples é a base para muitos sistemas de recomendação.

### TF-IDF
A ideia é reconhecer como o TF-IDF dá nota alta para as palavras que aparecem muito em um documento específico (Term Frequency), mas em poucos documentos de toda a coleção (Inverse Document Frequency).
