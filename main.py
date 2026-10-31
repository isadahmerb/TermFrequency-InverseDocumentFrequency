import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns 

documentos = [
    "Python é uma linguagem de programação poderosa e versátil",
    "Machine learning utiliza algoritmos para aprender padrões nos dados",
    "Análise de dados com Python é muito eficiente e popular",
    "Deep learning é uma subcategoria do machine learning",
    "Visualização de dados ajuda na interpretação de resultados",
    "Pandas é uma biblioteca Python essencial para análise de dados"
]

labels = ['Python', 'ML', 'Análise', 'Deep Learning', 'Visualização', 'Pandas']

vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words=["uma", "é", "e", "para"],
    ngram_range=(1,1),
    lowercase=True,
    token_pattern=r'\b[a-zA-ZÀ-ÿ]+\b'
)

tfidf_matrix = vectorizer.fit_transform(documentos)

feature_names = vectorizer.get_feature_names_out()

tfidf_dense = tfidf_matrix.toarray()

df_tfidf = pd.DataFrame(
    tfidf_dense,
    columns=feature_names,
    index=labels
)

print("Top 5 palavras mais importante por documento:")
for i, doc_label in enumerate(labels):
    top_scores = df_tfidf.iloc[i].nlargest(5)
    print(f"\n{doc_label}:")
    for palavra, score in top_scores.items():
        print(f" {palavra}: {score:.4f}")

similarity_matrix = cosine_similarity(tfidf_matrix)

df_similarity = pd.DataFrame(
    similarity_matrix,
    index=labels,
    columns=labels
)

plt.figure(figsize=(10,8))
sns.heatmap(
    df_similarity,
    annot=True,
    fmt='.3f',
    cmap='coolwarm',
    center=0,
    square=True
)
plt.title('Similaridade de Cosseno entre Documentos em TF-IDF')
plt.tight_layout()
plt.show()

def encontrar_similares(doc_index, top_n=2):
    similarities = df_similarity.iloc[doc_index].sort_values(ascending=False)

    similares = similarities[1:top_n+1]

    print(f"Documentos mais similares a '{labels[doc_index]}':")
    for doc, score in similares.items():
        print(f" {doc}: {score:.4f}")
    
encontrar_similares(0)