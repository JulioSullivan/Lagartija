from sklearn.externals import joblib
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = np.ravel(classifier)
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

loaded_model = joblib.load('trained_model_joinP1.pkl')
loaded_idf = joblib.load('tfidf_model_joinP1.pkl')

lista_aux = []
indice_peso = []
pesos = loaded_model.coef_
vocab = loaded_idf.vocabulary_

for i in range(pesos.shape[1]):
    indice_peso.append((pesos[0, i], i))

indice_peso.sort(key=lambda tup: tup[0], reverse = True)
vocabA = [0] * len(vocab)

for i in range(20):
    print(indice_peso[i])
    for seq, key in vocab.items():
        vocabA[key] = seq
        if key == indice_peso[i][1]:   
            print(seq)

solo_pesos = []
for tupla in indice_peso:
    solo_pesos.append(tupla[0])
  
features_name = []
plot_coefficients(solo_pesos, vocabA)