from sklearn.externals import joblib
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np

def plot_coefficients(coef, feature_names, top_features=20):
    top_positive_coefficients = coef[-top_features:]
    top_negative_coefficients = coef[:top_features]
    top_coefficients = top_negative_coefficients + top_positive_coefficients
    
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c[0] < 0 else 'blue' for c in top_coefficients]
    plt.bar(np.arange(2 * top_features), [top_w[0] for top_w in top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    print(top_coefficients)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[[top_w[1] for top_w in top_coefficients]], rotation=60, ha='right')
    plt.show()

loaded_model = joblib.load('Modelos-Entrenados-SVM/trained_model_joinP10.pkl')
loaded_idf = joblib.load('Vocab-TFIDFs-SVM/tfidf_model_joinP10.pkl')

lista_aux = []
indice_peso = []
pesos = loaded_model.coef_
vocab = loaded_idf.vocabulary_

for i in range(pesos.shape[1]):
    indice_peso.append((pesos[0, i], i))

indice_peso.sort(key=lambda tup: tup[0], reverse = False)
vocabA = [0] * len(vocab)

for i in range(20):
    print(indice_peso[i])
    for seq, key in vocab.items():
        vocabA[key] = seq.rstrip('\n')
        if key == indice_peso[i][1]:   
            print(seq)

solo_pesos = []
for tupla in indice_peso:
    solo_pesos.append(tupla[0])
  
features_name = []
plot_coefficients(indice_peso, vocabA)