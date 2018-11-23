from sklearn.externals import joblib
import pdb

loaded_model = joblib.load('trained_model_joinP10.pkl')
loaded_idf = joblib.load('tfidf_model_joinP10.pkl')

lista_aux = []
indice_peso = []
pesos = loaded_model.coef_
vocab = loaded_idf.vocabulary_

for i in range(pesos.shape[1]):
    indice_peso.append((pesos[0, i], i))
    
indice_peso.sort(key=lambda tup: tup[0], reverse = True)

for i in range(10):
    print(indice_peso[i])
    for n, seq in enumerate(vocab):
        if n == indice_peso[i][1]:   
            print(seq)