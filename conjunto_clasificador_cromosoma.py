from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import re
import numpy as np
from pdb import set_trace as st
import random
from time import time
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD

class Iterador(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, archivo, indices):
        self.archivo = archivo
        self.indices = indices
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for scaffold in self.generador():
            yield scaffold 

    def generador(self):
        with open(self.archivo) as f:
            for n, line in enumerate(f):
                if n in self.indices:
                    yield re.sub('N','',line.split("\t")[2])

cromosoma = Pipeline([
    ('tfidf', TfidfVectorizer(decode_error='replace', 
                analyzer='char', ngram_range=(3,5), 
                lowercase=False)),
    ('clf', svm.SVC(gamma='auto', random_state=42, C=50, 
                    decision_function_shape='ovo', kernel='linear', degree=1)),
])

parameters = {
    'tfidf__ngram_range':((2, 3), (2, 4), (2, 5), (3,4), (3, 5)),
    #'clf__degree': (1,2),
    'clf__C': (500, 1000, 3000, 5000),
}


for i in range(4, 11):
    vect = []
    Y = []
    X = []

    inputfa = 'Archivos-Join100/joinP'+str(i)+'.fa'
    out_model = 'Modelos-Entrenados-SVM/trained_model_joinP'+str(i)+'.pkl'
    out_tfidf = 'Vocab-TFIDFs-SVM/tfidf_model_joinP'+str(i)+'.pkl'
    resultados_GRID = 'ResultadosGRID-SVM/resultadosGRID_joinP'+str(i)+'.txt'

    with open(inputfa) as file:
        for line in file:
            Y.append(line.split('\t')[0])
            
    indices = list(range(len(Y)))
    in_train, in_test = train_test_split(indices, train_size = 0.7)
    getter_train = itemgetter(*in_train)

    it = Iterador(inputfa, in_train)

    grid = GridSearchCV(cromosoma, cv=3,  n_jobs=15, error_score=0.0 ,param_grid=parameters, verbose=100)

    resGRID = open(resultados_GRID, 'w')
    resGRID.flush()

    resGRID.write("Performing grid search...")
    resGRID.write("\npipeline: cromosoma")
    resGRID.write("\nparameters: " + str(parameters))
    t0 = time()
    grid.fit(list(it), getter_train(Y))
    resGRID.write("\ndone in %0.3fs" % (time() - t0))
    resGRID.write('\n')

    resGRID.write("\nBest score: %0.3f" % grid.best_score_)
    resGRID.write("\nBest parameters set:")
    best_parameters = grid.best_estimator_.get_params()
    for param_name in best_parameters:
        resGRID.write("\n\t%s: %r" % (param_name, best_parameters[param_name]))

    getter_test =  itemgetter(*in_test)
    it_test = Iterador(inputfa, in_test)

    predicted = grid.predict(it_test)
    resGRID.write("\nClassification report:\n")
    resGRID.write(classification_report(getter_test(Y), predicted))

    resGRID.write("\nModel properties:\n")
    #grid.best_estimator_.named_steps['clf'].support_
    resGRID.write("Support size: %s" % np.array2string(grid.best_estimator_.named_steps['clf'].n_support_))
    resGRID.write("Support indices: %s" % np.array2string(grid.best_estimator_.named_steps['clf'].support_))
    # Y así imprime los parametros del modelo en el archivo de resultados
    # ...
    resGRID.close()

    # Luego, guardamos el modelo
    joblib.dump(grid.best_estimator_.named_steps['clf'], out_model, compress = 1)
    joblib.dump(grid.best_estimator_.named_steps['tfidf'], out_tfidf, compress = 1)
