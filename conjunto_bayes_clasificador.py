from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    ('countV', CountVectorizer(analyzer='char', lowercase=False)), 
    ('clf', MultinomialNB())
])

parameters = {
    'countV__ngram_range': ((2, 3), (2, 4), (2, 5), (3, 4), (3,5)),
}


for i in range(2, 11):
    vect = []
    Y = []
    X = []

    inputfa = 'joinP'+str(i)+'.fa'
    out_model = 'trained_model_joinP'+str(i)+'_bayes.pkl'
    out_tfidf = 'countV_model_joinP'+str(i)+'_bayes.pkl'
    resultados_GRID = 'resultadosGRID_joinP'+str(i)+'_bayes.txt'

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
    # Luego, guardamos el modelo
    joblib.dump(grid.best_estimator_.named_steps['clf'], out_model, compress = 1)
    joblib.dump(grid.best_estimator_.named_steps['countV'], out_tfidf, compress = 1)

    resGRID.write("\nClassification report:\n")
    resGRID.write(classification_report(getter_test(Y), predicted))

    resGRID.write("\nModel properties:\n")
    #grid.best_estimator_.named_steps['clf'].support_
    resGRID.write("Matriz de Probabilidades:  %s" % np.array2string(grid.best_estimator_.named_steps['clf'].feature_log_prob_))
    # Y así imprime los parametros del modelo en el archivo de resultados
    # ...
    resGRID.close()
