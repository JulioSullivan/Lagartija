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

vect = []
Y = []
X = []

inputfa = 'joinP1.fa'
out_model = 'idf_model.pkl'

with open(inputfa) as file:
    for line in file:
        Y.append(line.split('\t')[0])

cromosoma = Pipeline([
    ('tfidf', TfidfVectorizer(decode_error='replace', 
                analyzer='char', ngram_range=(3,5), 
                lowercase=False)),
    ('svd', TruncatedSVD()),
    ('clf', svm.SVC(gamma='auto', random_state=42, C=5, 
                    decision_function_shape='ovo', degree=1, kernel='linear')),
])

indices = list(range(len(Y)))
in_train, in_test = train_test_split(indices, train_size = 0.7)
getter_train = itemgetter(*in_train)

it = Iterador(inputfa, in_train)
# cromosoma.fit(it, getter_train(Y))

cromosoma.fit(it, getter_train(Y))

getter_test =  itemgetter(*in_test)
it_test = Iterador(inputfa, in_test)

predicted = cromosoma.predict(it_test)
print(getter_test(Y))
print(predicted)

# Luego, guardamos el modelo
joblib.dump(cromosoma.named_steps['tfidf'], out_model, compress = 1)
