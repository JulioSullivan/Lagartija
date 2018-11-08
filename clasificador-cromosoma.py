from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from operator import itemgetter
import re
import pdb
import random


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

inputfa = '/home/juliosullivan/Documents/joinP1.fa'

with open(inputfa) as file:
    for line in file:
        Y.append(line.split('\t')[0])

cromosoma = Pipeline([('tfidf', TfidfVectorizer(decode_error='replace', analyzer='char', ngram_range=(3,5), lowercase=False)),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None, n_jobs=-1)),
])


indices = list(range(len(Y)))
in_train, in_test = train_test_split(indices, train_size = 0.7)
getter_train = itemgetter(*in_train)

it = Iterador(inputfa, in_train)
cromosoma.fit(it, getter_train(Y))

""""
test_file = '/home/juliosullivan/Documents/test.fa'

Y_test = []
vect_test = []

with open(test_file) as file:
    for line in file:
        Y_test.append(line.split('\t')[0])
"""
getter_test =  itemgetter(*in_test)
it_test = Iterador(inputfa, in_test)

predicted = cromosoma.predict(it_test)
print(getter_test(Y))
print(predicted)
print(classification_report(getter_test(Y), predicted))