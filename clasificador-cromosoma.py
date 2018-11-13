from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from operator import itemgetter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import re
import numpy as np
import matplotlib.pyplot as plt
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

inputfa = 'joinP1.fa'

with open(inputfa) as file:
    for line in file:
        Y.append(line.split('\t')[0])
"""
cromosoma = Pipeline([('tfidf', TfidfVectorizer(decode_error='replace', analyzer='char', ngram_range=(3,5), lowercase=False)),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None, n_jobs=-1)),
])
"""
cromosoma = Pipeline([('tfidf', TfidfVectorizer(decode_error='replace', analyzer='char', ngram_range=(3,5), lowercase=False)),
    ('clf', svm.SVC(gamma='auto', random_state=42, max_iter=5)),
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

N_FEATURES_OPTIONS = ['Scaffold']
C_OPTIONS = [0, 1]

param_grid = [
    {
        'reduce_dim': [TfidfVectorizer(decode_error='replace', analyzer='char', ngram_range=(3,5), lowercase=False)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]
reducer_labels = ['TfidfVectorizer', 'KBest(chi2)']

grid = GridSearchCV(cromosoma, cv=5,  n_jobs=-1, param_grid=param_grid)
grid.fit(list(it), getter_train(Y))

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), 1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = 'bgrcmyk'
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Digit classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')

plt.show()