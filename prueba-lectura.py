import itertools
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import gensim
from gensim import corpora, models, similarities
import pickle

class Iterador(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, archivo, analyzer):
        self.archivo = archivo
        self.analyzer = analyzer
        self.dictionary = gensim.corpora.Dictionary(self.generador())
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        for scaffold in self.generador():
            yield self.dictionary.doc2bow(scaffold) 

    def generador(self):
        with open(self.archivo) as f:
            for line in f:
                yield self.analyzer(re.sub('N','',line.split("\t")[1]))


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]]))

vectorizer = TfidfVectorizer(decode_error='replace', analyzer='char', ngram_range=(3,5), lowercase=False)

dic = open('dic.txt', 'wb')
# Crea el vocabulario o el iterador
it = Iterador('completo.fa', vectorizer.build_analyzer())

pdb.set_trace()
# Creación del tfidf de acuerdo al vocabulario
tfidf = models.TfidfModel(it)

# Dependiendo de la primea lectura, vuelve a evaluar los datos
corpus_tfidf = tfidf[it]
 

no_topics = 3
lsi = models.LsiModel(corpus_tfidf, id2word=it.dictionary, num_topics=no_topics)
lsi.print_topics(no_topics)
##corpus_lsi = lsi[corpus_tfidf]
pdb.set_trace()




