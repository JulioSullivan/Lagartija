import itertools
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

scaffolds = []
with open('prueba.fa') as f, open("out.fa", "w") as o:
    row = []
    dna = []
    for n, line in enumerate(f):
        if "REF" in line and n == 0:
            row.append(line.replace('\n',''))
            dna = []
        elif "REF" in line and n > 0:
            dna1d = ''.join(dna)
            #pdb.set_trace()
            row.append(dna1d)
            scaffolds.append(row)
            o.write("%s\t%s\n"%(row[0],row[1]))
            row = []
            dna = []
            dna1d = []
            row.append(line.replace('\n',''))
        else:
            dna.append(line.replace('\n',''))

    dna1d = ''.join(dna)
    row.append(dna1d)
    scaffolds.append(row) 
    o.write("%s\t%s\n"%(row[0],row[1]))

class Iterador(object):
    """
    Iterable: on each iteration, return bag-of-words vectors,
    one vector for each document.
 
    Process one document at a time using generators, never
    load the entire corpus into RAM.
 
    """
    def __init__(self, archivo):
        self.archivo = archivo
 
    def __iter__(self):
        """
        Again, __iter__ is a generator => TxtSubdirsCorpus is a streamed iterable.
        """
        with open(self.archivo) as f:
            for line in f:
                yield line.split("\t")[1]

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


it = Iterador('out.fa')


vectorizer = TfidfVectorizer(decode_error='replace', analyzer='char', ngram_range=(3,10))

#pdb.set_trace()
x_vectorizer = vectorizer.fit_transform(it)
no_topics = 3
nmf = NMF(n_components=no_topics, random_state=1, alpha=0.1, l1_ratio=.5, init='nndsvd').fit(x_vectorizer)

tfidf_feature_names = vectorizer.get_feature_names()
no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)




