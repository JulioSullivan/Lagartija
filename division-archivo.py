import itertools
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import gensim
from gensim import corpora, models, similarities

scaffolds = []
with open('/home/juliosullivan/Documents/Anolis_carolinensis.AnoCar2.0.dna.chromosome.1.fa') as f, open("out.fa", "w") as o:
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