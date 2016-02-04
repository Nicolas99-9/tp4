from __future__ import division

#Indiquez vos paths ici
path = "/people/labeau/Theano/theano/Script_RNN/data/"
filename = "newsco.en"
test_filename = "newstest2009.en"

import os
filepath = os.path.join(path, filename)
test_filepath = os.path.join(path, test_filename)

#1. Obtenir, a partir d'un fichier, un dictionnaire contenant les mots presents dans le texte comme cles, et leur compte comme valeur associee.

# En comptant ligne par ligne:
import codecs
import collections
def get_counts(filepath):
    counts = collections.defaultdict(int)
    with codecs.open(filepath, "rt") as file:
        for line in file:
            words = line.strip().split()
            for w in words:
                counts[w] += 1
    return counts

# Solution compacte, en utilisant la fonction collections du modeule Counter:
def get_counts_short(filepath):
    with codecs.open(filepath, "rt") as file:
        data = file.read().strip().split()
        counts = collections.Counter(data)
    return counts

counts = get_counts_short(filepath)

#2. Utiliser la fonction pyplot du module matplotlib, pour obtenir une courbe du rang des mots en fonction de leur frequence.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_counts(counts):
    y = sorted(counts.values(),reverse=True)
    x = range(1, len(y) + 1)

    plt.figure()
    plt.loglog()
    plt.ylabel('Frequencies')
    plt.xlabel('Ranks')
    plt.title("Frequencies versus ranks: visualisation of Zipf's law ")
    plt.plot(x,y,'r+')
    plt.savefig("zipf.png")

plot_counts(counts)

#3/4. Obtenir a partir de ces comptes, un modele de langue unigram, et calculer la probabilite des premieres phrases du fichier.
#  Essayer avec des phrases d'un autre fichier: corriger le probleme qui se pose avec du Add-one smoothing.

#5. Creer un modele de langue Bigram, et calculer la probabilite des premieres phrases du fichier.

import random
import math

class UnigramModel():
    # On peut creer une classe, avec un constructeur qui prendra en argument le chemin du fichier qui permettra de construire le modele
    # On ajoute comme attribut le nombre de mots differents et le nombre de mots total du fichier
    # On indique aussi si on utilise le Add-one smoothing
    def __init__(self, path, smoothing=True):
        with codecs.open(path, 'r') as f:
            self.counts = collections.Counter(f.read().strip().split())
            self.v = len(self.counts.keys())
            self.n = sum(self.counts.values())
            self.default = 1 if smoothing else 0

    # Permet d'appeller la classe sur une sequence de mot, et renvoie sa log probabilite. 
    def __getitem__(self, sent):
        return sum([ math.log(self.counts.get(word, 0) + self.default) - math.log( self.n + self.default*self.v ) for word in sent ])
                   
    # Genere un entier aleatoire qui indiquera quel mot le modele choisit. Plus un mot est frequent dans les compte, plus il a de chances d'etre choisi
    def sample(self, length):
        words = []
        for i in range(length):
            r = random.uniform(0, self.n)
            s = 0
            for word, count in self.counts.iteritems():
                s += count 
                if r < s: break
            words.append(word)
        return ' '.join(words)

Model_1 = UnigramModel(filepath)

from nltk.util import ngrams 

class BigramModel():
    # Similaire a unigram, mais on collecte aussi les bigrams.     
    def __init__(self, path, smoothing=True):
        with codecs.open(path, 'r') as f:
            n_grams = []
            words = []
            for sent in f:
                n_grams.extend(ngrams(sent.strip().split(),2))
                words.extend(sent.strip().split())
        self.counts = collections.Counter(words)
        self.n_grams = collections.Counter(n_grams)
        self.v = len(self.counts.keys())
        self.n = sum(self.counts.values())
        self.n_obs = sum(self.n_grams.values())
        self.default = 1 if smoothing else 0

    # On cree une methode qui permet de retourner le dictionnaire des mots qui vont suivre un autre mot
    def cond_count(self, cont):
        return dict([(word, count) for (cont, word), count in self.n_grams.iteritems()])

    # On cree une methode qui va utiliser la precedente pour calculer la probabilite conditionelle d'un mot selon un autre
    def cond_prob(self, cont, word):
        cond_dict = self.cond_count(cont)
        return math.log( cond_dict.get(word,0) + self.default ) - math.log( self.counts.get(cont, 0) )

    # Cela permet de creer une methode qui va calculer la log probabilite d'une sequence
    def __getitem__(self, sent):
        wordf = sent[0]
        n_grams = ngrams(sent,2)
        logprob_wordf = math.log(self.counts.get(wordf, self.default)) - math.log(self.n + self.default*self.v)
        logprob_cond = [ self.cond_prob( n_gram[0], n_gram[1] ) for n_gram in n_grams ]
        return  sum(logprob_cond) + logprob_wordf

    def sample(self, length):
        # Premier mot
        r = random.uniform(0, self.n)
        s = 0
        for word, count in self.counts.iteritems():
            s += count
            if r < s: break
        words = [word]
        # Reste de la sequence
        for i in range(length-1):
            current_counts = self.cond_count(words[-1])
            r = random.uniform(0, sum(current_counts.values()))
            s = 0
            for word, count in current_counts.iteritems():
                s += count
                if r < s: break
            words.append(word)
        return ' '.join(words)

Model_2 = BigramModel(filepath)

with codecs.open(filepath, 'rt') as f:
    t = range(100)
    c = random.sample(t,5)
    for i in t:
        line = f.readline()
        if i in c:
            print "Probabilite donnee a :" + line[:-2]
            print Model_1[line.strip().split()]
            print Model_2[line.strip().split()]

print "Phrase generee avec un modele Unigramme:"
print Model_1.sample(20)

print "Phrase generee avec un modele Bigramme:"
print Model_2.sample(20)


# A partir d'une liste de mots, calculer leur ordonnancement le plus probable:
# Exemple : 'which' 'works' 'better' '?' 

import itertools
orderings = itertools.permutations(['which','works','better','?'])

best_prob = 0.0
best = []

for ordering in orderings:
    prob = math.exp(Model_2[ordering])
    if best_prob < prob:
        print prob
        print ordering
        best_prob = prob
        best = ordering

print "Meilleur ordonnancement:"
print best
