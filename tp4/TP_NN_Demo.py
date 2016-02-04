from __future__ import division

import collections
import operator
import numpy as np
import random
from nltk.corpus import movie_reviews

Ids = movie_reviews.fileids()

def count_words(words):
    words = [ w.lower() for w in words ]
    return collections.Counter(words)

def combine_counts(counts):
    final_counts = collections.defaultdict(int)
    for count in counts:
        for key, value in count.items():
            final_counts[key] += value
    return final_counts

def get_n_top_words(count, n):
    n_top_counts =  sorted(count.items(), key=operator.itemgetter(1), reverse=True)[0:n]
    return [ w for w, f in n_top_counts ]

def get_top_values(count, top_keys):
    return [ count.get(key, 0) for key in top_keys ]

def normalize_counts(count):
    s = sum(count.values())
    return dict([ (k, v/s) for k, v in count.items() ])

def get_counts_matrix(Ids, n):
    counts = []
    for fid in Ids:
        Words = movie_reviews.words(fileids=fid)
        words = [w.lower() for w in Words]
        counts.append(collections.Counter(words))
    total_counts = combine_counts(counts)
    top_words = get_n_top_words(total_counts, n)
    top_values = [ get_top_values(normalize_counts(c), top_words) for c in counts ]
    return np.array(top_values)

n = 128
M = get_counts_matrix(Ids, n)

#Preparer les donnees
from matplotlib.mlab import PCA
MR_PCA = PCA(M)

X = M
Y = np.concatenate((np.zeros(1000),np.ones(1000)))

threshold_sets = 1600
index_shuffle = range(len(Y))
random.shuffle(index_shuffle)

X_train = np.array([X[i] for i in index_shuffle[:threshold_sets]])
Y_train = np.array([Y[i] for i in index_shuffle[:threshold_sets]])

X_test = np.array([X[i] for i in index_shuffle[threshold_sets:]])
Y_test = np.array([Y[i] for i in index_shuffle[threshold_sets:]])

#Perceptron
from Models import perceptron
model = perceptron(500)
print 'Perceptron:'
model.train(X_train, Y_train)
print 'Precision: %f' % model.test(X_test, Y_test)

#Regression Logistique
from Models import logistic_regression
print 'Regression logistique:'
model = logistic_regression(1)
for epoch in range(5):
    model.train(X_train, Y_train)
    print 'Iteration : %i, Precision: %f' % (epoch+1, model.test(X_test, Y_test))

#Reseau de neurones
import keras
print 'Reseau de neurones:'

print 'Model creation...'
from keras.models import Sequential
model = Sequential()

from keras.layers.core import Dense, Dropout
model.add(Dense(64, input_dim=n, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print 'Compiling...'
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')

print 'Training...'
model.fit(X_train, Y_train, nb_epoch=10, batch_size=1, show_accuracy=True)
print 'Precision : %f' % (1 - (sum( (model.predict_classes(X_test, batch_size=1,verbose = 0).T)[0] != Y_test)/len(Y_test)))


#Reseau de neurones recurrent: avec Word Embeddings

print 'Preprocessing...'
def _filter_count(count, threshold):
    return ((k, v) for k, v in count if v >= threshold)

def get_vocabulary(n = 10):
    Words = movie_reviews.words()
    words = [w.lower() for w in Words]
    count = collections.Counter(words)
    count_pairs = sorted(count.items(), key=lambda x: -x[1])
    words, counts = list(zip(*_filter_count(count_pairs, n)))
    word_to_id = dict(zip(words, range(1, len(words)+1)))
    return word_to_id

def get_index_sequences(Ids, voc):
    sentences = []
    for fid in Ids:
        Sentences = movie_reviews.sents(fileids=fid)
        sentences.extend([np.array([voc.get(w.lower(), 0) for w in s]) for s in Sentences])
    return sentences

class data_generator(object):
    def __init__(self, X, Y, batch_size):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.tot = len(X)
        self.ids = range(self.tot)
        self.cpt = 0
        self.batch_size = batch_size

    def next(self):
        if (self.cpt+self.batch_size > self.tot):
            self.cpt=0
        random.shuffle(self.ids)
        x = list()
        y = list()
        for i in range(self.batch_size):
            self.cpt+=1
            x.append(X[self.ids[self.cpt-1]])
            y.append(Y[self.ids[self.cpt-1]])
        return np.array(x), np.array(y)

Neg_Ids = Ids[:800]
Pos_Ids = Ids[1200:]
voc = get_vocabulary()
voc_size = len(voc)+1
X_pos = get_index_sequences(Pos_Ids, voc)
X_neg = get_index_sequences(Neg_Ids, voc)
Y_pos = [1] * len(X_pos)
Y_neg = [0] * len(X_neg)

X = X_pos + X_neg
Y = Y_pos + Y_neg

from keras.preprocessing import sequence
maxlen = 50
batch_size = 2048
X = sequence.pad_sequences(X, maxlen=maxlen)
datagen = data_generator(X, Y, batch_size)

print 'Reccurent neural network with Word Embeddings: Model creation...'
from keras.models import Sequential                                                                                                                                                          
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(Embedding(voc_size, output_dim=128, input_length=maxlen))
model.add(LSTM(input_dim=128, output_dim=64))
model.add(Dropout(0.5))
model.add(Dense(input_dim=64, output_dim=1, activation='sigmoid'))

print 'Compilation...'
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')

print 'Training...'
samples = 10000
model.fit_generator(generator = datagen,
                    samples_per_epoch = samples,
                    nb_epoch = 10,
                    show_accuracy = True)

print 'Testing...'
proba = []
for i in range(800,1200):
    X_test = get_index_sequences([Ids[i]], voc)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    proba.append(np.mean(model.predict_proba(X_test, batch_size = 1, verbose = 0)))

Y_eval = proba > 0.5*np.ones(400)
Y_test = np.array( [0]*200 + [1]*200)

print (1 -(sum(Y_eval != Y_test)/400))

