#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys
import codecs
import unicodedata
import string
import time
import datetime
import math
import pickle
from sklearn.preprocessing import scale
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import chi2, f_classif
from sklearn import metrics
from sklearn import cross_validation
import numpy as np
import scipy as sp

from weasel_classify.Weasels import *
from weasel_classify.Constants import *

#def removeStringNoise(data):
  #return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters).lower()
  
def main():
	print 'Begin...'
	with open('train_sentences.pickle') as f:
	train_sents = pickle.load(f)
	#train_sents = []
	print 'Corpus examples: ', len(train_sents)
	labels = [ex.certainty for ex in train_sents]
	le = LabelEncoder()
	le.fit(labels)
	transformed_labels = le.transform(labels)

	words = [ex.string for ex in train_sents]
	lemmas = [' '.join([genia[1] for genia in s.genia_words]) for s in train_sents]
	postags = [' '.join([genia[2] for genia in s.genia_words]) for s in train_sents]

	neses = []
	for s in train_sents:
	names_count = 0
	for genia_info in s.genia_words:
	    if genia_info[4][0] == 'B':
		    names_count += 1
	neses.append([names_count,0])

	wexes = [[re.search(w,s.string.lower()) != None for w in WEASELS] for s in train_sents]

	neses = np.array(neses)
	wexes = np.array(wexes)

	ch_words_vectorizer = CountVectorizer(input=u'content',analyzer=u'char',ngram_range=(1,8),binary=False)#,max_features=40000)
	words_vectorizer = CountVectorizer(input=u'content',analyzer=u'word',ngram_range=(1,7),binary=False,max_features=40000)
	lemmas_vectorizer = CountVectorizer(input=u'content',analyzer=u'word',ngram_range=(1,7),binary=True,max_features=40000)
	postags_vectorizer = CountVectorizer(input=u'content',analyzer=u'word',ngram_range=(1,7),binary=True,max_features=40000)

	#ngram features
	ch_words_dtm = ch_words_vectorizer.fit_transform(words)
	words_dtm = words_vectorizer.fit_transform(words)
	lemmas_dtm = lemmas_vectorizer.fit_transform(lemmas)
	postags_dtm = postags_vectorizer.fit_transform(postags)

	#print words_dtm.shape
	#print lemmas_dtm.shape
	#print postags_dtm.shape
	#print neses.shape
	#print wexes.shape

	#join all feature matrices  
	fsets = sp.sparse.hstack((ch_words_dtm, words_dtm, lemmas_dtm, postags_dtm, wexes, neses))
	#fsets = sp.sparse.hstack((words_dtm, lemmas_dtm, postags_dtm, wexes))
	print 'Feature sets matrix shape: ', fsets.shape

	print 'Feature selection in progress...'

	#fsets = fsets.toarray()

	#feature_selector = VarianceThreshold(threshold=(.999 * (1 - .999)))
	#fsets = feature_selector.fit_transform(fsets)

	#feature_selector = SelectKBest(chi2, k=(fsets.shape[1]/2))
	feature_selector = SelectKBest(chi2, k=50000)
	fsets = feature_selector.fit_transform(fsets, transformed_labels)

	print 'Feature sets matrix shape: ', fsets.shape

	#clf = LinearSVC(class_weight='auto',C=0.7)
	#clf = LinearSVC(C=0.5)
	#clf = NuSVC(nu=0.19)
	clf = MultinomialNB(fit_prior=False, alpha=1.0)
	#clf = MultinomialNB(fit_prior=False, alpha=0.7)
	#clf = MultinomialNB(fit_prior=False, alpha=0.5)


	#Ejemplo de uso
	#clf.fit(joined_dtm, transformed_labels)

	print 'Cross validation in progress...'
	cv_results = cross_validation.cross_val_score(clf, fsets, transformed_labels, cv=10, scoring='f1')
	#print cv_results
	print "Performance average: ", np.average(cv_results)  
	print 'Bye!'
  
if __name__ == "__main__":
	main()
 
 
