# -*- coding: utf-8 -*- 
"""
A script for training and evaluating a Naive Bayes classifier on the Wikipedia Weasels corpus.
Options:
	-h --help		Show this message.
	-e --eval		Use the evaluation corpus to evaluate. Otherwise a held out is built from the training corpus, using the HELDOUT_SIZE constant.
	-l --load		Load a pickled classifier given by its filename.
	-s --save		Saves the trained classifier to a pickle file using the given filename.
	-a --adjust		Trains multiple classifiers defined by the parameters defined in the constants file and evaluates them.
"""
from weasel_classify.NaiveBayesWeaselClassifier import NaiveBayesWeaselClassifier
from weasel_classify.ActiveNaiveBayesWeaselClassifier import ActiveNaiveBayesWeaselClassifier
from weasel_classify.SvmWeaselClassifier import SvmWeaselClassifier
from weasel_classify.EvaluationResult import EvaluationResult
from weasel_classify.Constants import *
import pickle
import sys
import getopt
import os
import math
import time
import operator

corpus_file = TRAIN_SENTS_FILE
print 'Loading sentences from: '+corpus_file
evalset = pickle.load(open(corpus_file,"rb"))
print 'Ok.'
uscount = 0
wcount = 0
wdict = dict()
for s in evalset:
	if(s.certainty=='uncertain'):
		for tk in s.matrix:
			if isinstance(tk, list):				
				weasel_phrase = (' '.join(tk)).lower()
				c = wdict.get(weasel_phrase,0) + 1
				wdict[weasel_phrase] = c 
				wcount += 1
		uscount += 1
sorted_wdict = sorted(wdict.iteritems(), key=operator.itemgetter(1))
print sorted_wdict
print 'Uncertain sentences: '+str(uscount)
print 'Weasel words: '+str(wcount)
print 'Distinct weasel words: '+str(wcount)
		