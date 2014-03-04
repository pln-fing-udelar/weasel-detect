# -*- coding: utf-8 -*- 
from nltk.classify.naivebayes import NaiveBayesClassifier
from WeaselCorpusReader import WeaselCorpusReader
from WeaselClassifier import WeaselClassifier
from Constants import *
from ProyALDebug import debug
import pprint
import pickle

class NaiveBayesWeaselClassifier(WeaselClassifier):
	"""
	This class implements a WeaselClassifier that uses NaiveBayesClassifier from nltk, with a Bag of Words approach.
	"""

	def train(self,training_set=None):
		"""
		Trains the BOW NaiveBayes classifier.
		"""
		if (training_set == None):			
			training_set = [(sent, sent.certainty) for sent in self._corpus.sents()]
		#training_set = training_set[0:10] #para comparar con los resultados anteriores
		#build features		
		self._build_bow_features(training_set)
		
		#build featuresets for each sentence
		labeled_featuresets = []
		for sent in training_set:
			featureset = self.sentenceFeatures(sent)
			labeled_featuresets.append((featureset,sent.certainty))

		debug('Size of training set: '+str(len(labeled_featuresets)))
		#pp = pprint.PrettyPrinter(indent=4)
		#pp.pprint(labeled_featuresets)
		#train the NaiveBayes
		self._classifier = NaiveBayesClassifier.train(labeled_featuresets)
		
