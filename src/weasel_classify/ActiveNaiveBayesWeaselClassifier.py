# -*- coding: utf-8 -*- 
from nltk.classify.naivebayes import NaiveBayesClassifier
from WeaselCorpusReader import WeaselCorpusReader
from ActiveWeaselClassifier import ActiveWeaselClassifier
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.probability import FreqDist
from nltk.probability import ELEProbDist
import math
import unicodedata
import random
import cPickle as pickle
from pprint import pprint

class ActiveNaiveBayesWeaselClassifier(ActiveWeaselClassifier):
	"""
	An implementation of the interface for weasel classifiers which uses active learning to minimize the size of the training examples set.
	The underlaying model is the nltk.classify.naivebayes.NaiveBayesClassifier class.
	"""
	
	
	def buildInitClassifier(self,examples):
		"""
		Trains the underlying classifier with the given examples. 
		@param examples: An object of type [(Dict,String)], where the Dict is a feature set representation of a sentence, and the String is its label.
		"""
		print 'Initial examples: '+str(len(examples))
		self._load_counters(examples)
		self._loadClassifier()

	
	def addExample(self,example):
		"""
		Add the given example to the underlying classifier variables, allowing its incremental training. Not implemented at this level.
		@param example: An object of type (Dict,String), where the Dict is a feature set representation of a sentence, and the String is its label.
		"""
		self._updateCounters(example)
		self._loadClassifier()
	
	
	def addExamples(self,examples):
		"""
		Add the given example to the underlying classifier variables, allowing its incremental training. Not implemented at this level.
		@param example: An object of type (Dict,String), where the Dict is a feature set representation of a sentence, and the String is its label.
		"""
		for example in examples:
			self._updateCounters(example)
		self._loadClassifier()
	
	
	def _load_counters(self,labeled_featuresets):
		"""
		This method is exactly the train method of the NaiveBayesClassifier, except that it
		does not create a classifier, and keeps the counter variables, so as to update them later
		"""
		self._label_freqdist = FreqDist() 
		self._feature_freqdist = defaultdict(FreqDist) 
		self._feature_values = defaultdict(set) 
		self._fnames = set() 
   
		# Count up how many times each feature value occured, given 
		# the label and featurename. 
		for featureset, label in labeled_featuresets: 
			self._label_freqdist.inc(label) 
			for fname, fval in featureset.items(): 
				# Increment freq(fval|label, fname) 
				self._feature_freqdist[label, fname].inc(fval) 
				# Record that fname can take the value fval. 
				self._feature_values[fname].add(fval) 
				# Keep a list of all feature names. 
				self._fnames.add(fname) 

		# If a feature didn't have a value given for an instance, then 
		# we assume that it gets the implicit value 'None.'  This loop 
		# counts up the number of 'missing' feature values for each 
		# (label,fname) pair, and increments the count of the fval 
		# 'None' by that amount. 
		for label in self._label_freqdist: 
			num_samples = self._label_freqdist[label] 
			for fname in self._fnames: 
				count = self._feature_freqdist[label, fname].N() 
				self._feature_freqdist[label, fname].inc(None, num_samples-count) 
				self._feature_values[fname].add(None) 
		
	def _updateCounters(self,labeled_featureset):
		fset = labeled_featureset[0]
		label = labeled_featureset[1]
		self._label_freqdist.inc(label) 
		for fname, fval in fset.items(): 
			# Increment freq(fval|label, fname) 
			self._feature_freqdist[label, fname].inc(fval) 
			# Record that fname can take the value fval. 
			self._feature_values[fname].add(fval) 
			# Keep a list of all feature names. 
			self._fnames.add(fname) 

	def _loadClassifier(self):
		# Choose estimator
		estimator = ELEProbDist
		# Create the P(label) distribution 
		label_probdist = estimator(self._label_freqdist)	
		# Create the P(fval|label, fname) distribution 
		feature_probdist = {} 
		for ((label, fname), freqdist) in self._feature_freqdist.items(): 
			probdist = estimator(freqdist, bins=len(self._feature_values[fname])) 
			feature_probdist[label,fname] = probdist 		
		self._classifier = NaiveBayesClassifier(label_probdist, feature_probdist)

		
