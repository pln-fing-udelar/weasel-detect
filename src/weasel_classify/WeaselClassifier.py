# -*- coding: utf-8 -*- 
from WeaselCorpusReader import WeaselCorpusReader
from EvaluationResult import EvaluationResult
from collections import Counter, defaultdict
from Weasels import *
from Constants import *
from ProyALDebug import debug
import nltk.tokenize
import math
import datetime
import pprint
import re


class WeaselClassifier:
	"""
	This is the abstract class for all the weasel classifiers.
	"""

	def __init__(self, 
					corpus_dir="../corpus/", 
					train_file="task1_train_wikipedia_rev2.xml", 
					eval_file="task1_weasel_eval.xml"					
					):		
		self._corpus = WeaselCorpusReader(corpus_dir,train_file)
		self._evalcorpus = WeaselCorpusReader(corpus_dir,eval_file)		
		self._tokenizer = nltk.tokenize.TreebankWordTokenizer()
		#training parameters
		self._mostCommonSize = 500
		self._stopWordsSize = 0
		self._useNES = True
		self._useBOL = True
		self._boolCount = True
		self._useWEXES = True
		self._WEXESCount = {w:0 for w in WEASELS} #weasel coverage
		
	def train(self,training_set):
		"""
		Not implemented at this level.
		"""
		raise NotImplementedError()

	def evaluate(self,evaluation_set=None):
		"""
		Evaluates the classifier on the given evaluation_set and returns an EvaluationResult object. If no evaluation set is provided, the
		evaluation corpus is used.
		"""
		#if (evaluation_set == None):			
		#	evaluation_set = [(sent, sent.certainty) for sent in self._evalcorpus.sents()]
		#evaluation_set = evaluation_set[0:50]
		eval_featuresets = [self.sentenceFeatures(sent) for sent in evaluation_set]
		results = self._classifier.batch_classify(eval_featuresets)
		labels = [s.certainty for s in evaluation_set]
		tp,fp,tn,fn = 0,0,0,0
		false_pos_sents = []
		false_neg_sents = []
		for (s,r,l) in zip(evaluation_set,results,labels):
			if r=='uncertain' and r==l:
				tp += 1
			elif r=='uncertain' and r!=l:
				fp += 1
				false_pos_sents.append(s)
				debug('FP:'+s.id.encode('UTF-8')+': '+s.string.encode('UTF-8'))
			elif r=='certain' and r==l:
				tn += 1
			elif r=='certain' and r!=l:
				fn += 1
				false_neg_sents.append(s)
				#debug('FN:'+s.id.encode('UTF-8')+': '+s.string.encode('UTF-8'))
			else:
				raise Exception("Evaluation error!")		
		debug('Size of evaluation set: '+ str(len(evaluation_set))	)
		return EvaluationResult(tp,fp,tn,fn)
	
	def sentenceFeatures(self,sent):	
		"""
		Returns the feature set of a sentence for this classifier.			
		"""
		featureset = dict(self._empty_int_set) #copy from the empty featureset to enhance performance		
		#BAG OF WORDS AND NAMED ENTITIES
		inside_name = False
		named_entities = 0
		for (word,lemma,pos,chunk,ne) in sent.genia_words:	
			# if self._useNES is False, names will be counted as any other words
			if self._useNES:
				if ne[0]=='B':						
					named_entities += 1
					inside_name = True
				if ne=='O':
					inside_name = False
			if self._useBOL:
				token = lemma.lower()
			else:
				token = word.lower()
			if not inside_name and token in self._empty_feature_set:
				if self._boolCount:
					featureset[(self._empty_feature_set[token])] = 1
				else:
					newval = int(featureset[(self._empty_feature_set[token])]) + 1
					featureset[(self._empty_feature_set[token])] = newval
					
		#LIST OF KNOWN WEASELS
		if self._useWEXES:		
			for w in WEASELS:
#				if w not in self._empty_feature_set:
				if re.search(w,sent.string.lower()) != None:									
					featureset[len(featureset)] = 1
					self._WEXESCount[w] += 1		
				else:					
					featureset[len(featureset)] = 0
		if self._boolCount:						
			featureset[len(featureset)] = 1
		else:
			featureset[len(featureset)] = named_entities
		#debug(sent.string)
		#debug("NEs: " + str(named_entities))
		#for v in featureset:
			#if featureset[v] > 0:
				#debug("V:"+str(v)+"=>"+str(featureset[v]))
		return featureset
	
	def classifySentence(self,sent):
		"""
		Classifies a Sentence object and returns the predicted label.
		"""
		return self._classifier.classify(self.sentenceFeatures(sent))
			
	@staticmethod
	def load(savefile):		
		raise NotImplementedError() 
		
	def save(self,savefile):
		raise NotImplementedError()
		
	def _build_bow_features(self,training_set):
		"""
		Builds the feature set that will be considered by the classifier. This is a Bag of Words approach. 
		The features are taken from the 'size' most common words after the first 'stop_words_size' most common, which are left out.
		Note that this set is dependant on the given training_set
		"""		
		#build features
		tks = []
		if self._useBOL:
			debug('Using word lemmas as features for the Bag of Words.')
			for sent in training_set:
				for (word,lemma,pos,chunk,ne) in sent.genia_words:
					tks.append(lemma.lower())		
		else:
			for sent in training_set:
				for (word,lemma,pos,chunk,ne) in sent.genia_words:
					tks.append(word.lower())		
		counted_words = Counter(tks)	
		most_common_words = [word[0] for word in counted_words.most_common(self._mostCommonSize)]
		too_common_words = [word[0] for word in counted_words.most_common(self._stopWordsSize)]
		debug("Words excluded from dictionary:")
		debug(too_common_words)
		self._feature_words = list(set(most_common_words) - set(too_common_words))		
		self._empty_feature_set = dict()
		self._empty_int_set = dict()
		for i in range(0,len(self._feature_words)):
			#debug(self._feature_words[i] + '=>' +str(i))
			self._empty_feature_set[self._feature_words[i]] = i
			self._empty_int_set[(i)] = 0		
		debug('Size of dictionary: '+ str(len(self._empty_feature_set)))
	
