from nltk.classify.svm import SvmClassifier
from collections import Counter, defaultdict
from WeaselCorpusReader import WeaselCorpusReader
from WeaselClassifier import WeaselClassifier
from libsvm.python.svmutil import *
from EvaluationResult import *
from Weasels import *
from ProyALDebug import *
import math
import re


class SvmWeaselClassifier(WeaselClassifier):
	"""
	This class implements a WeaselClassifier that uses SvmClassifier from nltk, with a Bag of Words approach.
	"""

	def getTrainingSet(self,size=0.9):
		tset = self._corpus.sents()
		limit = int(math.ceil(len(tset)*size))
		return [(sent.string, sent.certainty) for sent in tset[0:limit]] 
				
	def getEvaluationSet(self,size=0.1):
		eset = self._corpus.sents()
		ini = int(math.floor(len(eset)*size))
		return [(sent.string, sent.certainty) for sent in eset[-ini:]] 

	def train(self,training_set=None,paramstr='-t 0 -c 5 -b 1 -q'):
		"""
		Trains the BOW SVM classifier.
		"""
		if (training_set == None):
			debug("Using corpus training file.")	
			training_set = [(sent, sent.certainty) for sent in self._corpus.sents()]
		#training_set = training_set[0:100] #para probar con conjuntos mas chicos
		#build features	
		self._build_bow_features(training_set)		
		#build featuresets for each sentence
		labeled_featuresets = []
		for sent in training_set:
			featureset = self.sentenceFeatures(sent)
			labeled_featuresets.append((featureset,sent.certainty))
		train_size = len(labeled_featuresets)
		debug('Size of training set: '+str(train_size))
		#train the SVM
		#self._classifier = SvmClassifier.train(labeled_featuresets)
		debug("Building training examples for LIBSVM...")
		labels = [int(label=='uncertain') for (_,label) in labeled_featuresets]
		examples = [[val for (_,val) in sorted(dfset.items())] for (dfset,_) in labeled_featuresets]
		problemData = [labels,examples]
		debug("Ok.")
		labels = problemData[0]
		examples = problemData[1]
		debug("Creating problem and parameters for libsvm...")
		problem = svm_problem(labels,examples)
		parameter = svm_parameter(paramstr)
		debug("Parameters: "+paramstr)
		debug("Ok.")
		debug("Training svm model...")
		self._model = svm_train(problem, parameter)
		debug("Ok.")

	def evaluate(self,evaluation_set=None):
		"""
		Evaluates the classifier on the given evaluation_set and returns an EvaluationResult object. If no evaluation set is provided, the
		evaluation corpus is used. Uses the underlying LIBSVM model.
		"""
		eval_labeled_featuresets = []
		if (evaluation_set == None):
			evaluation_set = [(sent.string, sent.certainty) for sent in self._evalcorpus.sents()]
		#evaluation_set = evaluation_set[0:500] #para probar con conjuntos mas chicos
		for sent in evaluation_set:
			featureset = self.sentenceFeatures(sent)
			eval_labeled_featuresets.append((featureset,sent.certainty))
		debug('Size of evaluation set: '+str(len(eval_labeled_featuresets)))
		debug("Building evaluation examples for LIBSVM...")
		labels = [int(label=='uncertain') for (_,label) in eval_labeled_featuresets]
		examples = [[val for (_,val) in sorted(dfset.items())] for (dfset,_) in eval_labeled_featuresets]
		evalData = [labels,examples]
		debug("Ok.")
		debug("Predicting labels for the evaluation examples...")
		predicted_labels, _, _ = svm_predict(labels,examples,self._model)
		debug("Ok.")
		debug("Evaluating model (libsvm)...")
		ACC, MSE, SCC = evaluations(labels,predicted_labels)
		debug("Ok.")
		tp = sum([p==1 and p==l for (p,l) in zip(predicted_labels,labels)])
		fp = sum([p==1 and p!=l for (p,l) in zip(predicted_labels,labels)])
		tn = sum([p==0 and p==l for (p,l) in zip(predicted_labels,labels)])
		fn = sum([p==0 and p!=l for (p,l) in zip(predicted_labels,labels)])
		debug("Ok.")
		return EvaluationResult(tp,fp,tn,fn)

		

		
