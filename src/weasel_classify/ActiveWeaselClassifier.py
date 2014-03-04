# -*- coding: utf-8 -*- 
from WeaselClassifier import WeaselClassifier
from nltk.classify.naivebayes import NaiveBayesClassifier
from bisect import bisect
from math import *
import random
import pprint
import pickle
import sys


class ActiveWeaselClassifier(WeaselClassifier):
	"""
	An implementation of the interface for weasel classifiers which uses active learning to minimize the size of the training examples set.
	The underlaying model is the nltk.classify.naivebayes.NaiveBayesClassifier class.
	"""
	
	def train(self,training_set=None,held_out_size=1111,initial_queries_size=0.30,threshold=0.6,batch_size=1,agnostic_size=0):
		"""
		Trains the classifier using the Active Learning paradigm (Pool scenario, uncertainty query sampling).
		To use this method the builInitClassifier and addExample method must be implemented.
		"""		
		if (training_set == None):			
			training_set = [(sent, sent.certainty) for sent in self._corpus.sents()]
		#training_set = training_set[0:10] #para comparar con los resultados anteriores			
		#separate the held_out corpus
		held_out = training_set[-held_out_size:]
		#build features, leave out the held out examples!
		self._build_bow_features(training_set[0:-held_out_size])	
		#build featuresets for each sentence
		labeled_featuresets = []
		for sent in training_set[0:-held_out_size]:			
			featureset = self.sentenceFeatures(sent)
			labeled_featuresets.append((featureset,sent.certainty))
		#take random samples
		#random.shuffle(labeled_featuresets)	
		#calculate the number of initial samples
		limit = int(floor(len(labeled_featuresets)*initial_queries_size))
		#firstSent =  training_set[0]
		#lastSent =  training_set[limit-1]
		#print 'First train sent: '+firstSent.id.encode('UTF-8')+': '+firstSent.string.encode('UTF-8')
		#print 'Last train sent: '+lastSent.id.encode('UTF-8')+': '+lastSent.string.encode('UTF-8')		
		#pickle.dump(labeled_featuresets[0:limit],open("oracionesAL.pickle","wb"))
		
		#build initial classifier
		self.buildInitClassifier(labeled_featuresets[0:limit])		
		# count the number of "oracle queries"
		queries = limit	
		# delete the used samples from the pool
		del(labeled_featuresets[0:limit])	
		print 'Pool size: '+str(len(labeled_featuresets))
		# query by batches of batch_size
		#initialize the min_set with the first batch_size examples
		min_set = [self._classifier.prob_classify(labeled_featuresets[i][0]) for i in range(min(batch_size,len(labeled_featuresets)))]
		min_set = [p.prob(p.max()) for p in min_set]
		min_set = zip(range(min(batch_size,len(labeled_featuresets))),min_set)
		# sort min_set by its second member (the prob)		
		min_set.sort(key = lambda x : x[1])
		# find the sample which maximizes uncertainty and begin incremental training		
		done = False
		save_counter = 0
		eval_counter = 0
		current_max = min_set[-1][1]
		current_result = self.evaluate(held_out)
		current_prec = current_result.precision
		current_recall = current_result.recall
		current_f = current_result.fmeasure
		print str(current_max)+', '+str(queries)+' queries, P;R;F: '+str(current_prec)+';'+str(current_recall)+';'+str(current_f)
		while (not done):			
			# still havent reached the desired performance in the held out corpus
			if(current_f < threshold):
				for i in range(min(batch_size,len(labeled_featuresets)),len(labeled_featuresets)):
					pdist = self._classifier.prob_classify(labeled_featuresets[i][0])
					maxlocalprob = pdist.prob(pdist.max())				
					if(maxlocalprob < current_max):# or maxlocalprob==0.5):
						# add new example to the min set
						min_set.append((i,maxlocalprob))
						# re-sort the min set
						min_set.sort(key = lambda x : x[1])
						# kickout the max
						min_set = min_set[0:batch_size]						
						#pp.pprint(min_set)
					current_max = min_set[-1][1]
					#already found a set with minimum certainty! 
					if current_max < 0.51:					
						break					
				#add "agnostic" examples to the min_set
				examples_indices = [j for (j,_) in min_set]
				agcount = 0
				while agcount < agnostic_size:
					rind = random.randint(0,len(labeled_featuresets)-1)
					if rind not in examples_indices:
						examples_indices.append(rind)
						agcount += 1
				#add examples given by the min_set (eventually extended with agnostic examples) to the training set				
				examples_indices.sort(reverse=True) #always MUST delete starting from highest index
				self.addExamples([labeled_featuresets[j] for j in examples_indices])
				# delete the used samples from the pool
				for j in examples_indices:
					del(labeled_featuresets[j])
				queries += len(examples_indices)
				current_result = self.evaluate(held_out)
				current_prec = current_result.precision
				current_recall = current_result.recall
				current_f = current_result.fmeasure
				print str(current_max)+', '+str(queries)+' queries, P;R;F: '+str(current_prec)+';'+str(current_recall)+';'+str(current_f)
				save_counter += 1				
				if(save_counter >= 500):
					print 'Saving classifier...'
					#savefilename = "active_classifier_"+str(queries)+".pkl"
					#self.save(savefilename,labeled_featuresets,held_out,queries)
					save_counter = 0
					print 'Ok.'
			print 'Queries updated: '+str(queries)
			if(current_f >= threshold or len(labeled_featuresets) <= 0):
				done = True
			#reinit the min_set
			min_set = [self._classifier.prob_classify(labeled_featuresets[i][0]) for i in range(min(batch_size,len(labeled_featuresets)))]
			min_set = [p.prob(p.max()) for p in min_set]
			min_set = zip(range(batch_size),min_set)
			min_set.sort(key = lambda x : x[1])			
		return queries-1 #turn back the last increment of the query counter

		
	def buildInitClassifier(self,examples):
		"""
		Trains the underlying classifier with the given examples. Not implemented at this level.
		@param examples: An object of type [(Dict,String)], where the Dict is a feature set representation of a sentence, and the String is its label.
		"""
		raise NotImplementedError()

	
	def addExample(self,example):
		"""
		Add the given example to the underlying classifier variables, allowing its incremental training. Not implemented at this level.
		@param example: An object of type (Dict,String), where the Dict is a feature set representation of a sentence, and the String is its label.
		"""
		raise NotImplementedError()
	
	def addExamples(self,examples):
		"""
		Add the given examples to the underlying classifier variables, allowing its incremental training. Not implemented at this level.
		@param example: An object of type (Dict,String), where the Dict is a feature set representation of a sentence, and the String is its label.
		"""
		raise NotImplementedError()
