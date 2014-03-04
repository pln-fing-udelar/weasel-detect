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
from weasel_classify.Sentence import Sentence
import pickle
import sys
import getopt
import os
import math
import time
import operator

def main():	
	use_eval = False
	use_AL = False
	use_NaiveBayes = True
	load_classifier = False
	save_classifier = False	
	adjust = False
	wcoverage = False	
	incremental = False
	no_titles = False
	# parse command line options
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hel:sapwt:ix", ["help","eval","load","save","active","param-adjust","weasel-coverage","type","exclude-titles"])
	except getopt.error, msg:
		print msg
		print "for help use --help"
		sys.exit(2)
	# process options
	for o, a in opts:
		if o in ('x','--exclude-titles'):
			no_titles = True
			print 'Excluding titles from corpus.'
		if o in ('-i','--incremental'):
			incremental = True
			print 'Using incremental mode.'
		if o in ('-w','--weasel-coverage'):
			print 'Using Weasel match counter.'
			wcoverage = True
		if o in ('-a','--active'):
			print 'Using Naive Bayes Active Learner.'
			use_AL = True
		if o in ('-p','--param-adjust'):
			adjust = True
		if o in ('-e','--eval'):
			print 'Using Evaluation corpus.'
			use_eval = True
		if o in ('-l','--load'):
			load_classifier = True
			savedClassifierFile = a
			print ""+savedClassifierFile
		if o in ('-s','--save'):
			save_classifier = True
		if o in ('-t','--type'):		
			if a == 'svm':
				print 'Using SVM classifier.'
				use_NaiveBayes = False
			else:
				print 'Using Naive Bayes classifier.'
				use_NaiveBayes = True				
		if o in ('-h','--help'):
			print __doc__
			sys.exit(0)	
	if load_classifier:		
		print 'Loading classifier from: '+savedClassifierFile
		classifier = pickle.load(open(savedClassifierFile,"rb"))	
		print 'Ok.'
		if use_eval:
			print 'Loading evaluation sentences from: '+EVAL_SENTS_FILE
			evalset = pickle.load(open(EVAL_SENTS_FILE,"rb"))
			print 'Ok.'
		else:
			print 'Using held out corpus from the training corpus. Held out size: '+str(HELDOUT_SIZE)
			print 'Loading training sentences from: '+TRAIN_SENTS_FILE
			trainset = pickle.load(open(TRAIN_SENTS_FILE,"rb"))
			print 'Ok.'
			if no_titles:
				print 'Excluding article titles from the training sentences...'
				temptset = []
				for sent in trainset:
					if sent.id[-2:] != u'.1':
						temptset.append(sent)
				trainset = temptset	
				print 'Ok.'
			limit = len(trainset) - int(math.floor(len(trainset)*HELDOUT_SIZE))
			evalset = trainset[limit:]			
		print 'Evaluating classifier...'
		result = classifier.evaluate(evalset)
		result.printResults()
	#train a new classifier
	else:
		print 'Loading training sentences from: '+TRAIN_SENTS_FILE
		trainset = pickle.load(open(TRAIN_SENTS_FILE,"rb"))	
		if no_titles:
			print 'Excluding article titles from the training sentences...'
			temptset = []
			for sent in trainset:
				if sent.id[-2:] != u'.1':
					temptset.append(sent)
			trainset = temptset	
			print 'Ok ('+str(len(trainset))+' sentences).'
		# BEGIN: Choose evaluation corpus: ST EVAL or HELDOUT
		if use_eval:
			print 'Loading evaluation sentences from: '+EVAL_SENTS_FILE
			evalset = pickle.load(open(EVAL_SENTS_FILE,"rb"))
		#use the last HELDOUT_SIZE sentences from the training corpus to evaluate
		else:
			print 'Using held out corpus from the training corpus. Held out size: '+str(HELDOUT_SIZE)
			limit = len(trainset) - int(math.floor(len(trainset)*HELDOUT_SIZE))
			evalset = trainset[limit:]
			trainset = trainset[:limit]		
			#trainset = trainset[0:4100] #para comparar con AL
		# END: Choose evaluation corpus
		
		# BEGIN: Execution mode: Parameter adjustment, incremental training, or simple execution.
		# Execution mode 1: Parameter adjustment, trains a classifier for each combination of parameters
		if adjust:
			print 'Adjusting parameters (this may take a while!)...'
			test_count = 1
			total_tests = reduce(operator.mul,[len(e) for e in [USEBOL,USENES,USEWEXES,MOSTCOMMON,STOPWORDS,BOOLCOUNT]])
			for useBOL in USEBOL:
				for useNES in USENES:
					for useWEXES in USEWEXES:
						for bc in BOOLCOUNT:
							for mcs in MOSTCOMMON:
								for sws in STOPWORDS:								
									#print '*'*75
									#print 'Test '+str(test_count)+' of '+str(total_tests)
									#print 'Creating classifier...'
									if use_NaiveBayes:										
										classifier = NaiveBayesWeaselClassifier()
									else:										
										classifier = SvmWeaselClassifier()
									#print 'Ok.'
									#print 'Setting parameters...'
									#print "Bag of lemmas: "+str(useBOL)
									#print "Named entities: "+str(useNES)
									#print "Wexes: "+str(useWEXES)
									#print "Most common size: "+str(mcs)
									#print "Stop words: "+str(sws)
									#print "Bool count: "+str(bc)									
									classifier._useBOL = useBOL
									classifier._mostCommonSize = mcs
									classifier._stopWordsSize = sws
									classifier._boolCount = bc
									classifier._useNES = useNES
									classifier._useWEXES = useWEXES
									#print 'Ok.'
									#print 'Training classifier...'
									classifier.train(trainset)
									#print 'Evaluating classifier...'
									result = classifier.evaluate(evalset)
									#result.printResults()
									print str(test_count)+';'+str(not useBOL)+';'+str(useBOL)+';'+str(useNES)+';'+str(useWEXES)+';'+str(mcs)+';'+str(sws)+';'+str(bc)+';'+str(mcs-sws)+';'+str(result.precision)+';'+str(result.recall)+';'+str(result.fmeasure)
									test_count += 1
			print '*'*75
			
		# Execution mode 2: Train and evaluate several classifiers, increasing the size of the training set in each step
		elif incremental:
			print 'Calculating results for different training set sizes...'
			if use_NaiveBayes:						
				classifier = NaiveBayesWeaselClassifier()
			else:				
				classifier = SvmWeaselClassifier()
			for size in range(3000,len(trainset)+50,50):
				classifier._build_bow_features(trainset)
				classifier.train(trainset[:size])				
				result = classifier.evaluate(evalset)			
				print str(len(trainset[:size]))+';'+str(result.precision)+';'+str(result.recall)+';'+str(result.fmeasure)	
		# Execution mode 3: Train and evaluate just one classifier
		else:		
			if use_NaiveBayes:		
				if use_AL:
					classifier = ActiveNaiveBayesWeaselClassifier()
				else:
					classifier = NaiveBayesWeaselClassifier()
			else:				
				classifier = SvmWeaselClassifier()
			print 'Ok.'
			print 'Training classifier...'
			classifier.train(trainset)
			if(save_classifier):
				classifierFile = "classifier"+str(int(math.floor(time.time())))+".pickle"
				print "	Saving classifier to '"+classifierFile+"'"
				pickle.dump(classifier,open(classifierFile,"wb"))	
				print '	Ok.'
			print 'Ok.'
			print 'Evaluating classifier...'
			result = classifier.evaluate(evalset)
			result.printResults()
		# END: Execution mode
		
		# BEGIN: Print weasel count
		if wcoverage:
			print '-'*28+'WEASEL LIST COVERAGE'+'-'*27			
			for (w,c) in sorted(classifier._WEXESCount.items(), key=lambda x: x[1]):
				print w + ': '+str(c)
			print '-'*75
		# END: Print weasel count			
	print 'Bye!'


if __name__ == "__main__":
    main()


