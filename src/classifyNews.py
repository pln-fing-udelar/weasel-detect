# -*- coding: utf-8 -*- 
"""
A script for training and evaluating a Naive Bayes classifier on the given corpus.
"""
from weasel_classify.NaiveBayesWeaselClassifier import NaiveBayesWeaselClassifier
from weasel_classify.EvaluationResult import EvaluationResult
from weasel_classify.Constants import *
import pickle
import sys
import getopt
import os
import math
import time

def main():	
	use_eval = False
	load_classifier = False
	save_classifier = False	
	# parse command line options
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hc:s:", ["help","classifier","sents"])
	except getopt.error, msg:
		print msg
		print "for help use --help"
		sys.exit(2)
	# process options
	for o, a in opts:
		if o in ('-c','--classifier'):
			load_classifier = True
			savedClassifierFile = a
		if o in ('-s','--sents'):
			load_sents = True
			sentsFile = a
		if o in ('-h','--help'):
			print __doc__
			sys.exit(0)	
	if load_classifier:		
		print 'Loading classifier from: '+savedClassifierFile
		classifier = pickle.load(open(savedClassifierFile,"rb"))	
		print 'Loading sentences from: '+sentsFile
		sents = pickle.load(open(sentsFile,"rb"))
		print 'Ok.'		
	else:		
		print __doc__
		sys.exit(0)	
	print 'Classifying sentences...'
	print '-'*75
	weaselcount = 0
	for sent in sents:
		if classifier.classifySentence(sent) == u'uncertain':			
			print "SENTENCE ID: "+sent.id.encode("UTF-8")
			print sent.string.encode("UTF-8")+os.linesep
			weaselcount += 1
	print '-'*75
	print "Sentences classified: "+str(len(sents))
	print "Uncertain sentences found: "+str(weaselcount)
	print 'Bye!'


if __name__ == "__main__":
    main()


