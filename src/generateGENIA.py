# -*- coding: utf-8 -*- 
"""
Uso: 
generateCorpus [-t|-e|-r]
Genera los archivos ya procesados (.pickle) con las oraciones del corpus de entrenamiento, de evaluación o de feeds RSS, según el parámetro.
"""
from weasel_classify.Constants import *
from weasel_classify.WeaselCorpusReader import WeaselCorpusReader
from weasel_classify.Sentence import Sentence
import pickle
import sys
import getopt
import os
import subprocess
import feedparser


def main():
	# parse command line options
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hter", ["help","train","eval","rss"])
	except getopt.error, msg:
		print msg
		print "for help use --help"
		sys.exit(2)
	# process options
	for o, a in opts:
		if o in ('-t','--train'):
			print 'Loading sentences from '+TRAINING_FILE+'...'
			corpus = WeaselCorpusReader(CORPUS_DIR,TRAINING_FILE)
			sentences = corpus.sents()
			outputFileName = 'train_sentences.pickle'
			sentsFileName = 'train.sents'
		elif o in ('-e','--eval'):
			print 'Loading sentences from '+EVALUATION_FILE+'...'
			corpus = WeaselCorpusReader(CORPUS_DIR,EVALUATION_FILE)
			sentences = corpus.sents()
			outputFileName = 'eval_sentences.pickle'
			sentsFileName = 'eval.sents'
		elif o in ('-r','--rss'):			
			sentences = []			
			sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
			print 'Loading sentences from...'
			for url in RSSFEEDURLS:
				print url
				feed = feedparser.parse(url)
				count = 0
				for item in feed["items"]:
					for sent in sent_detector.tokenize(item["description"]):						
						ident = feed["feed"]["title"]+"."+str(count) #this is the sentence ID
						sentences.append(Sentence(None,sent,ident))
						count += 1
			outputFileName = 'rss_sentences.pickle'
			sentsFileName = 'rss.sents'
		else:
			print __doc__
			sys.exit(0)			
		print 'Processing sentences with GENIA...'
		present_dir=os.getcwd()
		print 'Changing current dir: '+GENIAHOME
		os.chdir(GENIAHOME)
		sentsFile = open(sentsFileName,'w')
		for sentence in sentences:
			sentsFile.write(sentence.string.encode('UTF-8')+os.linesep)
		sentsFile.close()
		p=subprocess.Popen(['./geniatagger',sentsFileName], stdout=subprocess.PIPE)
		genia_raw_sents = p.communicate()[0]
		print 'Changing current dir: '+GENIAHOME
		os.chdir(present_dir)
		print 'Reading GENIA results...'
		genia_sents = genia_raw_sents.split(os.linesep*2)
		outputFile = open(outputFileName,'w')
		i = 0
		for sentence in sentences:				
			sentence.loadGENIA(genia_sents[i])	
			i += 1
		#GUARDAR ORACIONES
		print 'Dumping parsed sentences to '+outputFileName+'...'
		pickle.dump(sentences, open(outputFileName, "wb" ))
		outputFile.close()
		print 'Bye!'

if __name__ == "__main__":
    main()

