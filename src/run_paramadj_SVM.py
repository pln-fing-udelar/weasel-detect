# -*- coding: utf-8 -*- 
from SvmWeaselClassifier import *
from EvaluationResult import *
from Constants import *

print 'Creating SvmClassifier...'
print 'Training file: '+CORPUS_DIR+TRAINING_FILE
print 'Evaluation file: '+CORPUS_DIR+EVALUATION_FILE
classifier = SvmWeaselClassifier(CORPUS_DIR,TRAINING_FILE,EVALUATION_FILE)
print 'Ok.'
parameters = ['-s 0 -t 0 -c 5 -wi 0.1 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.2 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.3 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.4 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.5 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.6 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.7 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.8 -b 1 -q',
 		      '-s 0 -t 0 -c 5 -wi 0.9 -b 1 -q']
training_set = classifier.getTrainingSet(size=TRAIN_SIZE)
heldout = classifier.getEvaluationSet(size=HELDOUT_SIZE)
for p in parameters:
	print 'Training classifier('+p+')...'
	classifier.train(training_set,p)
	print 'Evaluating classifier...'
	result = classifier.evaluate(heldout)
	print "PARAMETERS:"+p
	result.printResults()
print 'Bye!'
