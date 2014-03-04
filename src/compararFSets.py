# -*- coding: utf-8 -*- 
from weasel_classify.NaiveBayesWeaselClassifier import NaiveBayesWeaselClassifier
from weasel_classify.ActiveNaiveBayesWeaselClassifier import ActiveNaiveBayesWeaselClassifier
import pickle

def comparaOraciones(oracionA, oracionB):
	if oracionA[1] != oracionB[1]:
		return False
	for ((ka,va), (kb,vb)) in zip(oracionA[0].items(),oracionA[0].items()):
		if ka != kb:
			"Different keys!"
			return False
		if va != vb:
			"Different values!"
			return False		
	return True

oracionesNB = pickle.load(open("oracionesNB.pickle","rb"))
oracionesAL = pickle.load(open("oracionesAL.pickle","rb"))
todoOK = True
count = 0
for (a,b) in zip(oracionesNB,oracionesAL):
	if comparaOraciones(a,b):
		#print "Ok."
		todoOK = todoOK or todoOK
	else:
		todoOK = False
		print "Error!"
	count += 1
print "Total checks: "+str(count)
if todoOK:
	print "All OK!"
