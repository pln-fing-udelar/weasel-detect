# -*- coding: utf-8 -*- 
class EvaluationResult:
	"""Class for holding the results of the evaluation of a classifier."""
	
	def __init__(self, tp,fp,tn,fn):		
		self._true_positives = tp
		self._false_positives = fp
		self._true_negatives = tn
		self._false_negatives = fn
		if (tp+fp)!=0:
			self.precision = float(tp)/(tp+fp)			
		else:
			print "No positives examples predicted?"
			self.precision = 0
		if (tp+fn)!=0:
			self.recall = float(tp)/(tp+fn)			
		else:
			print "No positive examples in the evaluation set."
			self.recall = 0
		if (self.precision+self.recall)!=0:
			self.fmeasure = 2*(self.precision*self.recall)/(self.precision+self.recall)
		else:
			print "Both precision and recall are 0!"
			self.fmeasure = 0
			
	def printResults(self):
		print 'PRINTING RESULT'	
		print '-'*75
		print '   Total examples: '+str(self._true_positives + self._false_negatives + self._true_negatives + self._false_positives)
		print 'Positive examples: '+str(self._true_positives + self._false_negatives)
		print 'Negative examples: '+str(self._true_negatives + self._false_positives)
		print '-'*75
		print '   True positives: '+str(self._true_positives)
		print '  False positives: '+str(self._false_positives)
		print '   True negatives: '+str(self._true_negatives)
		print '  False negatives: '+str(self._false_negatives)		
		print '-'*75
		print 'Measures'
		print '        Precision:', self.precision
		print '           Recall:', self.recall
		print '        F-Measure:', self.fmeasure
		print '-'*75
		
