# -*- coding: utf-8 -*- 
from Constants import *
from xml.dom.minidom import parseString
import nltk.tokenize
import subprocess
import codecs
import os


class Sentence:
	"""Class for corpus sentences"""
		
	def __init__(self, sentElem=None, sentString=None, ident=None):
		"""
			@param sentElem: An object of type ...
		"""		
		if sentElem != None:
			self.id = sentElem.getAttribute('id')
			#create Tokenizer
			wt=nltk.tokenize.TreebankWordTokenizer()
			if sentElem.getAttribute('certainty') == u'certain':
				#no weasels in this sentence, only one child
				self.string = sentElem.childNodes[0].toxml()
				self.certainty = 'certain'
				self.matrix = wt.tokenize(self.string)
		                
			elif sentElem.getAttribute('certainty') == u'uncertain':
				#find the weasels
				self.string = ''
				self.certainty = 'uncertain'
				self.matrix = []			
				for elem in sentElem.childNodes:
					if elem.nodeType == 1:
						#element
						if elem.tagName == u'ccue':
							#weasel!
							self.matrix.append([])
							weasel_phrase = elem.childNodes[0].toxml()
							self.string += ' '+weasel_phrase #prevents blanks loss, TODO						
							for token in wt.tokenize(weasel_phrase):
								self.matrix[-1].append(token)			
					#just text
					elif elem.nodeType == elem.TEXT_NODE:                            
						sent = elem.toxml()
						self.string += ' '+sent #prevents blanks loss, TODO
						for token in wt.tokenize(sent):
							self.matrix.append(token)
		elif sentString != None and ident != None:
			self.id = ident
			self.string = sentString
			self.certainty = "unknown"
		else:
			raise "Invalid arguments for Sentence constructor."
		
	def words(self):
		wt=nltk.tokenize.TreebankWordTokenizer()
		return 	wt.tokenize(self.string)


	def loadGENIA(self, geniaRawWords):
		"""
		Loads the GENIA tags for each token in the attribute genia_words from the given string containing the GENIA output of the sentence's string.
		"""
		self.genia_words = []				
		for word in geniaRawWords.rstrip(os.linesep).split(os.linesep):
			(word,lemma,pos,chunk,ne) = word.rstrip(os.linesep).split('\t')		
			self.genia_words.append((word,lemma,pos,chunk,ne))

			




		
