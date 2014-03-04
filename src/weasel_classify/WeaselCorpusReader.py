# -*- coding: utf-8 -*- 
from Constants import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader
from nltk.corpus.reader.util import *
from xml.dom.minidom import parseString
from xml.dom.minidom import parse
from Sentence import *
from Document import *
import os
import unicodedata

class WeaselCorpusReader(XMLCorpusReader):
	"""Class for reading the weasel corpus."""
	
	def __init__(self, root, fileids):
		XMLCorpusReader.__init__(self, root, fileids)		
	
	def sents(self,fileids=None):
		"""
		Returns a list of Sentence objects containing all of the sentences in the files specified by the fileids parameter.
		"""
		sentences = concat([self._sents(fileid) for fileid in self.abspaths(fileids)])
		return sentences
		
	def docs(self,fileids=None):
		"""
		Returns a list of Document objects containing all of the documents in the files specified by the fileids parameter.
		"""
		result = []
		for fileid in self.abspaths(fileids):
			dom = parse(fileid)			
			for doc in dom.getElementsByTagName('Document'):				
				d = Document(doc)
				result.append(d)
		return result
		
	def _sents(self,fileid):
		"""
		Returns a list of Sentence objects containing all of the sentences in the specified file. The titles of each document part are not included.
		"""
		result = []	
		dom = parse(fileid)			
		for docpart in dom.getElementsByTagName('DocumentPart'):
			#to avoid titles 
			#if docpart.attributes['type'].nodeValue == 'Text': 
			for sent in docpart.getElementsByTagName('sentence'):
				wws = Sentence(sent)
				result.append(wws)					
		return result	
		
	
	
