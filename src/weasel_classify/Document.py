# -*- coding: utf-8 -*- 
from xml.dom.minidom import parseString
from Sentence import Sentence


class Document:
	"""Class for corpus Documents"""
		
	def __init__(self, docElem):
		"""
			@param docElem: An object of type ...
		"""
		self.wikipedia_article_id = docElem.getElementsByTagName('DocID')[0].getAttribute('WIKIPEDIA_ARTICLE_ID')
		self.type = docElem.getAttribute('type')
		self.docparts = []
		for docPartElem in docElem.getElementsByTagName('DocumentPart'):
				docpart = DocumentPart(docPartElem)
				self.docparts.append(docpart)
		
		
class DocumentPart:
	"""Class for corpus Document parts"""
		
	def __init__(self, docPartElem):
		"""
			@param docPartElem: An object of type ...
		"""
		self.type = docPartElem.getAttribute('type')
		self.sents = []
		for sentElem in docPartElem.getElementsByTagName('sentence'):
				sent = Sentence(sentElem)
				self.sents.append(sent)		
	
