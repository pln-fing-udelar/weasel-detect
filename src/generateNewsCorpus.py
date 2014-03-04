import feedparser
import pickle
from weasel_classify.Sentence import Sentence

feedURLs = ["http://thecaucus.blogs.nytimes.com/feed/", "http://www.nytimes.com/timeswire/feeds/"]
sentences = []
for url in feedURLs:
	feed = feedparser.parse(url)
	count = 0
	for item in feed["items"]:
		sent = Sentence(None,item["description"],feed["title"]+"."+count)
		sentences.append(sent)
		count += 1


sentsFile = "news_corpus.pickle"
pickle.dump(sentences,open(sentsFile,"wb"))
for sent in sentences:
	print sent.encode("UTF-8")

