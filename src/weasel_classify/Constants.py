# -*- coding: utf-8 -*- 
DEBUG = True
CORPUS_DIR = '../../corpus/'
GENIAHOME = '../../lib/geniatagger-3.0.1'
GENIATRAIN = 'task1_train_wikipedia_rev2.xml.genia'
GENIAEVAL = 'task1_weasel_eval.xml.genia'
TRAINING_FILE = "task1_train_wikipedia_rev2.xml"
EVALUATION_FILE = "task1_weasel_eval.xml"
TRAIN_SENTS_FILE = "train_sentences.pickle"
EVAL_SENTS_FILE = "eval_sentences.pickle"
TRAIN_SIZE = 0.9
HELDOUT_SIZE = 0.1
EVAL_SIZE = 0.1
INITIAL_QUERIES = 0.3
BATCH_SIZE = 50
THRESHOLD = 0.52
RSSFEEDURLS = [
	"http://thecaucus.blogs.nytimes.com/feed/", 
	"http://www.nytimes.com/timeswire/feeds/",
	"http://fivethirtyeight.blogs.nytimes.com/feed/",
	"http://newoldage.blogs.nytimes.com/feed/",
	"http://well.blogs.nytimes.com/feed/",
	"http://scientistatwork.blogs.nytimes.com/feed/"]
#PARAMETER ADJUST
MOSTCOMMON = [5000,4000,3000,2000,1000,500]
STOPWORDS = [50,20,10,0]
BOOLCOUNT = [True,False]
USEBOL = [True]
USENES = [True]
USEWEXES = [False]
