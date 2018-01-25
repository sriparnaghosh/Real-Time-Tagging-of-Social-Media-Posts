import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data import (load_line_corpus)
from stop_words import get_stop_words

stop_list = get_stop_words('en')

def bow(clean_train_reviews):

	print "Creating the bag of words...\n"

	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()
	vocab = vectorizer.get_feature_names()
	dist = np.sum(train_data_features, axis = 0)

	zipped = zip(vocab, dist)
	zipped = sorted(zipped, key = lambda t: t[1])[::-1]

	bow_list =  [tag for tag, count in zipped]

	bow_list = [i for i in bow_list if not i in stop_list]

	print bow_list[:5]

if __name__=="__main__":
	corpus_path = "cricket.txt"

	docs = load_line_corpus(corpus_path)
	
	final = []
	for i in docs:
		temp = [(' '.join(i)).encode('utf-8')]
		final = final + temp

	bow(final)