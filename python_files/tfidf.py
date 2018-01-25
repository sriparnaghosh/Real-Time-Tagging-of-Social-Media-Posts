import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from stop_words import get_stop_words

stop_list = get_stop_words('en')

def load_line_corpus(path, tokenize=True):
    docs = []
    tokenizer = RegexpTokenizer(r'\w+')
    with codecs.open(path, "r", "utf8") as f:
        for l in f:
            if tokenize:
                docs.append(l)
    return docs

def bow(clean_train_reviews):

	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()
	vocab = vectorizer.get_feature_names()

	transformer = TfidfTransformer()
	transformed_weights = transformer.fit_transform(train_data_features)

	weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
	weights_df = pd.DataFrame({'term': vocab, 'weight': weights})
	weights_df = weights_df.sort_values(by='weight', ascending=False)
	tfidf_list = weights_df.loc[:, 'term'].tolist()

	tfidf_list = [i for i in tfidf_list if not i in stop_list]

	print (tfidf_list[:5])


if __name__=="__main__":
	corpus_path = "cricket.txt"

	docs = []
	with open(corpus_path, "r") as f:
		for l in f:
			docs.append(l)

	final = []
	for i in docs:
		temp = [(' '.join(i)).encode('utf-8')]
		final = final + temp
	print(final)

	bow(final)