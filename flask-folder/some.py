import _pickle as pickle
import re

import sys
import json

import string
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder

#LDA
def test_repl(s):  # From S.Lott's solution
    for c in string.punctuation:
        s=s.replace(c,"")
    return s

def lda(text):

    tokenizer = RegexpTokenizer(r'\w+')
    stop_list = get_stop_words('en')
    stemwords = PorterStemmer()

    texts = []
    docs = []

    text = re.sub(r"http\S+", "", text)
    result = ''.join([i for i in text if not i.isdigit()])
    text = result.split(".")

    for l in text:
        docs.append(l)

    for i in docs:

        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [x for x in tokens if not x in stop_list]
        # print(stopped_tokens)
        # stemmed_tokens = [stemwords.stem(x) for x in stopped_tokens]
        texts.append(stopped_tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=50)

    result_list = ldamodel.print_topics(num_topics=6, num_words=4)
    tags = []
    for i in range(len(result_list)):
        t = result_list[i][1]
        t = t.split(" + ")
        w = []
        for j in t:
            s = j.split("*")[1]
            s = s.replace('"', '')
            w.append(s)
        if (w[0] not in tags):
            tags.append(w[0])
        if (w[1] not in tags):
            tags.append(w[1])

    return tags
#LDA

def normalize_text(s):
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s

def main(text):
    vectorizer = CountVectorizer()
    # output_file = open("files/title_file.txt", "w")
    # data = []
    # with open("files/data.json") as f:
    #     data = json.load(f)

    # for i in range(len(data)):

    # test_text = data[i]["title"]

    news = {"TITLE":[], "CATEGORY":[]}

    with open("output.txt") as f:
        for line in f:
            l = line.split("\t")
            news["TITLE"].append(l[0])
            news["CATEGORY"].append(l[1])
    
    news['TEXT'] = [normalize_text(s) for s in news['TITLE']]

    # pull the data into vectors
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(news['TEXT'])

    encoder = LabelEncoder()
    y = encoder.fit_transform(news['CATEGORY'])

    filename = 'model.sav'
    nb = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

    test_text = text
    test_text1 = normalize_text(test_text)
    tags = lda(test_text1)
    test_text2 = [test_text1]
    test_text3 = vectorizer.transform(test_text2)
    s = nb.predict(test_text3)

    if(s[0] == 0):
        tags.append("business")
    elif(s[0] == 2):
        tags.append("sports")
    elif (s[0] == 1):
        tags.append("sports")
    else:
        tags.append("sports")

    return tags
    # if s == 0:
    #     output_file.write(test_text + "\t" + "b" + "\n")
    # elif s == 1:
    #     output_file.write(test_text + "\t" + "e" + "\n")
    # elif s == 2:
    #     output_file.write(test_text + "\t" + "m" + "\n")
    # elif s == 3:
    #     output_file.write(test_text + "\t" + "t" + "\n")

# print(nb.predict(test_text3))

# text1 = vectorizer.fit_transform(text)
# text2 = vectorizer.transform(text)
# text = vectorizer.fit_transform(text)