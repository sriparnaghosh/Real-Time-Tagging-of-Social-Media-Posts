import re
import string
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from data import (load_line_corpus)

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