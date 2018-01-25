# get some libraries that will be useful
import sys
import re
import json
import lda
import _pickle as pickle

# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder


# grab the data

# news = {"ID":[], "TITLE":[], "URL":[], "PUBLISHER":[], "CATEGORY":[]}

# with open("files/test.csv") as f:
#     for line in f:
#         text = line.strip().split("\t")
#         news["ID"].append(text[0])
#         news["TITLE"].append(text[1])
#         news["URL"].append(text[2])
#         news["PUBLISHER"].append(text[3])
#         news["CATEGORY"].append(text[4])

news = {"TITLE":[], "CATEGORY":[]}

with open("output.txt") as f:
    for line in f:
        l = line.split("\t")
        news["TITLE"].append(l[0])
        news["CATEGORY"].append(l[1])

def normalize_text(s):
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s

news['TEXT'] = [normalize_text(s) for s in news['TITLE']]

# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(news['TEXT'])

encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 7)

# take a look at the shape of each of these
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

nb = MultinomialNB()
nb.fit(x_train, y_train)
score = nb.score(x_test, y_test)
print(score)
filename = 'model.sav'
pickle.dump(nb, open(filename, 'wb'))

# text = [normalize_text("Marijuana facility planned for Stellarton")]

# def main():
    
#     # output_file = open("files/title_file.txt", "w")
#     # data = []
#     # with open("files/data.json") as f:
#     #     data = json.load(f)

#     # for i in range(len(data)):

#     # test_text = data[i]["title"]
#     with open("article.txt") as f:
#         for line in f:
#             test_text = line.strip()
#             test_text1 = normalize_text(test_text)
#             # tags = lda.lda(test_text1)
#             # tag_text = ' '.join(tags)
#             test_text2 = [test_text1]
#             test_text3 = vectorizer.transform(test_text2)
#             s = nb.predict(test_text3)
#             print(s)
#             # if s == 0:
#             #     output_file.write(test_text + "\t" + "b" + "\n")
#             # elif s == 1:
#             #     output_file.write(test_text + "\t" + "e" + "\n")
#             # elif s == 2:
#             #     output_file.write(test_text + "\t" + "m" + "\n")
#             # elif s == 3:
#             #     output_file.write(test_text + "\t" + "t" + "\n")
        
#         # print(nb.predict(test_text3))

# main()

# # text1 = vectorizer.fit_transform(text)
# # text2 = vectorizer.transform(text)
# # text = vectorizer.fit_transform(text)