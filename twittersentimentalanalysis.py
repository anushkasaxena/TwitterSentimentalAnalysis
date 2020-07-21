import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import re


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import *
from sklearn.metrics import *

def clean_data(tweet):
    temp = TextBlob(tweet).words
    tb = ' '.join(temp)

    tweet_list = [ele for ele in tb.split() if ele != 'user']
    tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(tokens)
    clean_tweet = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]

    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word, 'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet

def lst_to_str(s):
    listToStr = ' '.join([str(elem) for elem in s])
    return listToStr

def apply_naive_bayes(X_train, Y_train):
    modelNB = MultinomialNB()
    modelNB.fit(X_train, Y_train)
    Y_test_predict = modelNB.predict(X_test)
    return Y_test_predict, modelNB

def apply_random_forest(X_train, Y_train):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    Y_test_predict = model.predict(X_test)
    return Y_test_predict, model

def accuraacy_parameter(Y_test_predict, Y_test):
    temp2 = mean_squared_error(Y_test, Y_test_predict)
    print('Mean square error on testing data: ', temp2)
    print("Accuracy score is: ", accuracy_score(Y_test_predict, Y_test))
    print("Precision score is: ", precision_score(Y_test, Y_test_predict))
    print("Recall score is: ", recall_score(Y_test, Y_test_predict))
    print("f1 score is: ", f1_score(Y_test, Y_test_predict))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test_predict, Y_test))



testdata = pd.read_csv('test.csv')
traindata = pd.read_csv('train.csv')
testdata.info()
traindata.info()
print("wait for some time")
print("data processing.....")
traindata['cleandata'] = traindata['tweet'].apply(clean_data)
traindata['processed_data'] = traindata.cleandata.apply(lst_to_str)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(traindata.processed_data)
Y = traindata.label
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)

print("----APPLY NAIVE BAYES----")
Y_test_predict, modelNB = apply_naive_bayes(X_train, Y_train);
accuraacy_parameter(Y_test_predict, Y_test)

print("----APPLY RANDOM FOREST----")
Y_test_predict, model = apply_random_forest(X_train, Y_train)
accuraacy_parameter(Y_test_predict, Y_test)

