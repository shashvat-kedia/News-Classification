import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.contrib.layers as layers
import os
import time
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    tokens_wo_stopwords = []
    for i in range(0,len(tokens)):
        if tokens[i].lower() not in stop_words:
            tokens_wo_stopwords.append(tokens[i].lower())
    return tokens_wo_stopwords

def get_pos_tag(token):
    pos_tag = nltk.pos_tag([token])[0][1]
    if pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    for i in range(0,len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i],pos=str(get_pos_tag(tokens[i])))
    return tokens

def preprocess(sentences):
    for i in range(0,len(sentences)):
        tokens_comment = word_tokenize(sentences[i])
        tokens_comment = remove_stopwords(tokens_comment)
        tokens_comment = lemmatize(tokens_comment)
    processed_str = ""
    for i in range(0,len(tokens_comment)):
        processed_str += str(tokens_comment[i]) + " "
    return processed_str

def create_dataset_from_chunks(path):
    starttime = time.time()
    dataset = {
    'label': [],
    'news': []
    }
    categories = os.listdir(path)
    for i in range(0,len(categories)):
        if(categories[i][0] != '.'):
            documents = os.listdir(path + categories[i])
            for j in range(0,len(documents)):
                if(documents[j][0] != '.'):
                    dataset['label'].append(categories[i])
                    with open(path + categories[i] + '/' +  documents[j],'rb') as f:
                        dataset['news'].append(str(f.read()))
    data = pd.DataFrame(data=dataset)
    data.to_csv(path + '/dataset.csv')
    print("Dataset shape:- ")
    print(data.shape)
    endtime = time.time()
    print("Time taken to create dataset from chunks:- ")
    print(endtime - starttime)
    return data
    
if __name__ == '__main__':
    if(not os.path.exists('data/dataset/dataset.csv')):
        print("Creating dataset from chunks:- ")
        dataset = create_dataset_from_chunks('data/dataset/')
    else:
        dataset = pd.read_csv('data/dataset/dataset.csv')
    print(dataset.shape)
    print(dataset.head())
    for index,row in dataset.iterrows():
        row['news'] = preprocess(sent_tokenize(row['news']))
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(dataset['news']).toarray()
    labelEncoder = LabelEncoder()
    labels = labelEncoder.fit_transform(dataset['label'])
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=22)
    randomForestClassifier = RandomForestClassifier()
    randomForestClassifier.fit(X_train,y_train)
    predictions = randomForestClassifier.predict(X_test)
    print(accuracy_score(y_test,predictions))