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
import re
from numpy import nan

stop_words = set(stopwords.words('english'))

dictionary = []
embeddings = {}

class HAN():
    
    def __init__(self,num_classes,embedding_size,max_no_sentence,max_sentence_length,hidden_size,batch_size,epochs,learning_rate):
        self.X = tf.placeholder(shape=[None,max_no_sentence,max_sentence_length],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None],dtype=tf.int64)
        self.sequence_length = tf.placeholder(shape=[None,max_no_sentence],dtype=tf.int64)
        self.document_length = tf.placeholder(shape=[None],dtype=tf.int64)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.max_no_sentence = max_no_sentence
        self.max_sentence_length = max_sentence_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model()
    
    def attention(self,inputs):
        with tf.variable_scope('attention_layer',reuse=tf.AUTO_REUSE):
            attention_weights = tf.get_variable('attention_weights',initializer=tf.truncated_normal_initializer(shape=[2 * self.hidden_size]),dtype=tf.float32)
            projection = layers.fully_connected(inputs,2 * self.hidden_size,activation_fn=tf.tanh,scope=tf.get_variable_scope())
            attention = tf.reduce_sum(tf.matmul(projection,attention_weights),axis=2,keep_dims=True)
            attention_softmax = tf.nn.softmax(attention,axis=1)
            final_weights = tf.matmul(projection,attention_softmax)
            output = tf.reduce_sum(final_weights,axis=1)
            return output   
        
    def model(self):
        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        lstm_cell_se_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        lstm_cell_se_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        with tf.variable_scope('bidirectional_lstm'):
            output_vals,output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = lstm_cell_fw,
            cell_bw = lstm_cell_bw,
            inputs = self.X,
            sequence_length = self.sequence_lengths,
            dtype = tf.float32
            )
        self.outputs = tf.concat([output_vals[0],output_vals[1]],2)
        self.final_state = tf.concat([output_states[0].c,output_states[1].c],1)
        attention_op = self.attention(self.outputs)
        attention_op = tf.reshape(attention_op,[None,self.max_no_sentence,self.max_sentence_length])
        with tf.variable_scope('bidirectional_lstm_se'):
            output_vals_se,output_states_se = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = lstm_cell_se_fw,
            cell_bw = lstm_cell_se_bw,
            inputs = attention_op,
            sequence_length = self.sentence_lengths,
            dtype = tf.float32
            )    
        self.final_state_se = tf.concat([output_states_se[0].c,output_states_se[1].c],1)
        attention_op_se = self.attention(self.outputs_se)
        with tf.variable_scope('softmax',reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('W',initializer=tf.truncated_normal_initializer(shape=[None,2 * self.hidden_size]),dtype=tf.float32)
            self.b = tf.get_variable('b',initializer=tf.constant_initializer(0.0,shape=[None]),dtype=tf.float32)
        self.logits = self.matmul(self.final_state_se,self.W) + self.b
        self.predictions= tf.nn.softmax(self.logits)
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
        self.cost = tf.reduce_mean(self.cross_entropy)
        
    def train(self,X,document_lengths,sequence_lengths,y):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = optimizer.minimize(self.cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1,self.epochs + 1):
                cost = 0
                for i in range(0,math.ceil(X.shape[3] / self.batch_size)):
                    X_batch = X[i * self.batch_size : min(X.shape[3], (i + 1) * self.batch_size)]
                    y_batch = y[i * self.batch_size : min(X.shape[3], (i + 1) * self.batch_size)]
                    sequence_lengths_batch = sequence_lengths[i * self.batch_size : min(X.shape[3],(i + 1) * self.batch_size)]
                    document_lengths_batch = document_lengths[i * self.batch_size : min(X.shape[3],(i + 1) * self.batch_size)]
                    fetches = {
                    'cross_entropy': self.cross_entropy,
                    'cost': self.cost,
                    'predictions': self.predictions,
                    'train_step': self.train_step
                    }
                    feed_dict = {
                    self.X: X_batch,
                    self.y: y_batch,
                    self.sequence_length: sequence_lengths_batch,
                    self.document_length: doucment_lengths_batch
                    }
                    resp = sess.run(fetched,feed_dict)
                    cost += resp['cost']   
                print("Cost at epoch: " + str(i))
                print(cost)

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

def add_to_dictionary(tokens):
    for token in tokens:
        if token not in dictionary:
            dictionary.append(token)

def save_dictionary():
    with open('data/processed/dictionary.txt','w') as file:
        file.writelines("%s\n" % word for word in dictionary)

def read_dictionary():
    with open('data/processed/dictionary.txt','r') as file:
        temp = file.read().splitlines()
        for i in range(0,len(temp)):
            dictionary.append(temp[i])

def create_dictionary(dataset):
    for index,row in dataset.iterrows():
        tokens_comment = preprocess(str(row['parent_comment']) + " " + str(row['comment']))
        add_to_dictionary(tokens_comment)
    save_dictionary()

def populate_dictionary():
    if not os.path.isfile('data/processed/dictionary.txt'):
        starttime = time.time()
        create_dictionary(df_new)
        endtime = time.time()
        print("Time to create dictionary")
        print(endtime - starttime)
    else:   
        read_dictionary()
    print("Length of dictionary:- ")
    print(len(dictionary))

def populate_embeddings_dict():
    starttime = time.time()
    with open('data/processed/glove.6B.300d.txt','r') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = np.asarray(values[1:])
            embeddings[word] = word_embedding
    endtime = time.time()
    print("Time taken to load embeddings:- ")
    print(endtime - starttime)

def pad_tokens(tokens,max_length):
    zeros = np.zeros(len(tokens[0]))
    while len(tokens) < max_length:
        tokens = np.vstack([tokens,zeros])
    return tokens

def embedding_lookup(x,embedding_dim=300):
    if(len(embeddings) == 0):
        populate_embeddings_dict()
    embedding = []
    for i in range(0,len(x)):
        if(x[i] in embeddings):
            embedding.append(embeddings[x[i]])
        else:
            zero_arr = np.zeros(embedding_dim).tolist()
            embedding.append(zero_arr)
    return np.array(embedding)

def get_sentence_and_words_attr(dataset):
    max_no_sentence = 0
    max_no_words = 0
    document_lengths = []
    sentence_lengths = []
    for index,rows in dataset.iterrows():
        sentence_tokens = sent_tokenize(rows['news'])
        if max_no_sentence < len(sentence_tokens):
            max_no_sentence: len(sentence_tokens)
        sentence_tokens = preprocess(sentence_tokens)
        for i in range(0,len(sentence_tokens)):
            if max_no_words < len(sentence_tokens[i]):
                max_no_words = len(sentence_tokens[i])
    return max_no_sentence,max_no_words

def preprocess(sentence):
    tokens_comment = word_tokenize(sentence)
    tokens_comment = remove_stopwords(tokens_comment)
    tokens_comment = lemmatize(tokens_comment)
    return tokens_comment

def dataset_preprocess(sentences,max_no_sentence,max_sentence_length):
    processed_tokens = []
    sentence_lengths = []
    for i in range(0,len(sentences)):
        tokens_comment = preprocess(sentences[i])
        sentence_length.append(tokens_comment)
        if(len(tokens_comment) < max_sentence_length):
            for j in range(len(tokens_comment),max_length):
                tokens_comment.append("<PAD>")
        processed_tokens.append(tokens_comment)
    if(len(processed_tokens) < max_no_sentence):
        processed_tokens.append(np.zeros(shape=[max_sentence_length]))
        sentence_lengths.append(0)
    return processed_tokens,len(sentences),sentence_lengths

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
                        dataset['news'].append(sent_tokenize(str(f.read())))
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
    max_no_sentence,max_no_words = get_sentence_and_words_attr(dataset)
    labels = []
    preprocessed_dataset = []
    document_lengths = []
    sentence_lengths = []
    preprocessed_dataset_train = []
    preprocessed_dataset_test = []
    document_lengths_train = []
    document_lengths_test = []
    sentence_lengths_train = []
    sentence_lengths_test = []
    for index,rows in dataset.iterrows():
        labels.append(rows['label'])
        processed_tokens,doc_length,sent_lengths = dataset_preprocess(rows['news'])
        preprocessed_dataset.append(processed_tokens)
        document_lengths.append(doc_length)
        sentence_lengths.append(sentence_lengths)
    labels = np.array(labels)
    X_train_index,X_test_index,y_train,y_test = train_test_split(np.arang(len(document_lengths)),labels,test_size=0.2,random_state=222)
    for i in range(0,len(X_train)):
        preprocessed_dataset_train.append(preprocessed_dataset[X_train[i]])
        document_lengths_train.append(document_lengths[X_train[i]])
        sentence_lengths_train.append(sentence_lengths[X_train[i]])
    for i in range(0,len(X_test)):
        preprocessed_dataset_test.append(preprocessed_dataset[X_test[i]])
        document_lengths_test.append(document_lengths[X_test[i]])
        sentence_lengths_test.append(sentence_lengths[X_test[i]])
    preprocessed_dataset_train = np.array(preprocessed_dataset_train)
    preprocessed_dataset_test = np.array(preprocessed_dataset_test)
    document_lengths_train = np.array(document_lengths_train)
    document_lengths_test = np.array(document_lengths_test)
    sentence_lengths_train = np.array(sentence_lengths_train)
    sentence_lengths_test = np.array(sentence_lengths_test)
    han = HAN(5,1024,max_no_sentence,max_no_words,1024,400,10,0.001)
    han.train(preprocessed_dataset_train,document_lengths_train,sentence_lengths_train,y_train)
    