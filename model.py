import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.contrib.layers as layers
import tensorflow_hub as hub
import os

def create_dataset_from_chunks(path):
    print(os.listdir(path))
    

class Han():
    
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
        self.final_state_se = tf.concat([output_states_se[0].c,output_states_se[1].c)],1)
        attention_op_se = self.attention(self.outputs_se)
        with tf.variable_scope('softmax',reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('W',initializer=tf.truncated_normal_initializer(shape=[None,2 * self.hidden_size]),dtype=tf.float32)
            self.b = tf.get_variable('b',initializer=tf.constant_initializer(0.0,shape=[None]),dtype=tf.float32)
        self.logits = self.matmul(self.final_state_se,self.W) + self.b
        self.predictions= tf.nn.softmax(self.logits)
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
        self.cost = tf.reduce_mean(self.cross_entropy)
        
    def train(self,X,y,sequence_lengths,document_lengths):
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
                print()