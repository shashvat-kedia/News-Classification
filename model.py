import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow.contrib.layers as layers

class Han():
    
    def __init__(self,num_classes,embedding_size,max_no_sentence,max_sentence_length,hidden_size,batch_size,epochs):
        self.X = tf.placeholder(shape=[None,max_no_sentence,max_sentence_length],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None],dtype=tf.int64)
        self.sequence_lengths = tf.placeholder(shape=[None],dtype=tf.int64)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.max_no_sentence = max_no_sentence
        self.max_sentence_length = max_sentence_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model()
        
    def model(self):
        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        with tf.variable_scope('bidirectional_lstm'):
            output_vals,output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = lstm_cell_fw,
            cell_bw = lstm_cell_bw,
            inputs = self.X,
            sequence_length = self.sequence_lengths,
            dtype = tf.float32
            )
            self.outputs = tf.concat([output_vals[0],output_vals[1]],axis=2)
            print(tf.shape(self.outputs))
            self.final_state = tf.concat([output_states[0].c,output_states[0].c],axis=1)
            print(tf.shape(self.final_state))
            with tf.variable_scope('dense'):
                self.W = tf.get_variable('W',shape=[2 * self.hidden_size,self.num_classes],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
                self.b = tf.get_variable('b',shape=[2],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
            self.logits = tf.matmul(self.final_state,self.W) + self.b 
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y))
            
    def attention(self,inputs):
        with tf.variable_scope('attention_layer',reuse=tf.AUTO_REUSE):
            attention_weights = tf.get_variable('attention_weights',initializer=tf.truncated_normal_initializer(shape=[2 * self.hidden_size]),dtype=tf.float32)
            projection = layers.fully_connected(inputs,2 * self.hidden_size,activation_fn=tf.tanh,scope=tf.get_variable_scope())
            attention = tf.reduce_sum(tf.matmul(projection,attention_weights),axis=2,keep_dims=True)
            attention_softmax = tf.nn.softmax(attention,axis=1)
            final_weights = tf.matmul(projection,attention_softmax)
            output = tf.reduce_sum(final_weights,axis=-1)
            return output
    
    def train(self,X,y,sequence_length):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_step = optimizer.minimize(self.cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            fetches = {
            'outputs': self.outputs,
            'final_state': self.final_state
            }
            feed_dict = {
            self.X: X,
            self.y: y,
            self.sequence_lengths: sequence_length
            }
            resp = sess.run(fetches,feed_dict)
            print('Outputs:- ')
            print(resp['outputs'])
            print('Final State:- ')
            print(resp['final_state'])


X = tf.Variable(tf.truncated_normal(shape=[10,10,20]),dtype=tf.float32)
y = tf.Variable(tf.constant(1,shape=[10],dtype=tf.int64))
sequence_length = tf.Variable(tf.constant(20,shape=[10],dtype=tf.int64))
han = Han(2,0,10,20,10,0,0)         
han.train(X,y,sequence_length)   