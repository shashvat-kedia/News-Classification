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
            self.outputs = tf.concat([output_vals[0],output_vals[1]],2)
            self.final_state = tf.concat([output_states[0].c,output_states[1].c],1)
            attention = self.attention(self.outputs)
            #Sentence embeddings to be generated here
            
    def attention(self,inputs):
        with tf.variable_scope('attention_layer',reuse=tf.AUTO_REUSE):
            attention_weights = tf.get_variable('attention_weights',initializer=tf.truncated_normal_initializer(shape=[2 * self.hidden_size]),dtype=tf.float32)
            projection = layers.fully_connected(inputs,2 * self.hidden_size,activation_fn=tf.tanh,scope=tf.get_variable_scope())
            attention = tf.reduce_sum(tf.matmul(projection,attention_weights),axis=2,keep_dims=True)
            attention_softmax = tf.nn.softmax(attention,axis=1)
            final_weights = tf.matmul(projection,attention_softmax)
            output = tf.reduce_sum(final_weights,axis=1)
            return output      