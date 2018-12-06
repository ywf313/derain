#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope

base_filters = 64
detail_filters = 16 
filters = 64

def inference(input_tensor,reuse=False, is_training=True):

    with tf.variable_scope('bese_net'):
        
        with tf.variable_scope('layer1', reuse=reuse):
            y1 = slim.conv2d(input_tensor, num_outputs=base_filters, 
                           kernel_size=[3, 3],
                           stride=1,
                           padding='SAME',
                           activation_fn=tf.nn.relu,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                           biases_initializer=None,
                           reuse=reuse,
                           scope='conv')
            
        with tf.variable_scope('layer2', reuse=reuse):
            y2 = slim.conv2d(y1, num_outputs=base_filters,
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        rate=2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')
        
        with tf.variable_scope('layer3', reuse=reuse):
            y3 = slim.conv2d(y2, num_outputs=base_filters,
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        rate=2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')
        
        
        with tf.variable_scope('layer4', reuse=reuse):
            y4 = slim.conv2d(y3, num_outputs=base_filters,
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        rate=2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')
        
        with tf.variable_scope('layer5', reuse=reuse):
            y5 = slim.conv2d(y4, num_outputs=base_filters,
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        rate=2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')  
            
        with tf.variable_scope('layer6', reuse=reuse):
            y6 = slim.conv2d(y5, num_outputs=base_filters,
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        rate=2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')
            
        with tf.variable_scope('layer7', reuse=reuse):
            y7 = slim.conv2d(y6, num_outputs=base_filters,
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        rate=2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv') 
            
        with tf.variable_scope('layer8', reuse=reuse):
            y8 = slim.conv2d(y7, num_outputs=base_filters,
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        rate=2,
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')
            
        with tf.variable_scope('layer9', reuse=reuse):
            y9 = slim.conv2d(y8, num_outputs=3, 
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME',
                        activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')
            base=y9
            
    with tf.variable_scope('detail_net', reuse=reuse):
        
        with tf.variable_scope('layer1', reuse=reuse):
            x1 = slim.conv2d(input_tensor, num_outputs=detail_filters, 
                        kernel_size=[3, 3],
                        stride=1,
                        padding='SAME', 
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                        biases_initializer=None,
                        reuse=reuse,
                        scope='conv')
            
        with tf.variable_scope('layer2',reuse=reuse):
            x2_ = bottleneck_layer(x1, scope='bottle_1',reuse=reuse,is_training=is_training )
            x2 = tf.concat([x1,x2_],3)
        
        with tf.variable_scope('layer3', reuse=reuse):
            x3_ = bottleneck_layer(x2, scope='bottle_2',reuse=reuse,is_training=is_training )
            x3 = tf.concat([x2,x3_],3)
        
        with tf.variable_scope('layer4', reuse=reuse):
            x4_ = bottleneck_layer(x3, scope='bottle_3',reuse=reuse,is_training=is_training )
            x4 = tf.concat([x3,x4_],3)
        
        with tf.variable_scope('layer5', reuse=reuse):
            x5_ = bottleneck_layer(x4, scope='bottle_4',reuse=reuse,is_training=is_training )
            x5 = tf.concat([x4,x5_],3)
        
        with tf.variable_scope('layer6', reuse=reuse):
            x6_ = bottleneck_layer(x5, scope='bottle_5',reuse=reuse,is_training=is_training )
            x6 = tf.concat([x5,x6_],3)
        
        with tf.variable_scope('layer7', reuse=reuse):
            x7_ = bottleneck_layer(x6, scope='bottle_6',reuse=reuse,is_training=is_training )
            x7 = tf.concat([x6,x7_],3)
        
        with tf.variable_scope('layer8', reuse=reuse):
            x8_ = bottleneck_layer(x7, scope='bottle_7',reuse=reuse,is_training=is_training )
            x8 = tf.concat([x7,x8_],3)
        
        with tf.variable_scope('layer9', reuse=reuse):
            x9_ = bottleneck_layer(x8, scope='bottle_8',reuse=reuse,is_training=is_training )
            x9 = tf.concat([x8,x9_],3)
        
        with tf.variable_scope('layer10', reuse=reuse):
            x10_ = bottleneck_layer(x9, scope='bottle_9',reuse=reuse,is_training=is_training )
            x10 = tf.concat([x9,x10_],3)
        
        with tf.variable_scope('layer11', reuse=reuse):
            x11_ = bottleneck_layer(x10, scope='bottle_10',reuse=reuse,is_training=is_training )
            x11 = tf.concat([x10,x11_],3)
        
        with tf.variable_scope('layer12', reuse=reuse):
            x12_ = bottleneck_layer(x11, scope='bottle_11',reuse=reuse,is_training=is_training )
            x12 = tf.concat([x11,x12_],3)
        
        with tf.variable_scope('layer13', reuse=reuse):
            x13_ = bottleneck_layer(x12, scope='bottle_12',reuse=reuse,is_training=is_training )
            x13 = tf.concat([x12,x13_],3)
        
        with tf.variable_scope('layer14', reuse=reuse):
            x14_ = bottleneck_layer(x13, scope='bottle_13',reuse=reuse,is_training=is_training )
            x14 = tf.concat([x13,x14_],3)
        
        with tf.variable_scope('layer15', reuse=reuse):
            x15_ = bottleneck_layer(x14, scope='bottle_14',reuse=reuse,is_training=is_training )
            x15 = tf.concat([x14,x15_],3)
        
        with tf.variable_scope('layer16', reuse=reuse):
            x16_ = bottleneck_layer(x15, scope='bottle_15',reuse=reuse,is_training=is_training )
            x16 = tf.concat([x15,x16_],3)
        
        with tf.variable_scope('layer17', reuse=reuse):
            x17_ = bottleneck_layer(x16, scope='bottle_16',reuse=reuse,is_training=is_training )
            x17 = tf.concat([x16,x17_],3)
        
        with tf.variable_scope('layer18', reuse=reuse):
            x18_ = bottleneck_layer(x17, scope='bottle_17',reuse=reuse,is_training=is_training )
            x18 = tf.concat([x17,x18_],3)
        
        with tf.variable_scope('layer19', reuse=reuse):
            x19_ = bottleneck_layer(x18, scope='bottle_18',reuse=reuse,is_training=is_training )
            x19 = tf.concat([x18,x19_],3)
        
        with tf.variable_scope('layer20', reuse=reuse):
            x20_ = bottleneck_layer(x19, scope='bottle_19',reuse=reuse,is_training=is_training )
            x20 = tf.concat([x19,x20_],3)
         
        with tf.variable_scope('layer21', reuse=reuse):
            x21_ = bottleneck_layer(x20, scope='bottle_20',reuse=reuse,is_training=is_training )
            x21 = tf.concat([x20,x21_],3)
        
        with tf.variable_scope('layer22', reuse=reuse):
            x22_ = bottleneck_layer(x21, scope='bottle_21',reuse=reuse,is_training=is_training )
            x22 = tf.concat([x21,x22_],3)
        
        with tf.variable_scope('layer23', reuse=reuse):
            x23_ = bottleneck_layer(x22, scope='bottle_22',reuse=reuse,is_training=is_training )
            x23 = tf.concat([x22,x23_],3)
        
        with tf.variable_scope('layer24', reuse=reuse):
            x24_ = bottleneck_layer(x23, scope='bottle_23',reuse=reuse,is_training=is_training )
            x24 = tf.concat([x23,x24_],3)
        
        with tf.variable_scope('layer25', reuse=reuse):
            x25_ = bottleneck_layer(x24, scope='bottle_24',reuse=reuse,is_training=is_training )
            x25 = tf.concat([x24,x25_],3)
        
        with tf.variable_scope('layer26', reuse=reuse):
            x26_ = bottleneck_layer(x25, scope='bottle_25',reuse=reuse,is_training=is_training )
            x26 = tf.concat([x25,x26_],3)
        
        with tf.variable_scope('layer27', reuse=reuse):
            x27_ = bottleneck_layer(x26, scope='bottle_26',reuse=reuse,is_training=is_training )
            x27 = tf.concat([x26,x27_],3)
        
        with tf.variable_scope('layer28', reuse=reuse):
            x28_ = bottleneck_layer(x27, scope='bottle_27',reuse=reuse,is_training=is_training )
            x28 = tf.concat([x27,x28_],3)
        
        with tf.variable_scope('layer29', reuse=reuse):
            x29_ = bottleneck_layer(x28, scope='bottle_28',reuse=reuse,is_training=is_training )
            x29 = tf.concat([x28,x29_],3)
        
        
        with tf.variable_scope('layer30',reuse=reuse):
            x30 = slim.conv2d(x29, num_outputs=3,
                            kernel_size=[3, 3],
                            stride=1,
                            padding='SAME',
                            activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                            biases_initializer=None,
                            reuse=reuse,
                            scope='conv')
            detail = x30
    with tf.variable_scope('add_layer', reuse=reuse):
            add = base+detail
            out = add
            
            return out,base,detail

        
def bottleneck_layer(x, scope,reuse,is_training):
        with tf.variable_scope(scope):
          
            x = slim.conv2d(x, num_outputs=4 * detail_filters,
                            kernel_size=[1, 1],
                            stride=1,
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                            biases_initializer=None,
                            reuse=reuse,
                            scope=scope+'_conv1_')
            
            
            x = slim.conv2d(x, num_outputs=detail_filters,
                            kernel_size=[3, 3],
                            stride=1,
                            padding='SAME',
                            rate=2,
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                            biases_initializer=None,
                            reuse=reuse,
                            scope=scope+'_conv2')
            return x

def test_graph(train_dir='./tensorboard'):
    input_tensor = tf.constant(np.ones([1, 64, 64, 3]), dtype=tf.float32)
    sess = tf.Session()
    result = inference(input_tensor, reuse=True)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
if __name__ == "__main__":
   tf.reset_default_graph()
   test_graph()
        
  