import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from typing import Optional, List, Union

# Model
def create_model_lstm(input_shape, input_units=30, learning_rate=0.0001, **kwargs):
    # Defining Model
    model = Sequential()
    model.add(LSTM(input_units, return_sequences=False, activation='relu', dropout=0.2, recurrent_dropout=0.2, 
                    input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(2))
    # Compilation Model
    optim = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="MSE", optimizer=optim)
    #model.compile(loss=tf.keras.losses.KLDivergence(), optimizer=optim)
    #model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=optim)
    # Display the model's architecture
    #model.summary()
    return model

def create_model_attention(input_shape, learning_rate=0.0001, **kwargs):
    """
    Attention
    """
    # Defining Model
    inputs_ = tf.keras.layers.Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=5, 
                           key_dim=5,
                           dropout=0.4)(inputs_, inputs_)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(80, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(2)(x)
    model = Model(inputs=inputs_, outputs=x)
    optim = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="MSE", optimizer=optim)
    # Display the model's architecture
    #model.summary()
    return model

def create_pretrained_CNN_model(input_shape=(224, 224), input_units=None, learning_rate=0.0001):
    pretrained_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    pretrained_model.trainable = False
    model = tf.keras.models.Sequential()
    model.add(pretrained_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(2))
    optim = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="MSE", optimizer=optim)
    model.summary()
    return model

def create_mobile_net_v2_model(input_shape=(224, 224, 3), **kwargs):
    pretrained_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    model = tf.keras.models.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    model.trainable = False
    return tf.keras.models.clone_model(model)

class SALICONtf():
    """
    https://github.com/ykotseruba/SALICONtf
    """
    def __init__(self, salicon_weights="", vgg16_weights=""):
        self.build_salicon_model(vgg16_weights)
        #load weights if provided
        if salicon_weights:
          self.model.load_weights(salicon_weights)
        self.input = self.model.input
        self.output = self.model.output

    def __call__(self, img=None):
        img = np.squeeze(img)
        smap = self.compute_saliency(img=img)
        return smap.flatten()

    def build_vgg16(self, input_shape, stream_type, weights_path="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
        img_input = tf.keras.layers.Input(shape=input_shape, name='Input_' + stream_type)
        # Block 1
        x = tf.keras.layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block1_conv1_'+stream_type)(img_input)
        x = tf.keras.layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block1_conv2_'+stream_type)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_'+stream_type)(x)
        # Block 2
        x = tf.keras.layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block2_conv1_'+stream_type)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block2_conv2_'+stream_type)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_'+stream_type)(x)
        # Block 3
        x = tf.keras.layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block3_conv1_'+stream_type)(x)
        x = tf.keras.layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block3_conv2_'+stream_type)(x)
        x = tf.keras.layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block3_conv3_'+stream_type)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_'+stream_type)(x)
        # Block 4
        x = tf.keras.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block4_conv1_'+stream_type)(x)
        x = tf.keras.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block4_conv2_'+stream_type)(x)
        x = tf.keras.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block4_conv3_'+stream_type)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_'+stream_type)(x)
        # Block 5
        x = tf.keras.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block5_conv1_'+stream_type)(x)
        x = tf.keras.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block5_conv2_'+stream_type)(x)
        output = tf.keras.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block5_conv3_'+stream_type)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_'+stream_type)(output)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        model = Model(inputs=img_input, outputs=x)
        #initialize each vgg16 stream with ImageNet weights
        try:
            model.load_weights(weights_path, by_name=False)
            model = Model(inputs=img_input, outputs=output)
        except OSError:
            raise Exception("ERROR: VGG weights are not found.")
        return model.input, model.output

    def build_salicon_model(self, vgg16_weights=""):
        #create two streams separately
        fine_stream_input, fine_stream_output = self.build_vgg16(input_shape=(600, 800, 3), stream_type='fine', weights_path=vgg16_weights)
        coarse_stream_input, coarse_stream_output = self.build_vgg16(input_shape=(300, 400, 3), stream_type='coarse', weights_path=vgg16_weights)
        #add interpolation layer to the coarse stream
        H, W = fine_stream_output.shape[1], fine_stream_output.shape[2]
        interp_layer = tf.keras.layers.Lambda(lambda input_tensor: tf.image.resize(input_tensor, (H, W)))(coarse_stream_output)
        #add concatenation layer followed by 1x1 convolution to combine streams
        concat_layer = tf.keras.layers.concatenate([fine_stream_output, interp_layer], axis=-1)
        sal_map_layer = tf.keras.layers.Conv2D(1, (1, 1),
                        name='saliency_map',
                        trainable=True,
                        activation='sigmoid',
                        kernel_initializer=tf.keras.initializers.Zeros(),
                        bias_initializer=tf.keras.initializers.Zeros())(concat_layer)
        self.model = Model(inputs=[fine_stream_input, coarse_stream_input], outputs=sal_map_layer)

    def compute_saliency(self, img_path=None, img=None):
        vgg_mean = np.array([123, 116, 103])
        if img_path:
            img_fine = img_to_array(load_img(img_path,
                grayscale=False,
                target_size=(600, 800),
                interpolation='nearest'))
            img_coarse = img_to_array(load_img(img_path,
                grayscale=False,
                target_size=(300, 400),
                interpolation='nearest'))
        else:
            img_fine = cv2.resize(img.copy(), (800, 600))
            img_coarse = cv2.resize(img.copy(), (400, 300))
        img_fine = img_fine - vgg_mean
        img_coarse = img_coarse - vgg_mean
        img_fine = img_fine[None, :]/255
        img_coarse = img_coarse[None, :]/255
        smap = self.model.predict([img_fine, img_coarse], batch_size=1, verbose=0)
        smap = np.squeeze(smap)
        if img_path:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
        else:
            h, w = img.shape[:2]
        smap = (smap - np.min(smap))/((np.max(smap) - np.min(smap)))
        #smap = cv2.resize(smap, (w, h), interpolation=cv2.INTER_CUBIC)  
        #smap = cv2.GaussianBlur(smap, (75, 75), 25, cv2.BORDER_DEFAULT) 
        return smap

class AddCoords(tf.keras.layers.Layer):
    """Add coords to a tensor"""
    def __init__(self, x_dim=64, y_dim=64, with_r=False, skiptile=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.skiptile = skiptile


    def call(self, input_tensor):
        """
        input_tensor: (batch, 1, 1, c), or (batch, x_dim, y_dim, c)
        In the first case, first tile the input_tensor to be (batch, x_dim, y_dim, c)
        In the second case, skiptile, just concat
        """
        if not self.skiptile:
            input_tensor = tf.tile(input_tensor, [1, self.x_dim, self.y_dim, 1]) # (batch, 64, 64, 2)
            input_tensor = tf.cast(input_tensor, 'float32')

        batch_size_tensor = tf.shape(input_tensor)[0]  # get batch size

        xx_ones = tf.ones([batch_size_tensor, self.x_dim], 
                          dtype=tf.int32)                       # e.g. (batch, 64)
        xx_ones = tf.expand_dims(xx_ones, -1)                   # e.g. (batch, 64, 1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), 
                            [batch_size_tensor, 1])             # e.g. (batch, 64)
        xx_range = tf.expand_dims(xx_range, 1)                  # e.g. (batch, 1, 64)


        xx_channel = tf.matmul(xx_ones, xx_range)               # e.g. (batch, 64, 64)
        xx_channel = tf.expand_dims(xx_channel, -1)             # e.g. (batch, 64, 64, 1)


        yy_ones = tf.ones([batch_size_tensor, self.y_dim], 
                          dtype=tf.int32)                       # e.g. (batch, 64)
        yy_ones = tf.expand_dims(yy_ones, 1)                    # e.g. (batch, 1, 64)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0),
                            [batch_size_tensor, 1])             # (batch, 64)
        yy_range = tf.expand_dims(yy_range, -1)                 # e.g. (batch, 64, 1)

        yy_channel = tf.matmul(yy_range, yy_ones)               # e.g. (batch, 64, 64)
        yy_channel = tf.expand_dims(yy_channel, -1)             # e.g. (batch, 64, 64, 1)

        xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
        xx_channel = xx_channel*2 - 1                           # [-1,1]
        yy_channel = yy_channel*2 - 1

        ret = tf.concat([input_tensor, 
                         xx_channel, 
                         yy_channel], axis=-1)    # e.g. (batch, 64, 64, c+2)

        if self.with_r:
            rr = tf.sqrt( tf.square(xx_channel)
                    + tf.square(yy_channel)
                    )
            ret = tf.concat([ret, rr], axis=-1)   # e.g. (batch, 64, 64, c+3)

        return ret
class CoordConv(tf.keras.layers.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, *args,  **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, 
                                   y_dim=y_dim, 
                                   with_r=with_r,
                                   skiptile=True)
        self.conv = tf.keras.layers.Conv2D(*args, **kwargs)

    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret
