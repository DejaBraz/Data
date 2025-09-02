#!/usr/bin/env python
# -*- coding: utf-8 -*-

# model_unet.py - U-Net balanceada para RTX 3050
# Sem acentos, performance otimizada

from tensorflow.keras import layers, Model
import tensorflow as tf

def attention_block(x, g, filters):
    """
    Attention Gate para focar em regioes importantes
    """
    # Gating signal
    g1 = layers.Conv2D(filters, 1, padding='same')(g)
    g1 = layers.BatchNormalization()(g1)
    
    # Input signal
    x1 = layers.Conv2D(filters, 1, padding='same')(x)
    x1 = layers.BatchNormalization()(x1)
    
    # Combine
    psi = layers.Add()([g1, x1])
    psi = layers.Activation('relu')(psi)
    psi = layers.Conv2D(1, 1, padding='same')(psi)
    psi = layers.Activation('sigmoid')(psi)
    
    # Apply attention
    return layers.Multiply()([x, psi])

# No decoder_block, adicione attention antes do merge:
def decoder_block(x, skip, filters, dropout_rate, name):
    up = layers.Conv2DTranspose(filters, 2, strides=2, padding='same', name=f'{name}_up')(x)
    
    # NOVO: Attention gate
    skip_att = attention_block(skip, up, filters//2)
    
    merge = layers.Concatenate(name=f'{name}_concat')([up, skip_att])
    

def build_large_unet(input_shape=(128,128,10), config=None):
    if config is None:
        class Dummy: pass
        config = Dummy()
        config.DROPOUT_RATE = 0.25
        config.L2_REGULARIZATION = 5e-5

    inputs = layers.Input(shape=input_shape)

    def encoder_block(x, filters, dropout_rate, name):
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
                             name=f'{name}_conv1')(x)
        conv = layers.BatchNormalization(name=f'{name}_bn1')(conv)
        conv = layers.Dropout(dropout_rate, name=f'{name}_drop')(conv)
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
                             name=f'{name}_conv2')(conv)
        conv = layers.BatchNormalization(name=f'{name}_bn2')(conv)
        
        # Skip connection quando possivel
        if x.shape[-1] == filters:
            conv = layers.Add(name=f'{name}_skip')([conv, x])
            
        pool = layers.MaxPooling2D(2, name=f'{name}_pool')(conv)
        return conv, pool

    def decoder_block(x, skip, filters, dropout_rate, name):
        up = layers.Conv2DTranspose(filters, 2, strides=2, padding='same', 
                                   name=f'{name}_up')(x)
        merge = layers.Concatenate(name=f'{name}_concat')([up, skip])
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
                             name=f'{name}_conv1')(merge)
        conv = layers.BatchNormalization(name=f'{name}_bn1')(conv)
        conv = layers.Dropout(dropout_rate, name=f'{name}_drop')(conv)
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
                             name=f'{name}_conv2')(conv)
        conv = layers.BatchNormalization(name=f'{name}_bn2')(conv)
        return conv

    # Encoder - filtros balanceados
    c1, p1 = encoder_block(inputs, 48, config.DROPOUT_RATE*0.5, 'enc1')
    c2, p2 = encoder_block(p1, 96, config.DROPOUT_RATE*0.7, 'enc2')  
    c3, p3 = encoder_block(p2, 192, config.DROPOUT_RATE*0.9, 'enc3')
    c4, p4 = encoder_block(p3, 384, config.DROPOUT_RATE, 'enc4')

    # Bottleneck
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
                               name='bottleneck_conv1')(p4)
    bottleneck = layers.BatchNormalization(name='bottleneck_bn1')(bottleneck)
    bottleneck = layers.Dropout(config.DROPOUT_RATE * 1.2, name='bottleneck_drop')(bottleneck)
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
                               name='bottleneck_conv2')(bottleneck)
    bottleneck = layers.BatchNormalization(name='bottleneck_bn2')(bottleneck)

    # Decoder
    u1 = decoder_block(bottleneck, c4, 384, config.DROPOUT_RATE, 'dec1')
    u2 = decoder_block(u1, c3, 192, config.DROPOUT_RATE*0.9, 'dec2')
    u3 = decoder_block(u2, c2, 96, config.DROPOUT_RATE*0.7, 'dec3')
    u4 = decoder_block(u3, c1, 48, config.DROPOUT_RATE*0.5, 'dec4')

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid', name='output')(u4)

    model = Model(inputs, outputs, name='Balanced_UNet')
    return model