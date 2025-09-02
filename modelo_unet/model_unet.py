#!/usr/bin/env python
# -*- coding: utf-8 -*-

# model_unet.py - U-Net melhorada baseada no paper Niebler et al.
# Incorpora skip connections, batch normalization, e regularizacao adequada

from tensorflow.keras import layers, Model
import tensorflow as tf

def build_enhanced_unet(input_shape=(128, 128, 4), config=None):
    # U-Net melhorada para deteccao de frentes
    if config is None:
        # Configuracao default
        class DefaultConfig:
            DROPOUT_RATE = 0.3
            L2_REGULARIZATION = 1e-4
            USE_BATCH_NORM = True
            INITIAL_FILTERS = 32
            MODEL_DEPTH = 4
        config = DefaultConfig()

    inputs = layers.Input(shape=input_shape)
    
    # Encoder blocks melhorados
    def encoder_block_enhanced(x, filters, stage, dropout_rate):
        # Primeiro bloco convolucional
        conv = layers.Conv2D(
            filters, 3, 
            activation='relu', 
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
            name=f'enc{stage}_conv1'
        )(x)
        
        if config.USE_BATCH_NORM:
            conv = layers.BatchNormalization(name=f'enc{stage}_bn1')(conv)
        
        conv = layers.Dropout(dropout_rate, name=f'enc{stage}_drop1')(conv)
        
        # Segundo bloco convolucional
        conv = layers.Conv2D(
            filters, 3, 
            activation='relu', 
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
            name=f'enc{stage}_conv2'
        )(conv)
        
        if config.USE_BATCH_NORM:
            conv = layers.BatchNormalization(name=f'enc{stage}_bn2')(conv)
        
        # Skip connection residual quando dimensoes compatveis
        if x.shape[-1] == filters:
            conv = layers.Add(name=f'enc{stage}_skip')([conv, x])
        
        # Max pooling para proxima camada
        pool = layers.MaxPooling2D(2, name=f'enc{stage}_pool')(conv)
        
        return conv, pool
    
    # Decoder blocks melhorados
    def decoder_block_enhanced(x, skip_connection, filters, stage, dropout_rate):
        # Upsampling
        up = layers.Conv2DTranspose(
            filters, 2, 
            strides=2, 
            padding='same', 
            name=f'dec{stage}_up'
        )(x)
        
        # Concatenar com skip connection
        merge = layers.Concatenate(name=f'dec{stage}_concat')([up, skip_connection])
        
        # Primeiro bloco convolucional
        conv = layers.Conv2D(
            filters, 3, 
            activation='relu', 
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
            name=f'dec{stage}_conv1'
        )(merge)
        
        if config.USE_BATCH_NORM:
            conv = layers.BatchNormalization(name=f'dec{stage}_bn1')(conv)
            
        conv = layers.Dropout(dropout_rate, name=f'dec{stage}_drop')(conv)
        
        # Segundo bloco convolucional
        conv = layers.Conv2D(
            filters, 3, 
            activation='relu', 
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
            name=f'dec{stage}_conv2'
        )(conv)
        
        if config.USE_BATCH_NORM:
            conv = layers.BatchNormalization(name=f'dec{stage}_bn2')(conv)
        
        return conv
    
    # Build encoder path
    skip_connections = []
    x = inputs
    
    filters = config.INITIAL_FILTERS
    for stage in range(config.MODEL_DEPTH):
        # Dropout progressivo - mais leve no inicio
        dropout_rate = config.DROPOUT_RATE * (0.5 + 0.5 * stage / config.MODEL_DEPTH)
        
        skip, x = encoder_block_enhanced(x, filters, stage, dropout_rate)
        skip_connections.append(skip)
        
        filters *= 2  # Dobrar filtros a cada nivel
    
    # Bottleneck melhorado
    bottleneck = layers.Conv2D(
        filters, 3, 
        activation='relu', 
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
        name='bottleneck_conv1'
    )(x)
    
    if config.USE_BATCH_NORM:
        bottleneck = layers.BatchNormalization(name='bottleneck_bn1')(bottleneck)
        
    bottleneck = layers.Dropout(config.DROPOUT_RATE * 1.2, name='bottleneck_drop')(bottleneck)
    
    bottleneck = layers.Conv2D(
        filters, 3, 
        activation='relu', 
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION),
        name='bottleneck_conv2'
    )(bottleneck)
    
    if config.USE_BATCH_NORM:
        bottleneck = layers.BatchNormalization(name='bottleneck_bn2')(bottleneck)
    
    # Build decoder path
    x = bottleneck
    for stage in range(config.MODEL_DEPTH):
        filters //= 2  # Reduzir filtros
        skip = skip_connections[-(stage + 1)]  # Skip connections em ordem reversa
        dropout_rate = config.DROPOUT_RATE * (1.0 - 0.3 * stage / config.MODEL_DEPTH)
        
        x = decoder_block_enhanced(x, skip, filters, stage, dropout_rate)
    
    # Output layer melhorada
    outputs = layers.Conv2D(
        1, 1, 
        activation='sigmoid', 
        name='output',
        kernel_regularizer=tf.keras.regularizers.l2(config.L2_REGULARIZATION * 0.5)
    )(x)
    
    # Build model
    model = Model(inputs, outputs, name='Enhanced_UNet_Front_Detection')
    
    return model

def build_large_unet(input_shape=(128, 128, 4), config=None):
    # Mantem compatibilidade com codigo existente
    return build_enhanced_unet(input_shape, config)

def build_attention_unet(input_shape=(128, 128, 4), config=None):
    # U-Net with attention gates (implementacao futura)
    # Por enquanto retorna versao enhanced
    return build_enhanced_unet(input_shape, config)

def attention_gate(F_g, F_l, F_int, name):
    # Attention gate implementation baseada no paper
    # "Attention U-Net: Learning Where to Look for the Pancreas"
    
    g = layers.Conv2D(F_int, 1, strides=1, padding='same', name=f'{name}_g')(F_g)
    x = layers.Conv2D(F_int, 1, strides=1, padding='same', name=f'{name}_x')(F_l)
    
    psi = layers.Add(name=f'{name}_add')([g, x])
    psi = layers.Activation('relu', name=f'{name}_relu')(psi)
    psi = layers.Conv2D(1, 1, strides=1, padding='same', name=f'{name}_psi')(psi)
    psi = layers.Activation('sigmoid', name=f'{name}_sigmoid')(psi)
    
    # Upsample psi to match F_l
    upsample_psi = layers.UpSampling2D(size=(F_l.shape[1] // psi.shape[1], 
                                            F_l.shape[2] // psi.shape[2]), 
                                       name=f'{name}_upsample')(psi)
    
    # Element-wise multiplication
    attended_x = layers.Multiply(name=f'{name}_mult')([F_l, upsample_psi])
    
    return attended_x

def build_multiscale_unet(input_shape=(128, 128, 4), config=None):
    # Multi-scale U-Net para capturar frentes em diferentes escalas
    # Implementacao futura baseada em "UNet++: A Nested U-Net Architecture"
    
    if config is None:
        return build_enhanced_unet(input_shape, config)
    
    # Por enquanto retorna versao enhanced
    return build_enhanced_unet(input_shape, config)

def get_model_summary(model):
    # Helper function para imprimir resumo do modelo
    print("=" * 60)
    print("ARQUITETURA DO MODELO")
    print("=" * 60)
    model.summary()
    
    # Contar parametros por tipo
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal de parametros: {total_params:,}")
    print(f"Parametros treinaveis: {trainable_params:,}")
    print(f"Parametros nao treinaveis: {non_trainable_params:,}")
    print("=" * 60)

# Funcoes de compatibilidade
def create_model(config):
    # Interface unificada para criar modelos
    input_shape = (config.INPUT_SIZE, config.INPUT_SIZE, config.N_CHANNELS)
    
    model = build_enhanced_unet(input_shape, config)
    
    if hasattr(config, 'LOG_MEMORY_USAGE') and config.LOG_MEMORY_USAGE:
        get_model_summary(model)
    
    return model

def compile_model(model, config):
    # Compila modelo com configuracoes otimizadas
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Loss function baseada na configuracao
    loss_function = config.get_loss_function()
    
    # Metricas baseadas na configuracao
    metrics = config.get_metrics()
    
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )
    
    return model