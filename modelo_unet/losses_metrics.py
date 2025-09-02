#!/usr/bin/env python
# -*- coding: utf-8 -*-

# losses_metrics.py - Losses e metricas customizadas CORRIGIDAS
# Versao final sem acentos, todas as funcoes exportadas e funcionais

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

def focal_tversky_loss(alpha=0.3, beta=0.7, gamma=2.0, focal_weight=0.25):
    """
    Loss combinando Focal Loss e Tversky Loss
    Otimizada para classes desbalanceadas
    
    Args:
        alpha: peso para falsos negativos
        beta: peso para falsos positivos  
        gamma: parametro focal
        focal_weight: peso da componente focal
    
    Returns:
        funcao de loss compilavel
    """
    def loss_function(y_true, y_pred):
        # Clipping para estabilidade numerica
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        
        # Flatten tensors
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        
        # Calcular componentes Tversky
        TP = K.sum(y_true_f * y_pred_f)
        FN = K.sum(y_true_f * (1 - y_pred_f))
        FP = K.sum((1 - y_true_f) * y_pred_f)
        
        # Tversky Index
        tversky = (TP + eps) / (TP + alpha * FN + beta * FP + eps)
        tversky_loss = 1 - tversky
        
        # Componente Focal
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), focal_weight, 1 - focal_weight)
        focal = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t + eps)
        focal_mean = tf.reduce_mean(focal)
        
        # Combinar losses
        return 0.7 * tversky_loss + 0.3 * focal_mean
    
    return loss_function

def combined_loss(alpha=0.3, beta=0.7, gamma=3.0):
    """
    Loss combinada alternativa com boundary loss
    """
    def loss_function(y_true, y_pred):
        # Focal Tversky
        ft_loss = focal_tversky_loss(alpha, beta, gamma, 0.25)(y_true, y_pred)
        
        # Dice loss
        dice_loss = 1 - advanced_dice_coefficient(y_true, y_pred)
        
        # Binary crossentropy
        bce_loss = K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        
        return 0.5 * ft_loss + 0.3 * dice_loss + 0.2 * bce_loss
    
    return loss_function

def tversky_loss(alpha=0.3, beta=0.7):
    """
    Tversky Loss simples
    """
    def loss_function(y_true, y_pred):
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        
        TP = K.sum(y_true_f * y_pred_f)
        FN = K.sum(y_true_f * (1 - y_pred_f))
        FP = K.sum((1 - y_true_f) * y_pred_f)
        
        tversky = (TP + eps) / (TP + alpha * FN + beta * FP + eps)
        return 1 - tversky
    
    return loss_function

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss para desequilibrio de classes
    """
    def loss_function(y_true, y_pred):
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t + eps)
        
        return tf.reduce_mean(focal)
    
    return loss_function

def advanced_dice_coefficient(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient melhorado com smoothing
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Dice Loss (1 - Dice Coefficient)
    """
    return 1.0 - advanced_dice_coefficient(y_true, y_pred, smooth)

def iou_metric_stable(y_true, y_pred, smooth=1e-7):
    """
    IoU (Intersection over Union) metric estavel
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision_metric(y_true, y_pred, smooth=1e-7):
    """
    Precision metric
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    PP = tf.reduce_sum(y_pred_f)
    return (TP + smooth) / (PP + smooth)

def recall_metric(y_true, y_pred, smooth=1e-7):
    """
    Recall metric  
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    AP = tf.reduce_sum(y_true_f)
    return (TP + smooth) / (AP + smooth)

def boundary_iou(y_true, y_pred, smooth=1e-7):
    """
    IoU das bordas usando gradientes Sobel
    """
    def get_boundaries(tensor):
        # Sobel operators
        sobel_x = tf.constant([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=tf.float32)
        
        # Reshape for conv2d
        sobel_x = tf.reshape(sobel_x, [3,3,1,1])
        sobel_y = tf.reshape(sobel_y, [3,3,1,1])
        
        # Apply convolution
        gx = tf.nn.conv2d(tensor, sobel_x, strides=[1,1,1,1], padding='SAME')
        gy = tf.nn.conv2d(tensor, sobel_y, strides=[1,1,1,1], padding='SAME')
        
        # Magnitude
        magnitude = tf.sqrt(tf.square(gx) + tf.square(gy))
        return tf.cast(magnitude > 0.1, tf.float32)
    
    # Get boundaries
    true_boundaries = get_boundaries(y_true)
    pred_boundaries = get_boundaries(y_pred)
    
    # Calculate IoU of boundaries
    intersection = tf.reduce_sum(true_boundaries * pred_boundaries)
    union = tf.reduce_sum(true_boundaries) + tf.reduce_sum(pred_boundaries) - intersection
    
    return (intersection + smooth) / (union + smooth)

def f1_score(y_true, y_pred, smooth=1e-7):
    """
    F1 Score metric
    """
    precision = precision_metric(y_true, y_pred, smooth)
    recall = recall_metric(y_true, y_pred, smooth)
    return 2 * (precision * recall) / (precision + recall + smooth)

def sensitivity_metric(y_true, y_pred, smooth=1e-7):
    """
    Sensitivity (True Positive Rate) - same as recall
    """
    return recall_metric(y_true, y_pred, smooth)

def specificity_metric(y_true, y_pred, smooth=1e-7):
    """
    Specificity (True Negative Rate)
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    TN = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
    AN = tf.reduce_sum(1 - y_true_f)
    
    return (TN + smooth) / (AN + smooth)

def weighted_bce_loss(pos_weight=1.0):
    """
    Weighted Binary Cross Entropy
    """
    def loss_function(y_true, y_pred):
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        
        loss = -(pos_weight * y_true * tf.math.log(y_pred) + 
                (1 - y_true) * tf.math.log(1 - y_pred))
        
        return tf.reduce_mean(loss)
    
    return loss_function

def combo_loss(alpha=0.5, ce_ratio=0.5):
    """
    Combo Loss: Weighted combination of Dice Loss and Cross Entropy
    """
    def loss_function(y_true, y_pred):
        dice = dice_loss(y_true, y_pred)
        
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        ce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        
        return (ce_ratio * ce) + ((1 - ce_ratio) * dice)
    
    return loss_function

# Post-processing functions (numpy-based)
def postprocess_prediction(pred, threshold=0.5, min_area=50, min_aspect_ratio=2.0):
    """
    Post-processamento das predicoes usando OpenCV
    Remove componentes pequenos e com aspect ratio inadequado
    """
    try:
        import cv2
        
        # Convert to binary mask
        mask = (pred > threshold).astype('uint8')
        
        # Connected components analysis
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Filter by size and aspect ratio
        sizes = stats[1:, -1]  # Skip background component
        nb_components -= 1
        filtered = np.zeros(mask.shape, dtype=np.uint8)
        
        for i in range(nb_components):
            if sizes[i] >= min_area:
                # Get coordinates of this component
                component_mask = (output == i + 1)
                coords = np.column_stack(np.where(component_mask))
                
                if coords.shape[0] > 0:
                    # Calculate bounding box
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    
                    # Calculate aspect ratio
                    height = y_max - y_min + 1
                    width = x_max - x_min + 1
                    aspect_ratio = max(width, height) / max(1, min(width, height))
                    
                    # Keep component if it meets aspect ratio criteria
                    if aspect_ratio >= min_aspect_ratio:
                        filtered[component_mask] = 1
        
        return filtered.astype(np.float32)
        
    except ImportError:
        # Fallback without OpenCV
        return (pred > threshold).astype(np.float32)

def morphological_postprocess(pred, threshold=0.5, kernel_size=3):
    """
    Post-processamento morfologico simples
    """
    try:
        import cv2
        
        # Convert to binary
        binary = (pred > threshold).astype('uint8') * 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close small gaps
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Open to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return (opened / 255.0).astype(np.float32)
        
    except ImportError:
        # Fallback without OpenCV
        return (pred > threshold).astype(np.float32)

def adaptive_threshold_optimize(y_true, y_pred_probs, metric='dice'):
    """
    Encontra threshold otimo baseado em metrica especifica
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred_binary = (y_pred_probs > threshold).astype(np.float32)
        
        if metric == 'dice':
            score = np.mean([
                advanced_dice_coefficient(y_true[i:i+1], y_pred_binary[i:i+1]).numpy()
                for i in range(len(y_true))
            ])
        elif metric == 'iou':
            score = np.mean([
                iou_metric_stable(y_true[i:i+1], y_pred_binary[i:i+1]).numpy()
                for i in range(len(y_true))
            ])
        elif metric == 'f1':
            score = np.mean([
                f1_score(y_true[i:i+1], y_pred_binary[i:i+1]).numpy()
                for i in range(len(y_true))
            ])
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

# Alias para compatibilidade com codigo existente
post_process_predictions = postprocess_prediction

# Lista de todas as funcoes exportadas
__all__ = [
    'focal_tversky_loss',
    'combined_loss',
    'tversky_loss', 
    'focal_loss',
    'advanced_dice_coefficient',
    'dice_loss',
    'iou_metric_stable',
    'boundary_iou',
    'precision_metric',
    'recall_metric',
    'f1_score',
    'sensitivity_metric',
    'specificity_metric',
    'weighted_bce_loss',
    'combo_loss',
    'postprocess_prediction',
    'post_process_predictions',  # alias
    'morphological_postprocess',
    'adaptive_threshold_optimize'
]