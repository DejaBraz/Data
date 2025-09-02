#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para melhorar modelo com pos-processamento - VERSAO CORRIGIDA
Execute APOS treinar o modelo
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import json

# Configuracao basica para evitar problemas de imports
class SimpleConfig:
    def __init__(self):
        self.CACHE_PATH = Path('cache')
        self.FIGURES_PATH = Path('figuras')
        self.MODELS_PATH = Path('modelos')

# Importar bibliotecas para pos-processamento
try:
    from scipy import ndimage
    from skimage import morphology
    print("Bibliotecas de pos-processamento carregadas")
except ImportError as e:
    print(f"Erro ao importar bibliotecas: {e}")
    print("Execute: pip install scipy scikit-image")
    sys.exit(1)

def load_data_from_cache():
    """Carregar dados do cache mais recente"""
    cache_files = [
        'cache/processed_data_enhanced.npz',
        'cache/processed_data.npz'
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            print(f"Carregando dados de: {cache_file}")
            data = np.load(cache_file, allow_pickle=False)
            X = data['X']
            y = data['y']
            print(f"Dados carregados: X{X.shape}, y{y.shape}")
            return X, y
    
    raise FileNotFoundError("Nenhum cache de dados encontrado")

def aplicar_pos_processamento_morfologico(y_pred, config):
    """Aplica operacoes morfologicas para melhorar predicoes"""
    
    y_processed = np.zeros_like(y_pred)
    
    for i in range(y_pred.shape[0]):
        mask = y_pred[i].squeeze().astype(bool)
        
        # 1. Remover objetos pequenos
        if config.get('remove_small_objects', True):
            min_size = config.get('min_size', 10)
            mask = morphology.remove_small_objects(mask, min_size=min_size)
        
        # 2. Preencher buracos pequenos
        if config.get('fill_holes', True):
            area_threshold = config.get('area_threshold', 15)
            mask = morphology.remove_small_holes(mask, area_threshold=area_threshold)
        
        # 3. Operacao de closing (conectar regioes proximas)
        if config.get('closing', True):
            kernel_size = config.get('closing_kernel', 2)
            kernel = morphology.disk(kernel_size)
            mask = morphology.closing(mask, kernel)
        
        # 4. Operacao de opening (remover ruido)
        if config.get('opening', False):
            kernel_size = config.get('opening_kernel', 1)
            kernel = morphology.disk(kernel_size)
            mask = morphology.opening(mask, kernel)
        
        y_processed[i] = mask.astype(np.uint8)[..., np.newaxis]
    
    return y_processed

def aplicar_suavizacao_gaussiana(y_pred_probs, sigma=0.8):
    """Aplica suavizacao gaussiana nas probabilidades"""
    
    y_smooth = np.zeros_like(y_pred_probs)
    
    for i in range(y_pred_probs.shape[0]):
        y_smooth[i] = ndimage.gaussian_filter(
            y_pred_probs[i].squeeze(), sigma=sigma
        )[..., np.newaxis]
    
    return y_smooth

def melhorar_modelo():
    """Aplica melhorias no modelo usando pos-processamento"""
    
    print("="*60)
    print("MELHORANDO MODELO COM POS-PROCESSAMENTO")
    print("="*60)
    
    # Configuracao basica
    config = SimpleConfig()
    
    # Tentar carregar threshold otimo
    threshold_otimo = 0.3  # default
    try:
        with open('config_threshold_otimo.json', 'r') as f:
            config_threshold = json.load(f)
        threshold_otimo = config_threshold['threshold_otimo']
        print(f"Threshold otimo carregado: {threshold_otimo:.3f}")
    except FileNotFoundError:
        print(f"Usando threshold default: {threshold_otimo:.3f}")
    except Exception as e:
        print(f"Erro ao carregar configuracao: {e}")
    
    # Carregar modelo mais recente
    model_files = list(Path('modelos').glob('*.keras'))
    if not model_files:
        print("Nenhum modelo encontrado. Execute primeiro o treinamento.")
        return None
    
    # Usar o arquivo mais recente
    model_path = max(model_files, key=os.path.getctime)
    print(f"Carregando modelo: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Modelo carregado com sucesso")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None
    
    # Carregar dados
    try:
        X, y = load_data_from_cache()
    except FileNotFoundError:
        print("Cache de dados nao encontrado. Execute primeiro o treinamento.")
        return None
    
    # Verificar compatibilidade modelo-dados
    model_input_shape = model.input_shape
    data_shape = X.shape
    
    print(f"Forma de entrada do modelo: {model_input_shape}")
    print(f"Forma dos dados: {data_shape}")
    
    if model_input_shape[-1] != data_shape[-1]:
        print(f"INCOMPATIBILIDADE: Modelo espera {model_input_shape[-1]} canais, dados tem {data_shape[-1]} canais")
        
        if data_shape[-1] > model_input_shape[-1]:
            print(f"Usando apenas os primeiros {model_input_shape[-1]} canais dos dados")
            X = X[..., :model_input_shape[-1]]
        else:
            print("Erro: Modelo tem mais canais que os dados disponiveis")
            return None
    
    # Dados de teste
    n_test = int(0.15 * len(X))
    X_test = X[-n_test:]
    y_test = y[-n_test:]
    
    print(f"Dados de teste: {X_test.shape}")
    
    # Funcao para calcular metricas
    def calcular_metricas(y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        
        iou = intersection / (union + 1e-7)
        dice = 2 * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-7)
        
        tp = intersection
        fp = np.sum(y_pred) - intersection
        fn = np.sum(y_true) - intersection
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        return {
            'iou': iou, 'dice': dice, 'precision': precision, 
            'recall': recall, 'f1': f1
        }
    
    # Gerar predicoes
    print("\nGerando predicoes...")
    try:
        y_pred_probs = model.predict(X_test, batch_size=4, verbose=1)
    except Exception as e:
        print(f"Erro durante predicao: {e}")
        return None
    
    print("\n" + "="*50)
    print("TESTANDO DIFERENTES MELHORIAS")
    print("="*50)
    
    # 1. Baseline (threshold 0.5, sem pos-processamento)
    y_pred_baseline = (y_pred_probs > 0.5).astype(np.uint8)
    metrics_baseline = calcular_metricas(y_test, y_pred_baseline)
    
    # 2. Threshold otimizado
    y_pred_threshold = (y_pred_probs > threshold_otimo).astype(np.uint8)
    metrics_threshold = calcular_metricas(y_test, y_pred_threshold)
    
    # 3. Suavizacao + threshold otimizado
    y_pred_smooth = aplicar_suavizacao_gaussiana(y_pred_probs, sigma=0.8)
    y_pred_smooth_thresh = (y_pred_smooth > threshold_otimo).astype(np.uint8)
    metrics_smooth = calcular_metricas(y_test, y_pred_smooth_thresh)
    
    # 4. Testar diferentes configuracoes de pos-processamento
    configs_morph = [
        {
            'nome': 'Basico',
            'remove_small_objects': True, 'min_size': 5,
            'fill_holes': True, 'area_threshold': 10,
            'closing': True, 'closing_kernel': 2,
            'opening': False
        },
        {
            'nome': 'Medio',
            'remove_small_objects': True, 'min_size': 10,
            'fill_holes': True, 'area_threshold': 15,
            'closing': True, 'closing_kernel': 3,
            'opening': True, 'opening_kernel': 1
        },
        {
            'nome': 'Agressivo',
            'remove_small_objects': True, 'min_size': 20,
            'fill_holes': True, 'area_threshold': 25,
            'closing': True, 'closing_kernel': 4,
            'opening': True, 'opening_kernel': 2
        }
    ]
    
    melhor_config = None
    melhor_score = 0
    
    print(f"{'Metodo':<25} {'IoU':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 70)
    
    # Baseline
    print(f"{'1. Baseline (0.5)':<25} {metrics_baseline['iou']:<8.3f} {metrics_baseline['dice']:<8.3f} "
          f"{metrics_baseline['precision']:<10.3f} {metrics_baseline['recall']:<8.3f} {metrics_baseline['f1']:<8.3f}")
    
    # Threshold otimo
    print(f"{'2. Threshold otimo':<25} {metrics_threshold['iou']:<8.3f} {metrics_threshold['dice']:<8.3f} "
          f"{metrics_threshold['precision']:<10.3f} {metrics_threshold['recall']:<8.3f} {metrics_threshold['f1']:<8.3f}")
    
    # Suavizacao
    print(f"{'3. + Suavizacao':<25} {metrics_smooth['iou']:<8.3f} {metrics_smooth['dice']:<8.3f} "
          f"{metrics_smooth['precision']:<10.3f} {metrics_smooth['recall']:<8.3f} {metrics_smooth['f1']:<8.3f}")
    
    # Pos-processamento
    for i, config_morph in enumerate(configs_morph):
        y_pred_processed = aplicar_pos_processamento_morfologico(y_pred_smooth_thresh, config_morph)
        metrics_processed = calcular_metricas(y_test, y_pred_processed)
        
        nome = f"4.{i+1} + Morph {config_morph['nome']}"
        print(f"{nome:<25} {metrics_processed['iou']:<8.3f} {metrics_processed['dice']:<8.3f} "
              f"{metrics_processed['precision']:<10.3f} {metrics_processed['recall']:<8.3f} {metrics_processed['f1']:<8.3f}")
        
        # Criterio: IoU + F1
        score = 0.6 * metrics_processed['iou'] + 0.4 * metrics_processed['f1']
        if score > melhor_score:
            melhor_score = score
            melhor_config = config_morph
            melhor_metrics = metrics_processed
    
    if melhor_config is None:
        melhor_config = configs_morph[0]
        melhor_metrics = calcular_metricas(y_test, aplicar_pos_processamento_morfologico(y_pred_smooth_thresh, configs_morph[0]))
    
    print("\n" + "="*50)
    print("RESULTADO FINAL")
    print("="*50)
    
    print(f"Melhor configuracao: {melhor_config['nome']}")
    if metrics_baseline['iou'] > 0:
        print(f"IoU: {metrics_baseline['iou']:.3f} -> {melhor_metrics['iou']:.3f} ({((melhor_metrics['iou']/metrics_baseline['iou']-1)*100):+.1f}%)")
        print(f"Dice: {metrics_baseline['dice']:.3f} -> {melhor_metrics['dice']:.3f} ({((melhor_metrics['dice']/metrics_baseline['dice']-1)*100):+.1f}%)")
        print(f"F1: {metrics_baseline['f1']:.3f} -> {melhor_metrics['f1']:.3f} ({((melhor_metrics['f1']/metrics_baseline['f1']-1)*100):+.1f}%)")
    else:
        print(f"IoU: {melhor_metrics['iou']:.4f}")
        print(f"Dice: {melhor_metrics['dice']:.4f}")
        print(f"F1: {melhor_metrics['f1']:.4f}")
    
    # Gerar relatorio
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Salvar configuracao final otimizada
    config_final = {
        'threshold_otimo': threshold_otimo,
        'usar_suavizacao': True,
        'sigma_suavizacao': 0.8,
        'pos_processamento': melhor_config,
        'metricas_baseline': metrics_baseline,
        'metricas_melhoradas': melhor_metrics,
        'timestamp': timestamp,
        'model_path': str(model_path),
        'data_shape': str(X.shape),
        'model_input_shape': str(model_input_shape)
    }
    
    with open('configuracao_final_otimizada.json', 'w') as f:
        json.dump(config_final, f, indent=2)
    
    print(f"\nConfiguracao final salva em: configuracao_final_otimizada.json")
    
    # Verificar metas
    print("\n" + "="*50)
    print("VERIFICACAO DE METAS")
    print("="*50)
    
    metas_atingidas = 0
    if melhor_metrics['iou'] > 0.40:
        metas_atingidas += 1
        print("? Meta IoU > 0.40 ATINGIDA!")
    else:
        print(f"? Meta IoU > 0.40 nao atingida (atual: {melhor_metrics['iou']:.3f})")
    
    if melhor_metrics['dice'] > 0.60:
        metas_atingidas += 1
        print("? Meta Dice > 0.60 ATINGIDA!")
    else:
        print(f"? Meta Dice > 0.60 nao atingida (atual: {melhor_metrics['dice']:.3f})")
    
    if melhor_metrics['f1'] > 0.50:
        metas_atingidas += 1
        print("? Meta F1 > 0.50 ATINGIDA!")
    else:
        print(f"? Meta F1 > 0.50 nao atingida (atual: {melhor_metrics['f1']:.3f})")
    
    print(f"\nMetas atingidas: {metas_atingidas}/3")
    
    return {
        'config_final': config_final,
        'melhor_metrics': melhor_metrics,
        'baseline_metrics': metrics_baseline,
        'metas_atingidas': metas_atingidas
    }

if __name__ == "__main__":
    resultados = melhorar_modelo()
    
    if resultados:
        print("\n" + "="*50)
        print("RESUMO FINAL")
        print("="*50)
        print("Melhorias aplicadas com sucesso!")
        
        metas_atingidas = resultados['metas_atingidas']
        if metas_atingidas >= 3:
            print(f"?? PERFEITO! Todas as {metas_atingidas}/3 metas atingidas!")
        elif metas_atingidas >= 2:
            print(f"?? EXCELENTE! {metas_atingidas}/3 metas atingidas!")
        elif metas_atingidas == 1:
            print(f"?? BOM! {metas_atingidas}/3 metas atingidas!")
        else:
            print(f"?? Ainda ha espaco para melhorias. {metas_atingidas}/3 metas atingidas.")
            print("\nSugestoes para melhorar:")
            print("1. Ajustar hiperparametros do modelo")
            print("2. Aumentar quantidade de dados de treinamento")
            print("3. Experimentar diferentes arquiteturas")
            print("4. Aplicar mais tecnicas de data augmentation")
    else:
        print("\n? Erro ao aplicar melhorias. Verifique:")
        print("- Se o modelo foi treinado corretamente")
        print("- Se os dados estao no cache")
        print("- Se as bibliotecas necessarias estao instaladas")