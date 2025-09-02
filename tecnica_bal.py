#!/usr/bin/env python
# -*- coding: utf-8 -*-

# balancing_techniques.py
# Técnicas avançadas de balanceamento para detecção de frentes meteorológicas

import numpy as np
import logging
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import resize
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter
import gc

logger = logging.getLogger("frentes")

class AdvancedBalancingTechniques:
    """
    Classe com múltiplas técnicas de balanceamento para testar antes do treinamento
    """
    
    def __init__(self, config):
        self.config = config
        
    def analyze_dataset_distribution(self, y):
        """
        Analisa a distribuição do dataset para entender o desbalanceamento
        """
        logger.info("=== ANÁLISE DE DISTRIBUIÇÃO DO DATASET ===")
        
        # Calcula ratio de pixels positivos por amostra
        positive_ratios = np.array([mask.mean() for mask in y])
        
        # Estatísticas gerais
        total_samples = len(y)
        total_pixels = y.size
        positive_pixels = np.sum(y > 0.5)
        negative_pixels = total_pixels - positive_pixels
        
        logger.info(f"Total de amostras: {total_samples}")
        logger.info(f"Total de pixels: {total_pixels}")
        logger.info(f"Pixels positivos: {positive_pixels} ({positive_pixels/total_pixels*100:.2f}%)")
        logger.info(f"Pixels negativos: {negative_pixels} ({negative_pixels/total_pixels*100:.2f}%)")
        logger.info(f"Ratio desbalanceamento: 1:{negative_pixels/positive_pixels:.1f}")
        
        # Distribuição por amostras
        zero_samples = np.sum(positive_ratios == 0)
        low_samples = np.sum((positive_ratios > 0) & (positive_ratios <= 0.01))
        medium_samples = np.sum((positive_ratios > 0.01) & (positive_ratios <= 0.1))
        high_samples = np.sum(positive_ratios > 0.1)
        
        logger.info(f"\nDistribuição por amostras:")
        logger.info(f"Sem frentes (0%): {zero_samples} ({zero_samples/total_samples*100:.1f}%)")
        logger.info(f"Poucas frentes (0-1%): {low_samples} ({low_samples/total_samples*100:.1f}%)")
        logger.info(f"Médias frentes (1-10%): {medium_samples} ({medium_samples/total_samples*100:.1f}%)")
        logger.info(f"Muitas frentes (>10%): {high_samples} ({high_samples/total_samples*100:.1f}%)")
        
        return {
            'positive_ratios': positive_ratios,
            'total_samples': total_samples,
            'pixel_imbalance_ratio': negative_pixels/positive_pixels,
            'zero_samples': zero_samples,
            'low_samples': low_samples,
            'medium_samples': medium_samples,
            'high_samples': high_samples
        }
    
    def technique_1_smart_oversampling(self, X, y, target_ratio=0.3):
        """
        Oversampling inteligente baseado na intensidade das frentes
        """
        logger.info("=== TÉCNICA 1: OVERSAMPLING INTELIGENTE ===")
        
        positive_ratios = np.array([mask.mean() for mask in y])
        
        # Categoriza amostras por intensidade de frentes
        high_intensity = positive_ratios > 0.05  # >5% pixels com frente
        medium_intensity = (positive_ratios > 0.01) & (positive_ratios <= 0.05)
        low_intensity = (positive_ratios > 0) & (positive_ratios <= 0.01)
        no_fronts = positive_ratios == 0
        
        indices_high = np.where(high_intensity)[0]
        indices_medium = np.where(medium_intensity)[0]
        indices_low = np.where(low_intensity)[0]
        indices_none = np.where(no_fronts)[0]
        
        logger.info(f"Alta intensidade: {len(indices_high)}")
        logger.info(f"Média intensidade: {len(indices_medium)}")
        logger.info(f"Baixa intensidade: {len(indices_low)}")
        logger.info(f"Sem frentes: {len(indices_none)}")
        
        # Estratégia de oversampling diferenciada
        oversample_high = len(indices_high) * 8  # Multiplica por 8
        oversample_medium = len(indices_medium) * 4  # Multiplica por 4
        oversample_low = len(indices_low) * 2  # Multiplica por 2
        keep_none = int(len(indices_none) * 0.3)  # Mantém 30% das sem frente
        
        # Gera índices balanceados
        balanced_indices = []
        
        if len(indices_high) > 0:
            balanced_indices.extend(np.random.choice(indices_high, oversample_high, replace=True))
        if len(indices_medium) > 0:
            balanced_indices.extend(np.random.choice(indices_medium, oversample_medium, replace=True))
        if len(indices_low) > 0:
            balanced_indices.extend(np.random.choice(indices_low, oversample_low, replace=True))
        if len(indices_none) > 0:
            balanced_indices.extend(np.random.choice(indices_none, keep_none, replace=False))
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        logger.info(f"Dataset balanceado: {len(balanced_indices)} amostras")
        logger.info(f"Nova proporção de pixels positivos: {y_balanced.mean()*100:.2f}%")
        
        return X_balanced, y_balanced, balanced_indices
    
    def technique_2_focal_sampling(self, X, y, focus_threshold=0.02, alpha=2.0):
        """
        Sampling baseado em 'dificuldade' - foca em amostras com poucos pixels positivos
        """
        logger.info("=== TÉCNICA 2: FOCAL SAMPLING ===")
        
        positive_ratios = np.array([mask.mean() for mask in y])
        
        # Calcula "dificuldade" - amostras com poucos pixels positivos são mais difíceis
        difficulty_scores = np.where(
            positive_ratios > 0, 
            1.0 / (positive_ratios + 1e-6),  # Inverso da proporção
            0.1  # Baixa dificuldade para amostras sem frente
        )
        
        # Normaliza scores
        difficulty_scores = difficulty_scores / difficulty_scores.max()
        
        # Probabilidade de amostragem baseada na dificuldade
        sampling_probs = np.power(difficulty_scores, alpha)
        sampling_probs = sampling_probs / sampling_probs.sum()
        
        # Número de amostras para cada categoria
        n_hard_samples = int(len(X) * 0.6)  # 60% amostras difíceis
        n_easy_samples = int(len(X) * 0.4)   # 40% amostras fáceis
        
        # Amostragem com probabilidades
        hard_indices = np.random.choice(
            len(X), size=n_hard_samples, replace=True, p=sampling_probs
        )
        easy_indices = np.random.choice(
            len(X), size=n_easy_samples, replace=True
        )
        
        balanced_indices = np.concatenate([hard_indices, easy_indices])
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        logger.info(f"Dataset com focal sampling: {len(balanced_indices)} amostras")
        logger.info(f"Proporção de pixels positivos: {y_balanced.mean()*100:.2f}%")
        
        return X_balanced, y_balanced, balanced_indices
    
    def technique_3_augmentation_oversampling(self, X, y, target_samples=None):
        """
        Oversampling com augmentação para amostras positivas
        """
        logger.info("=== TÉCNICA 3: OVERSAMPLING COM AUGMENTAÇÃO ===")
        
        positive_ratios = np.array([mask.mean() for mask in y])
        positive_indices = np.where(positive_ratios > 0.001)[0]  # >0.1% pixels
        negative_indices = np.where(positive_ratios <= 0.001)[0]
        
        if target_samples is None:
            target_samples = len(X)
        
        # Proporção desejada: 40% positivas, 60% negativas
        n_positive_target = int(target_samples * 0.4)
        n_negative_target = int(target_samples * 0.6)
        
        # Oversample positivas com augmentação
        if len(positive_indices) > 0:
            positive_oversampled = np.random.choice(
                positive_indices, size=n_positive_target, replace=True
            )
            X_pos = X[positive_oversampled]
            y_pos = y[positive_oversampled]
            
            # Aplica augmentação em 70% das amostras positivas
            n_augment = int(n_positive_target * 0.7)
            for i in range(n_augment):
                X_pos[i], y_pos[i] = self._augment_meteorological_sample(X_pos[i], y_pos[i])
        else:
            X_pos, y_pos = np.array([]), np.array([])
            n_positive_target = 0
        
        # Sample negativas normalmente
        if len(negative_indices) > 0:
            negative_sampled = np.random.choice(
                negative_indices, size=n_negative_target, replace=True
            )
            X_neg = X[negative_sampled]
            y_neg = y[negative_sampled]
        else:
            X_neg, y_neg = X[negative_indices], y[negative_indices]
        
        # Combina
        if len(X_pos) > 0 and len(X_neg) > 0:
            X_balanced = np.concatenate([X_pos, X_neg])
            y_balanced = np.concatenate([y_pos, y_neg])
        elif len(X_pos) > 0:
            X_balanced, y_balanced = X_pos, y_pos
        else:
            X_balanced, y_balanced = X_neg, y_neg
        
        # Embaralha
        indices = np.arange(len(X_balanced))
        np.random.shuffle(indices)
        X_balanced = X_balanced[indices]
        y_balanced = y_balanced[indices]
        
        logger.info(f"Dataset com augmentação: {len(X_balanced)} amostras")
        logger.info(f"Amostras positivas: {n_positive_target}, Negativas: {n_negative_target}")
        logger.info(f"Proporção de pixels positivos: {y_balanced.mean()*100:.2f}%")
        
        return X_balanced, y_balanced, indices
    
    def technique_4_gradient_based_sampling(self, X, y):
        """
        Sampling baseado nos gradientes meteorológicos - foca em regiões com variações
        """
        logger.info("=== TÉCNICA 4: GRADIENT-BASED SAMPLING ===")
        
        gradient_scores = []
        
        # Calcula score baseado nos gradientes meteorológicos (canais 4-9 são derivadas)
        for i in range(len(X)):
            # Gradientes de temperatura e umidade (canais 4 e 5)
            temp_grad = X[i, :, :, 4]
            q_grad = X[i, :, :, 5]
            
            # Score baseado na intensidade dos gradientes
            gradient_intensity = np.mean(temp_grad**2 + q_grad**2)
            gradient_scores.append(gradient_intensity)
        
        gradient_scores = np.array(gradient_scores)
        
        # Normaliza scores
        gradient_scores = (gradient_scores - gradient_scores.min()) / (gradient_scores.max() - gradient_scores.min())
        
        positive_ratios = np.array([mask.mean() for mask in y])
        
        # Combina gradient score com presença de frentes
        combined_scores = gradient_scores * (1 + positive_ratios * 10)  # Boost para amostras com frentes
        
        # Sampling probabilístico
        sampling_probs = combined_scores / combined_scores.sum()
        
        n_samples = int(len(X) * 0.8)  # 80% do dataset original
        sampled_indices = np.random.choice(
            len(X), size=n_samples, replace=True, p=sampling_probs
        )
        
        X_balanced = X[sampled_indices]
        y_balanced = y[sampled_indices]
        
        logger.info(f"Dataset gradient-based: {len(sampled_indices)} amostras")
        logger.info(f"Score médio de gradiente: {np.mean(gradient_scores[sampled_indices]):.4f}")
        logger.info(f"Proporção de pixels positivos: {y_balanced.mean()*100:.2f}%")
        
        return X_balanced, y_balanced, sampled_indices
    
    def _augment_meteorological_sample(self, X, y):
        """
        Augmentação específica para dados meteorológicos
        """
        # Copia para não alterar originais
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Flip horizontal/vertical (50% chance cada)
        if np.random.random() > 0.5:
            X_aug = np.fliplr(X_aug)
            y_aug = np.fliplr(y_aug)
        if np.random.random() > 0.5:
            X_aug = np.flipud(X_aug)
            y_aug = np.flipud(y_aug)
        
        # Rotação (90, 180, 270 graus)
        if np.random.random() > 0.3:
            k = np.random.randint(1, 4)
            X_aug = np.rot90(X_aug, k)
            y_aug = np.rot90(y_aug, k)
        
        # Ruído gaussiano leve apenas nas variáveis originais
        if np.random.random() > 0.7:
            noise_std = 0.01
            noise = np.random.normal(0, noise_std, X_aug[..., :4].shape)
            X_aug[..., :4] = X_aug[..., :4] + noise
        
        return X_aug.astype(np.float32), y_aug.astype(np.float32)
    
    def evaluate_technique(self, X_original, y_original, X_balanced, y_balanced, technique_name):
        """
        Avalia uma técnica de balanceamento
        """
        logger.info(f"=== AVALIAÇÃO: {technique_name} ===")
        
        # Estatísticas originais
        orig_pos_ratio = y_original.mean() * 100
        orig_samples = len(y_original)
        
        # Estatísticas balanceadas
        bal_pos_ratio = y_balanced.mean() * 100
        bal_samples = len(y_balanced)
        
        # Distribuição por intensidade
        positive_ratios_orig = np.array([mask.mean() for mask in y_original])
        positive_ratios_bal = np.array([mask.mean() for mask in y_balanced])
        
        high_orig = np.sum(positive_ratios_orig > 0.05)
        high_bal = np.sum(positive_ratios_bal > 0.05)
        
        medium_orig = np.sum((positive_ratios_orig > 0.01) & (positive_ratios_orig <= 0.05))
        medium_bal = np.sum((positive_ratios_bal > 0.01) & (positive_ratios_bal <= 0.05))
        
        logger.info(f"Amostras: {orig_samples} ? {bal_samples} ({bal_samples/orig_samples:.1f}x)")
        logger.info(f"Pixels positivos: {orig_pos_ratio:.2f}% ? {bal_pos_ratio:.2f}%")
        logger.info(f"Alta intensidade: {high_orig} ? {high_bal}")
        logger.info(f"Média intensidade: {medium_orig} ? {medium_bal}")
        
        return {
            'technique': technique_name,
            'original_samples': orig_samples,
            'balanced_samples': bal_samples,
            'original_pos_ratio': orig_pos_ratio,
            'balanced_pos_ratio': bal_pos_ratio,
            'high_intensity_samples': high_bal,
            'medium_intensity_samples': medium_bal
        }
    
    def compare_all_techniques(self, X, y):
        """
        Compara todas as técnicas de balanceamento
        """
        logger.info("=== COMPARAÇÃO DE TODAS AS TÉCNICAS ===")
        
        # Análise inicial
        stats = self.analyze_dataset_distribution(y)
        
        results = {}
        
        # Técnica 1: Smart Oversampling
        try:
            X1, y1, idx1 = self.technique_1_smart_oversampling(X, y)
            results['smart_oversampling'] = self.evaluate_technique(X, y, X1, y1, "Smart Oversampling")
            del X1, y1, idx1
            gc.collect()
        except Exception as e:
            logger.error(f"Erro na Técnica 1: {e}")
        
        # Técnica 2: Focal Sampling
        try:
            X2, y2, idx2 = self.technique_2_focal_sampling(X, y)
            results['focal_sampling'] = self.evaluate_technique(X, y, X2, y2, "Focal Sampling")
            del X2, y2, idx2
            gc.collect()
        except Exception as e:
            logger.error(f"Erro na Técnica 2: {e}")
        
        # Técnica 3: Augmentation Oversampling
        try:
            X3, y3, idx3 = self.technique_3_augmentation_oversampling(X, y)
            results['augmentation_oversampling'] = self.evaluate_technique(X, y, X3, y3, "Augmentation Oversampling")
            del X3, y3, idx3
            gc.collect()
        except Exception as e:
            logger.error(f"Erro na Técnica 3: {e}")
        
        # Técnica 4: Gradient-based Sampling
        try:
            X4, y4, idx4 = self.technique_4_gradient_based_sampling(X, y)
            results['gradient_sampling'] = self.evaluate_technique(X, y, X4, y4, "Gradient-based Sampling")
            del X4, y4, idx4
            gc.collect()
        except Exception as e:
            logger.error(f"Erro na Técnica 4: {e}")
        
        return results
    
    def recommend_best_technique(self, comparison_results, target_positive_ratio=15.0):
        """
        Recomenda a melhor técnica baseada nos resultados
        """
        logger.info("=== RECOMENDAÇÃO DE TÉCNICA ===")
        
        best_score = -1
        best_technique = None
        
        for technique_name, result in comparison_results.items():
            # Score baseado em múltiplos critérios
            pos_ratio_score = min(result['balanced_pos_ratio'] / target_positive_ratio, 1.0)
            diversity_score = (result['high_intensity_samples'] + result['medium_intensity_samples']) / result['balanced_samples']
            size_penalty = max(0, 1.0 - result['balanced_samples'] / (len(comparison_results) * 50000))  # Penaliza datasets muito grandes
            
            total_score = pos_ratio_score * 0.5 + diversity_score * 0.3 + size_penalty * 0.2
            
            logger.info(f"{technique_name}: Score = {total_score:.3f}")
            logger.info(f"  - Pos ratio score: {pos_ratio_score:.3f}")
            logger.info(f"  - Diversity score: {diversity_score:.3f}")
            logger.info(f"  - Size penalty: {size_penalty:.3f}")
            
            if total_score > best_score:
                best_score = total_score
                best_technique = technique_name
        
        logger.info(f"\nMelhor técnica: {best_technique} (Score: {best_score:.3f})")
        return best_technique, best_score

# Função auxiliar para usar no seu código principal
def test_balancing_techniques(X, y, config):
    """
    Função principal para testar todas as técnicas de balanceamento
    """
    balancer = AdvancedBalancingTechniques(config)
    
    # Testa todas as técnicas
    comparison_results = balancer.compare_all_techniques(X, y)
    
    # Recomenda a melhor
    best_technique, score = balancer.recommend_best_technique(comparison_results)
    
    return comparison_results, best_technique, score