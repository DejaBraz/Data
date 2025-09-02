#!/usr/bin/env python
# -*- coding: utf-8 -*-

# trainer.py - Trainer final com gerenciamento robusto de memoria
# Sem acentos, otimizado para estabilidade

import os
import numpy as np
import pickle
import time
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import logging

from .model_unet import build_large_unet
from .losses_metrics import focal_tversky_loss, advanced_dice_coefficient, iou_metric_stable, boundary_iou, precision_metric, recall_metric
from .data_processing import LargeDatasetGenerator
from .utils import log_memory

logger = logging.getLogger("frentes")

class FullDatasetTrainer:
    def __init__(self, config):
        self.config = config
        self.processor = None
        self.config.ensure_dirs()

    def prepare_data_splits(self, X, y):
        logger.info("Preparando divisoes estratificadas")
        positive_ratios = np.array([mask.mean() for mask in y])
        ratio_bins = np.percentile(positive_ratios, [0, 33, 66, 100])
        stratify_labels = np.digitize(positive_ratios, ratio_bins[1:-1])
        
        train_val_idx, test_idx = train_test_split(
            range(len(X)), 
            test_size=self.config.TEST_SPLIT, 
            random_state=self.config.RANDOM_SEED, 
            stratify=stratify_labels
        )
        
        train_idx, val_idx = train_test_split(
            train_val_idx, 
            test_size=self.config.VAL_SPLIT / (1 - self.config.TEST_SPLIT), 
            random_state=self.config.RANDOM_SEED, 
            stratify=stratify_labels[train_val_idx]
        )
        
        logger.info(f"Split - Treino: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
        return train_idx, val_idx, test_idx

    def create_callbacks(self, fold_num=0):
        callbacks_list = []
        
        # Early stopping otimizado
        callbacks_list.append(
            callbacks.EarlyStopping(
                monitor='val_advanced_dice_coefficient',
                patience=self.config.PATIENCE,
                mode='max',
                restore_best_weights=True,
                verbose=1,
                min_delta=0.0005
            )
        )
        
        # Reduce LR mais conservador
        callbacks_list.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=6,
                min_lr=1e-7,
                verbose=1,
                cooldown=2,
                mode='min'
            )
        )
        
        # Model checkpoint
        model_path = os.path.join(self.config.MODELS_PATH, f'best_model_fold_{fold_num}.keras')
        callbacks_list.append(
            callbacks.ModelCheckpoint(
                model_path,
                monitor='val_advanced_dice_coefficient',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        )
        
        # CSV logger
        log_path = os.path.join(self.config.MODELS_PATH, f'training_log_fold_{fold_num}.csv')
        callbacks_list.append(
            callbacks.CSVLogger(log_path, append=False)
        )
        
        return callbacks_list

    def aggressive_memory_cleanup(self):
        """Limpeza agressiva de memoria GPU e sistema"""
        try:
            # Tensorflow cleanup
            K.clear_session()
            tf.keras.backend.clear_session()
            
            # Force garbage collection multiplo
            for _ in range(3):
                gc.collect()
            
            # GPU reset se disponivel
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.reset_memory_growth(gpu)
                        tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.warning(f"GPU reset failed: {e}")
            
            # Aguarda estabilizacao
            time.sleep(2)
            logger.info("Limpeza de memoria concluida")
            
        except Exception as e:
            logger.warning(f"Erro na limpeza de memoria: {e}")

    def train_single_fold(self, X, y, train_idx, val_idx, fold_num=0):
        logger.info(f"=== INICIANDO FOLD {fold_num+1} ===")
        log_memory(f"fold_{fold_num}_inicio")
        
        # Limpeza pre-treino
        self.aggressive_memory_cleanup()
        
        # Build model
        model = build_large_unet(input_shape=X.shape[1:], config=self.config)
        
        # Class distribution info
        train_masks = y[train_idx]
        pos_pixels = np.sum(train_masks == 1)
        neg_pixels = np.sum(train_masks == 0)
        pos_ratio = pos_pixels / (pos_pixels + neg_pixels) if (pos_pixels + neg_pixels) > 0 else 0
        
        logger.info(f"Distribuicao classes - Positivos: {pos_ratio*100:.3f}%")
        logger.info(f"Pixels - Pos: {pos_pixels:,} | Neg: {neg_pixels:,}")
        
        # Compile with optimized settings
        optimizer = Adam(
            learning_rate=self.config.LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=focal_tversky_loss(alpha=0.1, beta=0.9, gamma=3.0, focal_weight=0.8),
            metrics=[
                'accuracy',
                advanced_dice_coefficient,
                iou_metric_stable,
                boundary_iou,
                precision_metric,
                recall_metric
            ]
        )
        
        # Data generators
        train_gen = LargeDatasetGenerator(
            train_idx, y, X,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            augment=self.config.USE_AUGMENTATION,
            config=self.config
        )
        
        val_gen = LargeDatasetGenerator(
            val_idx, y, X,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            augment=False,
            config=self.config
        )
        
        logger.info(f"Generators - Train: {len(train_gen)} batches | Val: {len(val_gen)} batches")
        
        # Callbacks
        callbacks_list = self.create_callbacks(fold_num)
        
        # Training with error handling
        start_time = time.time()
        try:
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=self.config.EPOCHS,
                callbacks=callbacks_list,
                verbose=1
            )
            
            training_time = (time.time() - start_time) / 60.0
            logger.info(f"Treino fold {fold_num+1} concluido em {training_time:.1f} minutos")
            
        except Exception as e:
            logger.error(f"Erro durante treino fold {fold_num+1}: {e}")
            # Cleanup em caso de erro
            del train_gen, val_gen
            try:
                del model
            except:
                pass
            self.aggressive_memory_cleanup()
            raise e
        
        # Cleanup pos-treino
        del train_gen, val_gen
        gc.collect()
        log_memory(f"fold_{fold_num}_fim")
        
        return model, history

    def cross_validation_training(self, X, y):
        logger.info(f"Iniciando validacao cruzada com {self.config.K_FOLDS} folds")
        
        positive_ratios = np.array([mask.mean() for mask in y])
        ratio_bins = np.percentile(positive_ratios, [0, 33, 66, 100])
        stratify_labels = np.digitize(positive_ratios, ratio_bins[1:-1])
        
        kfold = StratifiedKFold(
            n_splits=self.config.K_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )

        fold_histories = []
        fold_models = []
        successful_folds = 0

        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X, stratify_labels)):
            logger.info("=" * 80)
            logger.info(f"FOLD {fold_num+1}/{self.config.K_FOLDS}")
            
            try:
                model, history = self.train_single_fold(X, y, train_idx, val_idx, fold_num)
                fold_histories.append(history.history)
                fold_models.append(model)
                successful_folds += 1
                
                # Salva historico
                history_path = os.path.join(self.config.MODELS_PATH, f'fold_{fold_num}_history.pkl')
                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)
                    
            except Exception as e:
                logger.error(f"Fold {fold_num+1} falhou: {e}")
                # Adiciona historico vazio para manter consistencia
                fold_histories.append({})
                fold_models.append(None)
                
                # Tenta recuperar memoria
                self.aggressive_memory_cleanup()
                time.sleep(5)
            
            # Limpeza entre folds
            if self.config.FORCE_GC:
                try:
                    if 'model' in locals() and model is not None:
                        del model
                except:
                    pass
                self.aggressive_memory_cleanup()

        # Filtra resultados validos
        valid_histories = [h for h in fold_histories if h]
        valid_models = [m for m in fold_models if m is not None]
        
        logger.info(f"Cross-validation concluido: {successful_folds}/{self.config.K_FOLDS} folds bem-sucedidos")
        return valid_models, valid_histories

    def evaluate_cross_validation(self, fold_histories):
        logger.info("Avaliando resultados da validacao cruzada")
        
        metrics = [
            'val_loss',
            'val_advanced_dice_coefficient', 
            'val_iou_metric_stable',
            'val_boundary_iou',
            'val_precision_metric',
            'val_recall_metric'
        ]
        
        results = {}
        for metric in metrics:
            values = []
            for history in fold_histories:
                if metric in history and len(history[metric]) > 0:
                    if 'loss' in metric:
                        values.append(min(history[metric]))
                    else:
                        values.append(max(history[metric]))
            
            if values:
                results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
                logger.info(f"{metric}: {results[metric]['mean']:.4f} +/- {results[metric]['std']:.4f}")
        
        return results