#!/usr/bin/env python
# -*- coding: utf-8 -*-

# config.py - Configuracao melhorada baseada no paper Niebler et al.
# Otimizada para deteccao de frentes com dataset desbalanceado

from pathlib import Path

class FullDatasetConfig:
    # Paths principais
    BASE_DIR = Path('.')
    H5_FILE_PATH = BASE_DIR / 'Era5' / 'Treino' / 'ERA5_FULL.h5'
    MASK_DIR = BASE_DIR / 'Era5' / 'Treino' / 'rotulo_era5'
    MASK_PATTERN = 'superficie_20*.tif'

    FIGURES_PATH = BASE_DIR / 'figuras'
    MODELS_PATH = BASE_DIR / 'modelos'
    CACHE_PATH = BASE_DIR / 'cache'

    # Dataset parameters baseados na analise estatistica
    TOTAL_SAMPLES = 2912
    INPUT_SIZE = 128
    N_CHANNELS = 4  # t, u, v, gradT_phys (substituindo q por gradT)
    
    # Training parameters otimizados para frentes
    BATCH_SIZE = 4          # Reduzido devido ao dataset desbalanceado
    EPOCHS = 150            # Aumentado para convergencia melhor
    LEARNING_RATE = 1e-4    # Taxa mais conservadora para estabilidade
    PATIENCE = 20           # Early stopping mais paciente
    
    # Memory management
    CHUNK_SIZE = 100
    USE_MEMMAP = True
    CACHE_PROCESSED = True
    
    # Regularization melhorada
    DROPOUT_RATE = 0.3      # Aumentado devido ao overfitting comum
    L2_REGULARIZATION = 1e-4
    USE_BATCH_NORM = True   # Ativado para estabilidade
    
    # Data splits otimizados
    TRAIN_SPLIT = 0.70      # Reduzido para mais dados de validacao
    VAL_SPLIT = 0.20        # Aumentado para melhor validacao
    TEST_SPLIT = 0.10
    
    # Loss function parameters baseados no paper
    USE_ENHANCED_LOSS = True
    LOSS_TYPE = 'combined_niebler'  # IoU + BCE como no paper
    IoU_WEIGHT = 0.8               # Peso maior para IoU
    BCE_WEIGHT = 0.2               # Peso menor para BCE
    
    # Class balancing para dataset desbalanceado  
    USE_CLASS_WEIGHTS = True
    POSITIVE_WEIGHT = 10.0          # Peso maior para frentes (classe minoritaria)
    
    # Data augmentation meteorologico
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.4         # Aumentado devido ao dataset pequeno
    AUGMENTATION_TYPES = [
        'horizontal_flip',
        'vertical_flip', 
        'rotation_90',
        'gaussian_noise',
        'elastic_deform'            # Importante para dados meteorologicos
    ]
    
    # Post-processing baseado no paper
    USE_POST_PROCESSING = True
    POST_PROCESS_THRESHOLD = 0.45   # Threshold otimizado
    MIN_COMPONENT_AREA = 15         # Remove componentes pequenos
    USE_MORPHOLOGICAL_OPS = True
    
    # Model architecture melhorada
    MODEL_DEPTH = 4                 # Profundidade da U-Net
    INITIAL_FILTERS = 32            # Filtros iniciais
    FILTER_MULTIPLIER = 2           # Multiplicador por nivel
    USE_SKIP_CONNECTIONS = True
    USE_ATTENTION = False           # Desabilitado por enquanto
    
    # Training strategies do paper
    USE_PROGRESSIVE_TRAINING = False # Para implementacao futura
    USE_MULTI_SCALE_LOSS = False    # Para implementacao futura
    
    # Validation e metrics
    VALIDATION_FREQUENCY = 5        # Validar a cada 5 epochs
    SAVE_BEST_ONLY = True
    MONITOR_METRIC = 'val_iou_metric_stable'
    
    # Reproducibilidade
    RANDOM_SEED = 42
    
    # Thresholds otimizados baseados na analise
    DETECTION_THRESHOLD = 0.45      # Baseado em analise estatistica
    NMS_THRESHOLD = 0.5             # Non-maximum suppression
    
    # Feature engineering baseado na analise estatistica
    USE_GRADIENT_FEATURES = True    # gradT_phys mostrou AUC=0.805
    GRADIENT_SMOOTH_ITERATIONS = 10 # Suavizacao do gradiente
    
    # Performance monitoring
    LOG_MEMORY_USAGE = True
    PROFILE_TRAINING = False
    
    # Advanced training techniques
    USE_COSINE_ANNEALING = False    # Para implementacao futura
    USE_WARM_RESTART = False        # Para implementacao futura
    
    # Cross-validation
    USE_KFOLD = True
    K_FOLDS = 3                     # Reduzido devido ao dataset pequeno
    
    # Memory constraints
    CLEAR_SESSION = True
    FORCE_GC = True
    MAX_MEMORY_GB = 12
    
    def ensure_dirs(self):
        # Cria diretorios necessarios
        self.FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        self.MODELS_PATH.mkdir(parents=True, exist_ok=True)
        self.CACHE_PATH.mkdir(parents=True, exist_ok=True)
    
    def get_loss_function(self):
        # Retorna funcao de loss baseada na configuracao
        if self.LOSS_TYPE == 'combined_niebler':
            from losses_metrics import combined_loss_niebler
            return lambda y_true, y_pred: combined_loss_niebler(
                y_true, y_pred, 
                alpha=self.BCE_WEIGHT, 
                beta=self.IoU_WEIGHT
            )
        elif self.LOSS_TYPE == 'focal':
            from losses_metrics import focal_loss
            return focal_loss
        else:
            return 'binary_crossentropy'
    
    def get_metrics(self):
        # Lista de metricas baseadas no paper
        from losses_metrics import (
            enhanced_dice_coefficient,
            iou_metric_stable, 
            precision_metric,
            recall_metric,
            f1_score
        )
        
        return [
            'accuracy',
            enhanced_dice_coefficient,
            iou_metric_stable,
            precision_metric,
            recall_metric,
            f1_score
        ]
    
    def get_callbacks(self, fold=0):
        # Callbacks otimizados
        from tensorflow.keras.callbacks import (
            EarlyStopping, 
            ReduceLROnPlateau, 
            ModelCheckpoint,
            CSVLogger
        )
        
        callbacks = []
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor=self.MONITOR_METRIC,
            patience=self.PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ))
        
        # Model checkpoint
        model_path = self.MODELS_PATH / f'best_model_fold_{fold}.keras'
        callbacks.append(ModelCheckpoint(
            model_path,
            monitor=self.MONITOR_METRIC,
            save_best_only=self.SAVE_BEST_ONLY,
            mode='max',
            verbose=1
        ))
        
        # CSV logger
        log_path = self.MODELS_PATH / f'training_log_fold_{fold}.csv'
        callbacks.append(CSVLogger(log_path))
        
        return callbacks
    
    def print_config(self):
        # Imprime configuracao atual
        print("=" * 60)
        print("CONFIGURACAO DO MODELO - Melhorada")
        print("=" * 60)
        print(f"Dataset: {self.TOTAL_SAMPLES} amostras")
        print(f"Input size: {self.INPUT_SIZE}x{self.INPUT_SIZE}x{self.N_CHANNELS}")
        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Epochs: {self.EPOCHS}")
        print(f"Learning rate: {self.LEARNING_RATE}")
        print(f"Loss function: {self.LOSS_TYPE}")
        print(f"Use class weights: {self.USE_CLASS_WEIGHTS}")
        print(f"Positive weight: {self.POSITIVE_WEIGHT}")
        print(f"Post-processing: {self.USE_POST_PROCESSING}")
        print(f"Gradient features: {self.USE_GRADIENT_FEATURES}")
        print(f"K-folds: {self.K_FOLDS}")
        print("=" * 60)