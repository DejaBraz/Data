#!/usr/bin/env python
# -*- coding: utf-8 -*-

# main.py - Pipeline principal versao final
# Sem acentos, otimizado e robusto

import os
import sys
import logging
import gc
import time
import tensorflow as tf
from pathlib import Path

# Suprimir warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Imports relativos
from .config import FullDatasetConfig
from .data_processing import ChunkedDataProcessor  
from .trainer import FullDatasetTrainer
from .visualization import plot_comprehensive_results, visualize_predictions_detailed
from .utils import set_seeds, try_configure_gpus, log_memory

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger("frentes")

def check_system_requirements():
    """Verifica requisitos do sistema"""
    logger.info("=== VERIFICACAO DO SISTEMA ===")
    
    # Verificar TensorFlow
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Verificar GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPUs encontradas: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
    else:
        logger.warning("Nenhuma GPU encontrada - executando em CPU")
    
    # Verificar memoria
    log_memory("sistema")
    
    return len(gpus) > 0

def main():
    """Pipeline principal"""
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE DETECCAO DE FRENTES FRIAS")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Configuracao inicial
    config = FullDatasetConfig()
    config.ensure_dirs()
    
    # Setup reprodutibilidade
    set_seeds(config.RANDOM_SEED)
    
    # Setup GPU
    has_gpu = check_system_requirements()
    try_configure_gpus()
    
    # Log configuracao
    logger.info("=== CONFIGURACAO ===")
    logger.info(f"Total samples: {config.TOTAL_SAMPLES}")
    logger.info(f"Input size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(f"K-folds: {config.K_FOLDS}")
    logger.info(f"Augmentation: {config.USE_AUGMENTATION}")
    
    try:
        # Inicializar componentes
        processor = ChunkedDataProcessor(config)
        trainer = FullDatasetTrainer(config) 
        trainer.processor = processor
        
        # FASE 1: Processamento dos dados
        logger.info("=" * 60)
        logger.info("FASE 1: PROCESSAMENTO DOS DADOS")
        logger.info("=" * 60)
        
        log_memory("pre_processamento")
        X, y = processor.process_full_dataset()
        log_memory("pos_processamento")
        
        logger.info(f"Dados processados - X: {X.shape}, y: {y.shape}")
        logger.info(f"Pixels positivos: {y.mean()*100:.4f}%")
        
        # FASE 2: Treinamento
        logger.info("=" * 60)
        logger.info("FASE 2: TREINAMENTO")
        logger.info("=" * 60)
        
        if config.USE_KFOLD and config.K_FOLDS > 1:
            logger.info("Executando validacao cruzada...")
            fold_models, fold_histories = trainer.cross_validation_training(X, y)
            cv_results = trainer.evaluate_cross_validation(fold_histories)
        else:
            logger.info("Executando treino simples...")
            train_idx, val_idx, test_idx = trainer.prepare_data_splits(X, y)
            model, history = trainer.train_single_fold(X, y, train_idx, val_idx, 0)
            fold_models = [model]
            fold_histories = [history.history]
            cv_results = {}
        
        # FASE 3: Visualizacao
        logger.info("=" * 60)
        logger.info("FASE 3: VISUALIZACAO DOS RESULTADOS")  
        logger.info("=" * 60)
        
        if fold_histories and any(h for h in fold_histories):
            plot_comprehensive_results(fold_histories, config, cv_results)
            
            # Visualizar predicoes se temos modelo valido
            valid_models = [m for m in fold_models if m is not None]
            if valid_models:
                best_model = valid_models[0]
                visualize_predictions_detailed(best_model, X, y, config, n_samples=6)
                logger.info("Visualizacoes geradas com sucesso")
        else:
            logger.warning("Nenhum historico valido para visualizar")
        
        # Estatisticas finais
        total_time = (time.time() - start_time) / 60.0
        logger.info("=" * 80)
        logger.info("PIPELINE CONCLUIDO COM SUCESSO!")
        logger.info(f"Tempo total: {total_time:.1f} minutos")
        logger.info(f"Modelos gerados: {len([m for m in fold_models if m is not None])}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrompido pelo usuario")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Erro durante execucao do pipeline: {e}")
        logger.error("Verifique os logs acima para mais detalhes")
        raise
        
    finally:
        # Limpeza final
        logger.info("Realizando limpeza final...")
        try:
            if 'X' in locals(): del X
            if 'y' in locals(): del y  
            if 'fold_models' in locals(): 
                for model in fold_models:
                    if model is not None:
                        del model
            if 'fold_histories' in locals(): del fold_histories
        except:
            pass
            
        # Limpeza TensorFlow
        tf.keras.backend.clear_session()
        gc.collect()
        log_memory("limpeza_final")

if __name__ == "__main__":
    main()