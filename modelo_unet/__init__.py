#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pacote modelo_unet para deteccao de frentes frias
"""

# Importar as classes principais para facilitar o acesso
from .config import FullDatasetConfig
from .data_processing import ChunkedDataProcessor, LargeDatasetGenerator
from .trainer import FullDatasetTrainer
from .model_unet import build_large_unet
from .losses_metrics import (
    advanced_dice_coefficient,
    iou_metric_stable,
    precision_metric,
    recall_metric,
    post_process_predictions
)
from .visualization import plot_comprehensive_results, visualize_predictions_detailed
from .utils import set_seeds, try_configure_gpus, log_memory, get_memory_usage_gb

__version__ = "1.0.0"
__author__ = "Dejanira F. Braz"

# Lista de modulos exportados
__all__ = [
    'FullDatasetConfig',
    'ChunkedDataProcessor',
    'LargeDatasetGenerator',
    'FullDatasetTrainer',
    'build_large_unet',
    'advanced_dice_coefficient',
    'iou_metric_stable',
    'precision_metric',
    'recall_metric',
    'post_process_predictions',
    'plot_comprehensive_results',
    'visualize_predictions_detailed',
    'set_seeds',
    'try_configure_gpus',
    'log_memory',
    'get_memory_usage_gb'
]