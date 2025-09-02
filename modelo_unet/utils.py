#!/usr/bin/env python
# -*- coding: utf-8 -*-

#utils.py
#Funcoes utilitarias: logging de memoria, seed, pequenas ferramentas.


import os
import psutil
import numpy as np
import tensorflow as tf
import gc
import logging

logger = logging.getLogger("frentes")
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def log_memory(step=""):
    mem = get_memory_usage_gb()
    logger.info(f"Memoria ({step}): {mem:.2f} GB")

def set_seeds(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

def try_configure_gpus():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            logger.info(f"{len(gpus)} GPU(s) configurada(s) com crescimento dinamico")
        except RuntimeError as e:
            logger.warning(f"Erro na configuracao da GPU: {e}")
    else:
        logger.info("Nenhuma GPU detectada; executando em CPU")