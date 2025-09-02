#!/usr/bin/env python
# -*- coding: utf-8 -*-

# data_processing.py - Processamento melhorado incorporando gradT_phys
# Baseado na analise estatistica que mostrou gradT_phys com AUC=0.805

import os
from pathlib import Path
import numpy as np
import h5py
from glob import glob
from skimage.transform import resize
import rasterio
import gc
from tensorflow.keras.utils import Sequence
from scipy.ndimage import convolve
from .utils import log_memory
import logging

logger = logging.getLogger("frentes")

class ChunkedDataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.CACHE_PATH)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_file(self):
        return self.cache_dir / 'processed_data_enhanced.npz'

    def check_cache(self):
        return self.cache_file().exists()

    def load_from_cache(self):
        logger.info("Carregando dados do cache melhorado...")
        data = np.load(self.cache_file(), allow_pickle=False)
        X = data['X']
        y = data['y']
        logger.info(f"Dados carregados do cache: X{X.shape}, y{y.shape}")
        return X, y

    def save_to_cache(self, X, y):
        logger.info("Salvando dados melhorados no cache...")
        np.savez_compressed(self.cache_file(), X=X, y=y)
        logger.info("Dados salvos no cache")

    def compute_gradient_temperature(self, T, lat, lon, smooth_iterations=10):
        """
        Computa gradiente de temperatura baseado na analise que mostrou AUC=0.805
        Usar o mesmo metodo que funcionou na analise estatistica
        """
        
        # Suavizacao com filtro 5-pontos como na analise
        kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 8.0
        T_smooth = T.copy()
        for _ in range(smooth_iterations):
            T_smooth = convolve(T_smooth, kernel, mode="nearest")
        
        # Calcular espacamento de grade (mesmo metodo da analise)
        R = 6371000.0  # metros
        deg_to_rad = np.pi / 180.0
        
        if len(lat) > 1 and len(lon) > 1:
            dy_deg = np.abs(lat[1] - lat[0])
            dx_deg = np.abs(lon[1] - lon[0])
        else:
            dy_deg = 0.25  # ERA5 tipico
            dx_deg = 0.25
        
        dy = dy_deg * deg_to_rad * R
        lat_mean = np.mean(lat)
        dx = dx_deg * deg_to_rad * R * np.cos(np.deg2rad(lat_mean))
        
        # Gradientes usando diferenca finita
        dT_dy = np.zeros_like(T_smooth)
        dT_dx = np.zeros_like(T_smooth)
        
        # Diferenca finita centrada
        dT_dy[1:-1, :] = (T_smooth[2:, :] - T_smooth[:-2, :]) / (2 * dy)
        dT_dx[:, 1:-1] = (T_smooth[:, 2:] - T_smooth[:, :-2]) / (2 * dx)
        
        # Bordas
        dT_dy[0, :] = (T_smooth[1, :] - T_smooth[0, :]) / dy
        dT_dy[-1, :] = (T_smooth[-1, :] - T_smooth[-2, :]) / dy
        dT_dx[:, 0] = (T_smooth[:, 1] - T_smooth[:, 0]) / dx
        dT_dx[:, -1] = (T_smooth[:, -1] - T_smooth[:, -2]) / dx
        
        # Magnitude do gradiente (essa e a feature que funcionou!)
        grad_mag = np.sqrt(dT_dx**2 + dT_dy**2)
        
        # Converter para unidades meteorologicas (K/100km)
        grad_mag_scaled = grad_mag * 100000
        
        return grad_mag_scaled

    def compute_wind_speed_and_direction(self, u, v):
        """
        Computa velocidade e direcao do vento
        Adiciona informacao fisica importante para frentes
        """
        # Velocidade do vento
        wind_speed = np.sqrt(u**2 + v**2)
        
        # Direcao do vento (em radianos)
        wind_dir = np.arctan2(v, u)
        
        return wind_speed, wind_dir

    def compute_divergence(self, u, v, lat, lon):
        """
        Computa divergencia do vento
        Importante para identificar frentes (convergencia/divergencia)
        """
        # Calcular espacamento de grade
        R = 6371000.0  # metros
        deg_to_rad = np.pi / 180.0
        
        if len(lat) > 1 and len(lon) > 1:
            dy_deg = np.abs(lat[1] - lat[0])
            dx_deg = np.abs(lon[1] - lon[0])
        else:
            dy_deg = 0.25  # ERA5 tipico
            dx_deg = 0.25
        
        dy = dy_deg * deg_to_rad * R
        lat_mean = np.mean(lat)
        dx = dx_deg * deg_to_rad * R * np.cos(np.deg2rad(lat_mean))
        
        # Gradientes de u e v
        du_dx = np.zeros_like(u)
        dv_dy = np.zeros_like(v)
        
        # Diferenca finita centrada
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
        
        # Bordas
        du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        dv_dy[0, :] = (v[1, :] - v[0, :]) / dy
        dv_dy[-1, :] = (v[-1, :] - v[-2, :]) / dy
        
        # Divergencia
        divergence = du_dx + dv_dy
        
        return divergence

    def process_h5_chunk_enhanced(self, h5_file, start_idx, end_idx, lat, lon):
        """Processa chunk com features melhoradas"""
        with h5py.File(h5_file, 'r') as hf:
            t_chunk = hf['t_levels'][start_idx:end_idx]
            u_chunk = hf['u_levels'][start_idx:end_idx]
            v_chunk = hf['v_levels'][start_idx:end_idx]
            q_chunk = hf['q_levels'][start_idx:end_idx]

            if len(t_chunk.shape) == 4:
                t_data = t_chunk[:, 0, :, :]  # nivel 850hPa
                u_data = u_chunk[:, 0, :, :]
                v_data = v_chunk[:, 0, :, :]
                q_data = q_chunk[:, 0, :, :]
            else:
                t_data = t_chunk
                u_data = u_chunk
                v_data = v_chunk
                q_data = q_chunk

        # Escolher features baseado na configuracao
        if self.config.USE_GRADIENT_FEATURES:
            # Usar gradT_phys que mostrou AUC=0.805
            features_data = np.zeros((t_data.shape[0], t_data.shape[1], t_data.shape[2], 4))
            
            for i in range(t_data.shape[0]):
                # Canal 0: temperatura
                features_data[i, :, :, 0] = t_data[i]
                
                # Canal 1: componente u do vento
                features_data[i, :, :, 1] = u_data[i]
                
                # Canal 2: componente v do vento
                features_data[i, :, :, 2] = v_data[i]
                
                # Canal 3: gradiente de temperatura (feature que funcionou!)
                features_data[i, :, :, 3] = self.compute_gradient_temperature(
                    t_data[i], lat, lon, 
                    smooth_iterations=self.config.GRADIENT_SMOOTH_ITERATIONS
                )
            
            logger.debug(f"Usando features: t, u, v, gradT_phys")
            
        elif self.config.USE_WIND_FEATURES:
            # Usar features de vento
            features_data = np.zeros((t_data.shape[0], t_data.shape[1], t_data.shape[2], 4))
            
            for i in range(t_data.shape[0]):
                wind_speed, wind_dir = self.compute_wind_speed_and_direction(u_data[i], v_data[i])
                
                features_data[i, :, :, 0] = t_data[i]  # temperatura
                features_data[i, :, :, 1] = wind_speed  # velocidade do vento
                features_data[i, :, :, 2] = wind_dir    # direcao do vento
                features_data[i, :, :, 3] = q_data[i]   # umidade
            
            logger.debug(f"Usando features: t, wind_speed, wind_dir, q")
            
        elif self.config.USE_DIVERGENCE_FEATURES:
            # Usar divergencia
            features_data = np.zeros((t_data.shape[0], t_data.shape[1], t_data.shape[2], 4))
            
            for i in range(t_data.shape[0]):
                divergence = self.compute_divergence(u_data[i], v_data[i], lat, lon)
                
                features_data[i, :, :, 0] = t_data[i]    # temperatura
                features_data[i, :, :, 1] = u_data[i]    # u
                features_data[i, :, :, 2] = v_data[i]    # v
                features_data[i, :, :, 3] = divergence   # divergencia
            
            logger.debug(f"Usando features: t, u, v, divergence")
            
        else:
            # Features originais
            features_data = np.zeros((t_data.shape[0], t_data.shape[1], t_data.shape[2], 4))
            for i in range(t_data.shape[0]):
                features_data[i, :, :, 0] = t_data[i]   # temperatura
                features_data[i, :, :, 1] = u_data[i]   # u
                features_data[i, :, :, 2] = v_data[i]   # v
                features_data[i, :, :, 3] = q_data[i]   # umidade
            
            logger.debug(f"Usando features originais: t, u, v, q")

        return features_data

    def process_mask_chunk(self, mask_files, start_idx, end_idx, target_shape):
        """Processamento de mascaras melhorado"""
        masks = []
        for i in range(start_idx, min(end_idx, len(mask_files))):
            try:
                with rasterio.open(mask_files[i]) as src:
                    mask = src.read(1)
                
                # Resize se necessario
                if mask.shape != target_shape:
                    mask_resized = resize(mask, target_shape, 
                                        anti_aliasing=True, preserve_range=True)
                else:
                    mask_resized = mask
                
                # Binarizacao robusta
                if np.any(mask_resized > 0):
                    threshold = np.percentile(mask_resized[mask_resized > 0], 50)
                else:
                    threshold = 0.5
                
                mask_binary = (mask_resized > threshold).astype(np.float32)
                masks.append(mask_binary)
                
            except Exception as e:
                logger.warning(f"Erro ao processar mascara {mask_files[i]}: {e}")
                masks.append(np.zeros(target_shape, dtype=np.float32))
        
        return np.array(masks, dtype=np.float32)

    def normalize_enhanced(self, data_chunk, stats=None):
        """Normalizacao melhorada usando estatisticas robustas"""
        if stats is None:
            stats = {}
            for ch in range(data_chunk.shape[-1]):
                ch_data = data_chunk[..., ch]
                # Usar percentis para normalizacao robusta
                p25 = np.percentile(ch_data, 25)
                p75 = np.percentile(ch_data, 75)
                median = np.percentile(ch_data, 50)
                iqr = p75 - p25
                stats[ch] = {
                    'median': median, 
                    'iqr': max(iqr, 1e-6),
                    'p25': p25, 
                    'p75': p75
                }
        
        normalized = data_chunk.copy().astype(np.float32)
        for ch in range(data_chunk.shape[-1]):
            median = stats[ch]['median']
            iqr = stats[ch]['iqr']
            # Normalizacao robusta: (x - median) / (IQR / 1.35)
            normalized[..., ch] = (data_chunk[..., ch] - median) / (iqr / 1.35)
            
        return normalized, stats

    def process_full_dataset_enhanced(self):
        """Processamento completo com melhorias"""
        logger.info("=== PROCESSAMENTO MELHORADO DO DATASET ===")
        log_memory("inicio processamento melhorado")

        if self.config.CACHE_PROCESSED and self.check_cache():
            return self.load_from_cache()

        if not os.path.exists(self.config.H5_FILE_PATH):
            raise FileNotFoundError(f"HDF5 nao encontrado: {self.config.H5_FILE_PATH}")

        # Carregar masks
        mask_files = sorted(glob(os.path.join(self.config.MASK_DIR, self.config.MASK_PATTERN)))
        if len(mask_files) == 0:
            raise FileNotFoundError("Nenhum arquivo de mascara encontrado")

        # Obter dimensoes
        with h5py.File(self.config.H5_FILE_PATH, 'r') as hf:
            actual_samples = hf['t_levels'].shape[0]
            lat = hf['lat'][:]
            lon = hf['lon'][:]

        total_samples = min(actual_samples, len(mask_files), self.config.TOTAL_SAMPLES)
        
        # Log do tipo de features sendo usadas
        feature_type = "ORIGINAL (t,u,v,q)"
        if self.config.USE_GRADIENT_FEATURES:
            feature_type = "GRADIENT (t,u,v,gradT_phys) - AUC=0.805"
        elif self.config.USE_WIND_FEATURES:
            feature_type = "WIND (t,wind_speed,wind_dir,q)"
        elif self.config.USE_DIVERGENCE_FEATURES:
            feature_type = "DIVERGENCE (t,u,v,divergence)"
        
        logger.info(f"Processando {total_samples} amostras")
        logger.info(f"Tipo de features: {feature_type}")

        all_X = []
        all_y = []
        normalization_stats = None

        chunk_size = self.config.CHUNK_SIZE
        n_chunks = (total_samples + chunk_size - 1) // chunk_size
        target_shape = (self.config.INPUT_SIZE, self.config.INPUT_SIZE)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            current_chunk_size = end_idx - start_idx
            
            logger.info(f"Chunk {chunk_idx+1}/{n_chunks}: amostras {start_idx}-{end_idx-1}")
            log_memory(f"chunk_{chunk_idx}_start")

            # Processar dados H5 com features melhoradas
            features_data = self.process_h5_chunk_enhanced(
                self.config.H5_FILE_PATH, start_idx, end_idx, lat, lon
            )
            
            # Resize para target shape
            chunk_data = np.zeros((current_chunk_size, target_shape[0], target_shape[1], 
                                 self.config.N_CHANNELS), dtype=np.float32)

            for i in range(current_chunk_size):
                for ch in range(self.config.N_CHANNELS):
                    chunk_data[i, :, :, ch] = resize(features_data[i, :, :, ch], target_shape, 
                                                   anti_aliasing=True, preserve_range=True)

            # Normalizacao melhorada
            if chunk_idx == 0:
                chunk_data_norm, normalization_stats = self.normalize_enhanced(chunk_data)
                logger.info("Estatisticas de normalizacao robusta calculadas")
                # Log das estatisticas para debug
                for ch, stats in normalization_stats.items():
                    logger.info(f"Canal {ch}: mediana={stats['median']:.6f}, "
                               f"IQR={stats['iqr']:.6f}")
            else:
                chunk_data_norm, _ = self.normalize_enhanced(chunk_data, normalization_stats)

            # Processar mascaras
            chunk_masks = self.process_mask_chunk(mask_files, start_idx, end_idx, target_shape)
            chunk_masks = chunk_masks[..., np.newaxis]

            all_X.append(chunk_data_norm)
            all_y.append(chunk_masks)

            # Limpeza
            del features_data, chunk_data, chunk_data_norm, chunk_masks
            gc.collect()
            log_memory(f"chunk_{chunk_idx}_end")

        # Concatenar todos os chunks
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        del all_X, all_y
        gc.collect()
        log_memory("concatenate_end")

        logger.info(f"Processamento melhorado concluido: X{X.shape}, y{y.shape}")
        logger.info(f"Positivos: {y.mean()*100:.4f}%")
        logger.info(f"Features: {feature_type}")

        if self.config.CACHE_PROCESSED:
            self.save_to_cache(X, y)

        return X, y

    def process_full_dataset(self):
        """Metodo principal - usa versao melhorada se configurado"""
        # Sempre usar versao melhorada se disponivel
        if hasattr(self.config, 'USE_GRADIENT_FEATURES') or \
           hasattr(self.config, 'USE_WIND_FEATURES') or \
           hasattr(self.config, 'USE_DIVERGENCE_FEATURES'):
            return self.process_full_dataset_enhanced()
        else:
            return self._process_original()
    
    def _process_original(self):
        """Versao original simplificada para compatibilidade"""
        logger.info("=== PROCESSAMENTO ORIGINAL ===")
        
        # Usar cache original se existir
        original_cache = self.cache_dir / 'processed_data.npz'
        if self.config.CACHE_PROCESSED and original_cache.exists():
            logger.info("Carregando dados do cache original...")
            data = np.load(original_cache, allow_pickle=False)
            return data['X'], data['y']
        
        # Implementacao basica rapida
        with h5py.File(self.config.H5_FILE_PATH, 'r') as hf:
            n_samples = min(hf['t_levels'].shape[0], self.config.TOTAL_SAMPLES)
            t_data = hf['t_levels'][:n_samples, 0, :, :]
            u_data = hf['u_levels'][:n_samples, 0, :, :]
            v_data = hf['v_levels'][:n_samples, 0, :, :]
            q_data = hf['q_levels'][:n_samples, 0, :, :]
        
        # Resize para tamanho alvo
        target_shape = (self.config.INPUT_SIZE, self.config.INPUT_SIZE)
        X = np.zeros((n_samples, target_shape[0], target_shape[1], 4), dtype=np.float32)
        
        for i in range(n_samples):
            X[i, :, :, 0] = resize(t_data[i], target_shape, preserve_range=True)
            X[i, :, :, 1] = resize(u_data[i], target_shape, preserve_range=True)
            X[i, :, :, 2] = resize(v_data[i], target_shape, preserve_range=True)
            X[i, :, :, 3] = resize(q_data[i], target_shape, preserve_range=True)
        
        # Mascaras basicas
        mask_files = sorted(glob(os.path.join(self.config.MASK_DIR, self.config.MASK_PATTERN)))
        y = []
        for i in range(min(len(mask_files), n_samples)):
            with rasterio.open(mask_files[i]) as src:
                mask = src.read(1)
            mask_resized = resize(mask, target_shape, preserve_range=True)
            y.append((mask_resized > 0.5).astype(np.float32))
        
        y = np.array(y)[..., np.newaxis]
        
        logger.info(f"Processamento original: X{X.shape}, y{y.shape}")
        return X, y


class LargeDatasetGenerator(Sequence):
    """Generator melhorado com augmentation meteorologico"""
    
    def __init__(self, X_indices, y_data, X_full_data, batch_size=4, shuffle=True, 
                 augment=False, config=None, **kwargs):
        super().__init__(**kwargs)
        self.X_indices = np.array(list(X_indices))
        self.y_data = y_data
        self.X_full_data = X_full_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.config = config
        self.indices = np.arange(len(self.X_indices))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X_indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X_indices))
        batch_idx = self.indices[start:end]
        current_batch_size = end - start

        X_batch = np.zeros((current_batch_size, self.config.INPUT_SIZE, 
                           self.config.INPUT_SIZE, self.config.N_CHANNELS), dtype=np.float32)
        y_batch = np.zeros((current_batch_size, self.config.INPUT_SIZE, 
                           self.config.INPUT_SIZE, 1), dtype=np.float32)

        for i, bi in enumerate(batch_idx):
            data_idx = self.X_indices[bi]
            X_batch[i] = self.X_full_data[data_idx]
            y_batch[i] = self.y_data[data_idx]
            
            # Augmentation meteorologico melhorado
            if self.augment and np.random.random() < self.config.AUGMENTATION_PROB:
                X_batch[i], y_batch[i] = self._augment_meteorological(X_batch[i], y_batch[i])

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment_meteorological(self, X, y):
        """Augmentation especifico para dados meteorologicos"""
        
        # Determinar tipo de features para augmentation correto
        use_gradient = getattr(self.config, 'USE_GRADIENT_FEATURES', False)
        use_wind = getattr(self.config, 'USE_WIND_FEATURES', False)
        use_divergence = getattr(self.config, 'USE_DIVERGENCE_FEATURES', False)
        
        # Flips horizontais
        if np.random.random() > 0.5:
            X = np.fliplr(X)
            y = np.fliplr(y)
            
            if use_gradient:
                # Canal 1 = u, precisa inverter sinal
                X[:, :, 1] = -X[:, :, 1]
            elif use_wind:
                # Canal 2 = wind_dir, precisa ajustar
                X[:, :, 2] = np.pi - X[:, :, 2]  # inverter direcao
            elif use_divergence:
                # Canal 1 = u, precisa inverter
                X[:, :, 1] = -X[:, :, 1]
            else:
                # Original: canal 1 = u
                X[:, :, 1] = -X[:, :, 1]
        
        # Flips verticais
        if np.random.random() > 0.5:
            X = np.flipud(X)
            y = np.flipud(y)
            
            if use_gradient:
                # Canal 2 = v, precisa inverter sinal
                X[:, :, 2] = -X[:, :, 2]
            elif use_wind:
                # Canal 2 = wind_dir, precisa ajustar
                X[:, :, 2] = -X[:, :, 2]  # inverter direcao
            elif use_divergence:
                # Canal 2 = v, precisa inverter
                X[:, :, 2] = -X[:, :, 2]
            else:
                # Original: canal 2 = v
                X[:, :, 2] = -X[:, :, 2]
        
        # Rotacoes de 90 graus
        if np.random.random() > 0.7:
            k = np.random.randint(1, 4)
            X = np.rot90(X, k)
            y = np.rot90(y, k)
            
            # Rotacionar componentes do vento para features que usam u,v
            if (use_gradient or use_divergence or not any([use_gradient, use_wind, use_divergence])):
                if k % 2 == 1:  # Rotacoes impares (90, 270 graus)
                    u_temp = X[:, :, 1].copy()
                    X[:, :, 1] = -X[:, :, 2] if k == 1 else X[:, :, 2]
                    X[:, :, 2] = u_temp if k == 1 else -u_temp
            elif use_wind:
                # Para wind features, ajustar direcao
                X[:, :, 2] = X[:, :, 2] + k * np.pi / 2  # adicionar rotacao
                X[:, :, 2] = np.mod(X[:, :, 2], 2 * np.pi)  # manter em [0, 2pi]
        
        # Ruido gaussiano leve
        if np.random.random() > 0.8:
            noise_std = 0.01
            # Sempre adicionar ruido na temperatura (canal 0)
            X[:, :, 0] += np.random.normal(0, noise_std, X[:, :, 0].shape)
            
            # Adicionar ruido em features derivadas
            if use_gradient:
                # Canal 3 = gradT, pode ter ruido
                X[:, :, 3] += np.random.normal(0, noise_std * 0.5, X[:, :, 3].shape)
            elif use_divergence:
                # Canal 3 = divergence, pode ter ruido
                X[:, :, 3] += np.random.normal(0, noise_std * 0.5, X[:, :, 3].shape)
        
        # Deformacao elastica leve (importante para frentes)
        if np.random.random() > 0.9:
            X, y = self._elastic_deformation(X, y, alpha=2.0, sigma=0.5)
        
        return X.astype(np.float32), y.astype(np.float32)

    def _elastic_deformation(self, X, y, alpha=1.0, sigma=1.0):
        """Deformacao elastica simples"""
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates
            
            shape = X.shape[:2]
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
            
            x, y_coord = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y_coord + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            X_def = np.zeros_like(X)
            for ch in range(X.shape[-1]):
                X_def[:, :, ch] = map_coordinates(X[:, :, ch], indices, order=1).reshape(shape)
            
            y_def = map_coordinates(y[:, :, 0], indices, order=1).reshape(shape)[:, :, np.newaxis]
            
            return X_def, y_def
        except ImportError:
            # Fallback sem scipy
            return X, y