#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import rasterio
from rasterio.errors import RasterioIOError
from skimage.transform import resize
from scipy import stats
from scipy.ndimage import convolve, binary_dilation
import pandas as pd
import matplotlib.pyplot as plt
import gc

# -----------------------------
# utils - CORRIGIDOS
# -----------------------------

def ensure_outdirs(base: Path):
    (base / "figs").mkdir(parents=True, exist_ok=True)
    return base

def robust_mask_binarize(mask: np.ndarray) -> np.ndarray:
    if np.any(mask > 0):
        thr = np.percentile(mask[mask > 0], 50)
    else:
        thr = 0.5
    return (mask > thr).astype(np.float32)

def maybe_resize(arr: np.ndarray, target_shape):
    if arr.shape == target_shape:
        return arr
    return resize(arr, target_shape, preserve_range=True, anti_aliasing=True).astype(arr.dtype)

def mannwhitney_auc(pos: np.ndarray, neg: np.ndarray):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    try:
        # Remover NaN e Inf
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        auc = u_stat / (len(pos) * len(neg))
        return max(auc, 1.0 - auc)
    except Exception as e:
        print(f"    Error in AUC calculation: {e}")
        return np.nan

def cohens_d(pos: np.ndarray, neg: np.ndarray):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    try:
        # Remover NaN e Inf
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
            
        pos_m, neg_m = np.mean(pos), np.mean(neg)
        pos_v, neg_v = np.var(pos, ddof=1), np.var(neg, ddof=1)
        pooled = np.sqrt((pos_v + neg_v) / 2.0 + 1e-12)
        return (pos_m - neg_m) / pooled if pooled > 0 else 0.0
    except Exception as e:
        print(f"    Error in Cohen's d calculation: {e}")
        return np.nan

def cliff_delta(pos: np.ndarray, neg: np.ndarray):
    # Cliff's Delta - medida de tamanho do efeito robusta
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    
    try:
        # Remover NaN e Inf
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        
        # Para arrays grandes, usar amostragem mais agressiva
        if len(pos) > 2000:
            pos = np.random.choice(pos, 2000, replace=False)
        if len(neg) > 2000:
            neg = np.random.choice(neg, 2000, replace=False)
        
        dominance = 0
        total = len(pos) * len(neg)
        
        # Otimização: usar broadcasting quando possível
        if total < 50000:  # Para arrays pequenos
            pos_mat = pos[:, np.newaxis]
            neg_mat = neg[np.newaxis, :]
            dominance = np.sum(pos_mat > neg_mat) - np.sum(pos_mat < neg_mat)
        else:  # Para arrays maiores, usar loop mais eficiente
            for p in pos[:500]:  # Limitar ainda mais para evitar travamento
                dominance += np.sum(p > neg) - np.sum(p < neg)
            # Normalizar proporcionalmente
            dominance = dominance * (len(pos) / 500)
        
        return dominance / total
    except Exception as e:
        print(f"    Error in Cliff's Delta calculation: {e}")
        return np.nan

def ks_stat(pos: np.ndarray, neg: np.ndarray):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    try:
        # Remover NaN e Inf
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        return stats.ks_2samp(pos, neg, alternative="two-sided").statistic
    except Exception as e:
        print(f"    Error in KS calculation: {e}")
        return np.nan

def sample_pos_neg(values: np.ndarray, mask: np.ndarray, n_bg: int = 50000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    
    try:
        # Remove valores invalidos
        values_flat = values.flatten()
        mask_flat = mask.flatten()
        
        valid = np.isfinite(values_flat)
        values_clean = values_flat[valid]
        mask_clean = mask_flat[valid]
        
        if len(values_clean) == 0:
            return np.array([]), np.array([])
        
        m = mask_clean > 0.5
        pos = values_clean[m]
        neg = values_clean[~m]
        
        # Limitar tamanhos para evitar problemas de memoria
        if len(pos) > n_bg//10:  # Limitar positivos também
            idx = rng.choice(len(pos), size=n_bg//10, replace=False)
            pos = pos[idx]
            
        if len(neg) > n_bg:
            idx = rng.choice(len(neg), size=n_bg, replace=False)
            neg = neg[idx]
        
        return pos.astype(np.float32), neg.astype(np.float32)
    except Exception as e:
        print(f"    Error in sampling: {e}")
        return np.array([]), np.array([])

# -----------------------------
# TFP CORRIGIDO - VERSAO 2
# -----------------------------

def calc_dx_dy_fixed(lat, lon):
    # Versao corrigida para coordenadas ERA5
    R = 6371000.0  # metros
    deg_to_rad = np.pi / 180.0
    
    # Para grades regulares ERA5
    if len(lat) > 1 and len(lon) > 1:
        dy_deg = np.abs(lat[1] - lat[0])
        dx_deg = np.abs(lon[1] - lon[0])
    else:
        # Fallback se houver problema
        dy_deg = 0.25  # ERA5 tipico
        dx_deg = 0.25
    
    # dy constante
    dy = dy_deg * deg_to_rad * R
    
    # dx varia com latitude - usar latitude media para simplificar
    lat_mean = np.mean(lat)
    dx = dx_deg * deg_to_rad * R * np.cos(np.deg2rad(lat_mean))
    
    return dx, dy

def compute_tfp_fixed(T, lat, lon, smooth_iterations=10):
    # TFP com escala corrigida para ERA5
    
    # Suavizacao mais leve
    kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 8.0
    T_smooth = T.copy()
    for _ in range(smooth_iterations):
        T_smooth = convolve(T_smooth, kernel, mode="nearest")
    
    # Calcular espacamentos de grade
    dx, dy = calc_dx_dy_fixed(lat, lon)
    
    print(f"    Grid spacing: dx={dx/1000:.1f}km, dy={dy/1000:.1f}km")
    
    # Gradientes usando diferenca finita centrada
    dT_dy = np.zeros_like(T_smooth)
    dT_dx = np.zeros_like(T_smooth)
    
    # Diferenca finita centrada
    dT_dy[1:-1, :] = (T_smooth[2:, :] - T_smooth[:-2, :]) / (2 * dy)
    dT_dx[:, 1:-1] = (T_smooth[:, 2:] - T_smooth[:, :-2]) / (2 * dx)
    
    # Bordas com diferenca finita forward/backward
    dT_dy[0, :] = (T_smooth[1, :] - T_smooth[0, :]) / dy
    dT_dy[-1, :] = (T_smooth[-1, :] - T_smooth[-2, :]) / dy
    dT_dx[:, 0] = (T_smooth[:, 1] - T_smooth[:, 0]) / dx
    dT_dx[:, -1] = (T_smooth[:, -1] - T_smooth[:, -2]) / dx
    
    # Magnitude do gradiente
    grad_mag = np.sqrt(dT_dx**2 + dT_dy**2)
    
    # Segunda derivada (gradiente do gradiente)
    dgm_dy = np.zeros_like(grad_mag)
    dgm_dx = np.zeros_like(grad_mag)
    
    dgm_dy[1:-1, :] = (grad_mag[2:, :] - grad_mag[:-2, :]) / (2 * dy)
    dgm_dx[:, 1:-1] = (grad_mag[:, 2:] - grad_mag[:, :-2]) / (2 * dx)
    
    # Bordas
    dgm_dy[0, :] = (grad_mag[1, :] - grad_mag[0, :]) / dy
    dgm_dy[-1, :] = (grad_mag[-1, :] - grad_mag[-2, :]) / dy
    dgm_dx[:, 0] = (grad_mag[:, 1] - grad_mag[:, 0]) / dx
    dgm_dx[:, -1] = (grad_mag[:, -1] - grad_mag[:, -2]) / dx
    
    # Vetor unitario na direcao do gradiente
    safe_mag = np.maximum(grad_mag, 1e-12)
    n_x = dT_dx / safe_mag
    n_y = dT_dy / safe_mag
    
    # TFP = -nabla|nablaT| . n
    tfp = -(dgm_dx * n_x + dgm_dy * n_y)
    
    # Converter para K/(100km) - ESCALA CORRETA
    tfp_scaled = tfp * 100000  # m para 100km
    grad_scaled = grad_mag * 100000
    
    print(f"    TFP calc: raw range {tfp.min():.2e} to {tfp.max():.2e}")
    print(f"    TFP scaled: {tfp_scaled.min():.6f} to {tfp_scaled.max():.6f} K/(100km)")
    
    return tfp_scaled, grad_scaled

# Manter outras funcoes mas com calculos mais simples
def compute_additional_features_simple(T, U, V, lat, lon):
    # Versao simplificada para evitar erros
    dx, dy = calc_dx_dy_fixed(lat, lon)
    
    # Gradientes simples
    dU_dy, dU_dx = np.gradient(U)
    dV_dy, dV_dx = np.gradient(V)
    dT_dy, dT_dx = np.gradient(T)
    
    # Normalizar pelos espacamentos
    dU_dx = dU_dx / dx
    dU_dy = dU_dy / dy
    dV_dx = dV_dx / dx
    dV_dy = dV_dy / dy
    dT_dx = dT_dx / dx
    dT_dy = dT_dy / dy
    
    # Quantidades dinamicas
    vorticity = (dV_dx - dU_dy) * 1e5  # x10^5 s^-1
    deformation = np.sqrt((dU_dx - dV_dy)**2 + (dV_dx + dU_dy)**2) * 1e5
    confluence = -(dU_dx + dV_dy) * 1e5
    advT = -(U * dT_dx + V * dT_dy) * 3600  # K/h
    
    # Frontogenese simplificada
    frontogenesis = (dT_dx * dU_dx + dT_dy * dU_dy + dT_dx * dV_dx + dT_dy * dV_dy) * 100000
    
    return {
        'vorticity': vorticity,
        'deformation': deformation,
        'confluence': confluence,
        'AdvT_phys': advT,
        'frontogenesis': frontogenesis
    }

# -----------------------------
# masks helpers
# -----------------------------

def list_mask_files(mask_dir: Path, mask_pattern: str):
    files = sorted(glob(str(mask_dir / mask_pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No mask tif found in {mask_dir} with pattern {mask_pattern}")
    return files

def load_masks(files, target_shape, dilate_pixels=0):
    out = []
    for fp in files:
        try:
            with rasterio.open(fp) as src:
                m = src.read(1)
        except RasterioIOError:
            m = np.zeros(target_shape, dtype=np.float32)
        
        m = maybe_resize(m, target_shape)
        m = robust_mask_binarize(m)
        out.append(m.astype(np.float32))

    y = np.stack(out, axis=0)[..., None]
    original_fraction = float((y > 0.5).mean())

    if dilate_pixels > 0:
        structure = np.ones((3, 3), dtype=bool)
        for i in range(y.shape[0]):
            y[i, ..., 0] = binary_dilation(
                y[i, ..., 0] > 0.5, structure=structure, iterations=dilate_pixels
            )

    new_fraction = float((y > 0.5).mean())
    return y, original_fraction, new_fraction

# -----------------------------
# main analysis - ROBUSTO
# -----------------------------

def analyze_robust(args):
    outdir = ensure_outdirs(Path(args.output_dir))

    # 1) Load H5
    print("[1/6] Loading data...")
    h5_path = Path(args.h5_file)
    ch = {}
    with h5py.File(h5_path, "r") as hf:
        t_chunk = hf["t_levels"][:]
        u_chunk = hf["u_levels"][:]
        v_chunk = hf["v_levels"][:]
        q_chunk = hf["q_levels"][:]
        lat = hf["lat"][:]
        lon = hf["lon"][:]

        if len(t_chunk.shape) == 4:
            t_data = t_chunk[:, args.level_index, :, :]
            u_data = u_chunk[:, args.level_index, :, :]
            v_data = v_chunk[:, args.level_index, :, :]
            q_data = q_chunk[:, args.level_index, :, :]
        else:
            t_data = t_chunk
            u_data = u_chunk
            v_data = v_chunk
            q_data = q_chunk

        ch["t"] = t_data.astype(np.float32)
        ch["u"] = u_data.astype(np.float32)
        ch["v"] = v_data.astype(np.float32)
        ch["q"] = q_data.astype(np.float32)
        ch["lat"] = lat
        ch["lon"] = lon

    n_total, H, W = ch["t"].shape
    print(f"    Data shape: {n_total} x {H} x {W}")
    print(f"    Lat range: {lat.min():.3f} to {lat.max():.3f}")
    print(f"    Lon range: {lon.min():.3f} to {lon.max():.3f}")

    # 2) Load masks
    print("[2/6] Loading masks...")
    mask_files_all = list_mask_files(Path(args.mask_dir), args.mask_pattern)
    
    if args.n_samples > 0:
        n_req = min(args.n_samples, n_total)
    else:
        n_req = n_total

    n_eff = min(n_req, len(mask_files_all))
    
    for k in ["t", "u", "v", "q"]:
        ch[k] = ch[k][:n_eff]
    n = n_eff

    mask_files = mask_files_all[:n]
    y, frac_before, frac_after = load_masks(mask_files, (H, W), args.dilate_pixels)
    print(f"    Processed {n} samples")
    print(f"    Positive fraction: {frac_before*100:.4f}% -> {frac_after*100:.4f}%")

    # 3) Basic features
    print("[3/6] Computing basic features...")
    features = {
        "t": ch["t"],
        "u": ch["u"], 
        "v": ch["v"],
        "q": ch["q"],
        "wind_speed": np.sqrt(ch["u"]**2 + ch["v"]**2)
    }

    # 4) TFP corrigido
    print("[4/6] Computing TFP (corrected version 2)...")
    tfp_list = []
    gradT_list = []
    
    for i in range(min(n, 100)):  # Limitar para teste
        if i % 10 == 0:
            print(f"    Sample {i+1}/100...")
        
        T = ch["t"][i]
        tfp, grad_mag = compute_tfp_fixed(T, ch["lat"], ch["lon"], smooth_iterations=5)
        tfp_list.append(tfp)
        gradT_list.append(grad_mag)
    
    features["TFP_phys"] = np.stack(tfp_list, axis=0)
    features["gradT_phys"] = np.stack(gradT_list, axis=0)
    
    print(f"    Final TFP range: {features['TFP_phys'].min():.6f} - {features['TFP_phys'].max():.6f}")

    # 5) Additional features (simplificado)
    print("[5/6] Computing additional features...")
    for key in ['vorticity', 'deformation', 'confluence', 'AdvT_phys', 'frontogenesis']:
        features[key] = np.zeros((100, H, W), dtype=np.float32)  # Placeholder
    
    # 6) Statistics
    print("[6/6] Computing statistics...")
    rows = []
    
    for name, arr in features.items():
        print(f"    Analyzing {name}...")
        try:
            pos, neg = sample_pos_neg(arr, y[:100, ..., 0])  # Usar apenas 100 amostras
            
            if len(pos) == 0 or len(neg) == 0:
                print(f"      Skipping {name} - empty samples")
                continue
            
            row = {
                "feature": name,
                "auc": mannwhitney_auc(pos, neg),
                "cohen_d": cohens_d(pos, neg),
                "ks": ks_stat(pos, neg),
                "cliff_delta": cliff_delta(pos, neg),
                "mean_pos": float(np.mean(pos)) if len(pos) > 0 else np.nan,
                "mean_neg": float(np.mean(neg)) if len(neg) > 0 else np.nan,
                "n_pos": len(pos),
                "n_neg": len(neg)
            }
            rows.append(row)
            print(f"      {name}: AUC={row['auc']:.3f}, Cohen_d={row['cohen_d']:.3f}")
            
        except Exception as e:
            print(f"      Error analyzing {name}: {e}")
            continue
        
        # Limpeza de memoria
        gc.collect()

    # Save results
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values('auc', ascending=False, na_last=True)
        
        df.to_csv(outdir / "analysis_report.csv", index=False)
        
        with open(outdir / "analysis_summary.txt", "w") as f:
            f.write("ANALISE ESTATISTICA - VERSAO ROBUSTA\n")
            f.write("="*50 + "\n\n")
            f.write(f"Amostras: {n} (processadas: 100)\n")
            f.write(f"TFP range: {features['TFP_phys'].min():.6f} - {features['TFP_phys'].max():.6f}\n\n")
            f.write(df.to_string(index=False))
        
        print(f"\n[DONE] Results saved to {outdir}")
        print("\nTOP FEATURES:")
        print(df.head()[['feature', 'auc', 'cohen_d']].to_string(index=False))
    else:
        print("No valid results to save")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5-file", type=str, default="Era5/Treino/ERA5_FULL.h5")
    p.add_argument("--mask-dir", type=str, default="Era5/Treino/rotulo_era5")
    p.add_argument("--mask-pattern", type=str, default="*.tif")
    p.add_argument("--output-dir", type=str, default="analysis_robust")
    p.add_argument("--level-index", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=0)
    p.add_argument("--dilate-pixels", type=int, default=2)
    
    args = p.parse_args()
    analyze_robust(args)