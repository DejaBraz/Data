#!/usr/bin/env python
# -*- coding: utf-8 -*-

#visualization.py
#Plota graficos e predicoes detalhadas.


import os
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger("frentes")

def plot_comprehensive_results(fold_histories, config, cv_results):
    logger.info("Gerando visualizacoes abrangentes...")
    n_folds = len(fold_histories)
    fig = plt.figure(figsize=(20, 15))

    # Loss
    plt.subplot(3,3,1)
    for i, h in enumerate(fold_histories):
        if 'loss' in h: plt.plot(h['loss'], alpha=0.7, label=f'Fold {i+1} Train')
        if 'val_loss' in h: plt.plot(h['val_loss'], '--', alpha=0.7, label=f'Fold {i+1} Val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    # Dice
    plt.subplot(3,3,2)
    for i, h in enumerate(fold_histories):
        if 'advanced_dice_coefficient' in h: plt.plot(h['advanced_dice_coefficient'], alpha=0.7)
        if 'val_advanced_dice_coefficient' in h: plt.plot(h['val_advanced_dice_coefficient'], '--', alpha=0.7)
    plt.title('Dice'); plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.grid(True)

    # IoU
    plt.subplot(3,3,3)
    for i,h in enumerate(fold_histories):
        if 'iou_metric_stable' in h: plt.plot(h['iou_metric_stable'], alpha=0.7)
        if 'val_iou_metric_stable' in h: plt.plot(h['val_iou_metric_stable'], '--', alpha=0.7)
    plt.title('IoU'); plt.grid(True)

    # Boundary IoU
    plt.subplot(3,3,4)
    for i,h in enumerate(fold_histories):
        if 'boundary_iou' in h: plt.plot(h['boundary_iou'], alpha=0.7)
        if 'val_boundary_iou' in h: plt.plot(h['val_boundary_iou'], '--', alpha=0.7)
    plt.title('Boundary IoU'); plt.grid(True)

    # Precision
    plt.subplot(3,3,5)
    for i,h in enumerate(fold_histories):
        if 'precision_metric' in h: plt.plot(h['precision_metric'], alpha=0.7)
        if 'val_precision_metric' in h: plt.plot(h['val_precision_metric'], '--', alpha=0.7)
    plt.title('Precision'); plt.grid(True)

    # Recall
    plt.subplot(3,3,6)
    for i,h in enumerate(fold_histories):
        if 'recall_metric' in h: plt.plot(h['recall_metric'], alpha=0.7)
        if 'val_recall_metric' in h: plt.plot(h['val_recall_metric'], '--', alpha=0.7)
    plt.title('Recall'); plt.grid(True)

    # Boxplot
    plt.subplot(3,3,7)
    metrics_names = ['Dice', 'IoU', 'Boundary IoU', 'Precision', 'Recall']
    keys = ['val_advanced_dice_coefficient','val_iou_metric_stable','val_boundary_iou','val_precision_metric','val_recall_metric']
    box = []
    for k in keys:
        if k in cv_results:
            box.append(cv_results[k]['values'])
        else:
            box.append([])
    plt.boxplot(box, labels=metrics_names)
    plt.xticks(rotation=45)
    plt.title('Cross-Validation Results')

    # LR (if present)
    plt.subplot(3,3,8)
    for h in fold_histories:
        if 'lr' in h:
            plt.plot(h['lr'])
    plt.yscale('log')
    plt.title('Learning rate schedule')

    # Summary
    plt.subplot(3,3,9)
    plt.axis('off')
    summary_text = f"""
    Dataset: {config.TOTAL_SAMPLES}
    Folds: {n_folds}
    Epochs: {config.EPOCHS}
    Batch size: {config.BATCH_SIZE}

    Dice: {cv_results.get('val_advanced_dice_coefficient',{}).get('mean',0):.4f} +/- {cv_results.get('val_advanced_dice_coefficient',{}).get('std',0):.4f}
    IoU: {cv_results.get('val_iou_metric_stable',{}).get('mean',0):.4f} +/- {cv_results.get('val_iou_metric_stable',{}).get('std',0):.4f}
    Precision: {cv_results.get('val_precision_metric',{}).get('mean',0):.4f} +/- {cv_results.get('val_precision_metric',{}).get('std',0):.4f}
    Recall: {cv_results.get('val_recall_metric',{}).get('mean',0):.4f} +/- {cv_results.get('val_recall_metric',{}).get('std',0):.4f}
    """
    plt.text(0.01, 0.99, summary_text, fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()
    out = os.path.join(config.FIGURES_PATH, 'comprehensive_results.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Figura salva em {out}")

def visualize_predictions_detailed(model, X, y, config, n_samples=6):
    logger.info(f"Visualizando {n_samples} predicoes detalhadas")
    positive_ratios = [mask.mean() for mask in y]
    n_third = max(1, n_samples // 3)
    low_idx = np.argsort(positive_ratios)[:n_third]
    mid_idx = np.argsort(positive_ratios)[len(positive_ratios)//2:len(positive_ratios)//2+n_third]
    high_idx = np.argsort(positive_ratios)[-n_third:]
    selected = np.concatenate([low_idx, mid_idx, high_idx])[:n_samples]

    preds = []
    for idx in selected:
        p = model.predict(X[idx:idx+1], verbose=0)[0]
        preds.append(p)
    preds = np.array(preds)

    fig, axes = plt.subplots(len(selected), 6, figsize=(24, 4*len(selected)))
    for i, idx in enumerate(selected):
        axes[i,0].imshow(X[idx,:,:,0]); axes[i,0].set_title('Umidade'); axes[i,0].axis('off')
        axes[i,1].imshow(X[idx,:,:,1]); axes[i,1].set_title('Temperatura'); axes[i,1].axis('off')
        axes[i,2].imshow(X[idx,:,:,2]); axes[i,2].set_title('Vento U'); axes[i,2].axis('off')
        axes[i,3].imshow(y[idx,:,:,0], vmin=0, vmax=1); axes[i,3].set_title('Frente Real'); axes[i,3].axis('off')
        axes[i,4].imshow(preds[i,:,:,0], vmin=0, vmax=1); axes[i,4].set_title('Predicao'); axes[i,4].axis('off')
        diff = np.abs(y[idx,:,:,0] - preds[i,:,:,0])
        axes[i,5].imshow(diff, vmin=0, vmax=1); axes[i,5].set_title('Diferenca'); axes[i,5].axis('off')

        y_true = y[idx,:,:,0].ravel()
        y_pred = (preds[i,:,:,0] > 0.5).astype(float).ravel()
        dice = 2 * np.sum(y_true*y_pred) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)
        intersection = np.sum(y_true*y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        iou = intersection / (union + 1e-7)
        axes[i,5].text(0.02, 0.98, f'Dice: {dice:.3f}\nIoU: {iou:.3f}', transform=axes[i,5].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = os.path.join(config.FIGURES_PATH, 'detailed_predictions.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Figura salva em {out}")