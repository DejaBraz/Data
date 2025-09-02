#!/usr/bin/env python
# -*- coding: utf-8 -*-

# avaliar_modelo.py - Script de avaliacao do modelo treinado
# Versao final sem acentos, compativel com todos os modulos

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Suprimir warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Adicionar diretorio pai ao path para importar modulos
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Importar modulos do projeto
    from modelo_unet.config import FullDatasetConfig
    from modelo_unet.data_processing import ChunkedDataProcessor
    from modelo_unet.losses_metrics import (
        advanced_dice_coefficient, 
        iou_metric_stable, 
        boundary_iou,
        precision_metric, 
        recall_metric,
        postprocess_prediction,
        focal_tversky_loss
    )
    from modelo_unet.utils import set_seeds, log_memory
    
except ImportError as e:
    print("=" * 70)
    print("ERRO DE IMPORTACAO")
    print("=" * 70)
    print(f"Erro: {e}")
    print("Certifique-se de que:")
    print("1. O diretorio 'modelo_unet' existe")
    print("2. Todos os arquivos .py estao presentes")
    print("3. Execute primeiro: python run_modelo.py")
    print("=" * 70)
    sys.exit(1)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("avaliacao")

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.processor = ChunkedDataProcessor(config)
        
    def load_best_model(self):
        # Carrega o melhor modelo salvo
        model_files = list(self.config.MODELS_PATH.glob('best_model_fold_*.keras'))
        if not model_files:
            model_files = list(self.config.MODELS_PATH.glob('*.keras'))
        
        if not model_files:
            raise FileNotFoundError("Nenhum modelo encontrado. Execute primeiro run_modelo.py")
        
        # Usar o primeiro modelo encontrado
        model_path = model_files[0]
        logger.info(f"Carregando modelo: {model_path}")
        
        try:
            # Importar a funcao de loss necessaria
            from modelo_unet.losses_metrics import focal_tversky_loss
            
            # Criar instancia da funcao de loss
            loss_function = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=2.0, focal_weight=0.25)
            
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'loss_function': loss_function,
                    'focal_tversky_loss': focal_tversky_loss,
                    'advanced_dice_coefficient': advanced_dice_coefficient,
                    'iou_metric_stable': iou_metric_stable,
                    'boundary_iou': boundary_iou,
                    'precision_metric': precision_metric,
                    'recall_metric': recall_metric
                }
            )
            logger.info("Modelo carregado com sucesso")
            return model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            # Tentar carregamento alternativo sem loss function
            logger.info("Tentando carregamento sem loss function...")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'advanced_dice_coefficient': advanced_dice_coefficient,
                        'iou_metric_stable': iou_metric_stable,
                        'boundary_iou': boundary_iou,
                        'precision_metric': precision_metric,
                        'recall_metric': recall_metric
                    },
                    compile=False
                )
                logger.info("Modelo carregado sem compilacao")
                return model
            except Exception as e2:
                logger.error(f"Falha no carregamento alternativo: {e2}")
                raise e
    
    def load_test_data(self):
        # Carrega dados de teste
        logger.info("Carregando dados de teste...")
        X, y = self.processor.process_full_dataset()
        
        # Usar ultimos 10% como teste
        n_test = int(len(X) * 0.1)
        test_indices = np.arange(len(X) - n_test, len(X))
        
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        logger.info(f"Dados de teste: {X_test.shape}, {y_test.shape}")
        return X_test, y_test
    
    def evaluate_model_metrics(self, model, X_test, y_test):
        # Avalia metricas do modelo
        logger.info("Avaliando metricas do modelo...")
        
        # Predicoes
        y_pred = model.predict(X_test, batch_size=4, verbose=1)
        
        # Metricas por amostra
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        
        for i in range(len(y_test)):
            y_true = y_test[i].flatten()
            y_pred_binary = (y_pred[i] > 0.1).flatten().astype(float)
            
            # Dice
            intersection = np.sum(y_true * y_pred_binary)
            dice = (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_binary) + 1e-7)
            dice_scores.append(dice)
            
            # IoU
            union = np.sum(y_true) + np.sum(y_pred_binary) - intersection
            iou = intersection / (union + 1e-7)
            iou_scores.append(iou)
            
            # Precision
            if np.sum(y_pred_binary) > 0:
                precision = intersection / np.sum(y_pred_binary)
            else:
                precision = 0
            precision_scores.append(precision)
            
            # Recall
            if np.sum(y_true) > 0:
                recall = intersection / np.sum(y_true)
            else:
                recall = 0
            recall_scores.append(recall)
        
        metrics = {
            'dice': {'mean': np.mean(dice_scores), 'std': np.std(dice_scores), 'scores': dice_scores},
            'iou': {'mean': np.mean(iou_scores), 'std': np.std(iou_scores), 'scores': iou_scores},
            'precision': {'mean': np.mean(precision_scores), 'std': np.std(precision_scores), 'scores': precision_scores},
            'recall': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores), 'scores': recall_scores}
        }
        
        return metrics, y_pred
    
    def plot_evaluation_results(self, metrics, y_test, y_pred):
        # Plota resultados da avaliacao
        logger.info("Gerando visualizacoes...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Metricas boxplot
        ax = axes[0, 0]
        metric_data = [metrics['dice']['scores'], metrics['iou']['scores'], 
                      metrics['precision']['scores'], metrics['recall']['scores']]
        ax.boxplot(metric_data, labels=['Dice', 'IoU', 'Precision', 'Recall'])
        ax.set_title('Distribuicao das Metricas')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        
        # Histograma Dice
        ax = axes[0, 1]
        ax.hist(metrics['dice']['scores'], bins=20, alpha=0.7, color='blue')
        dice_mean = metrics['dice']['mean']
        ax.axvline(dice_mean, color='red', linestyle='--', 
                  label=f'Media: {dice_mean:.3f}')
        ax.set_xlabel('Dice Score')
        ax.set_ylabel('Frequencia')
        ax.set_title('Distribuicao Dice Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter plot correlacoes
        ax = axes[0, 2]
        ax.scatter(metrics['precision']['scores'], metrics['recall']['scores'], alpha=0.6)
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')
        ax.set_title('Precision vs Recall')
        ax.grid(True, alpha=0.3)
        
        # Exemplos de predicoes
        n_examples = 3
        indices = np.random.choice(len(y_test), n_examples, replace=False)
        
        for i, idx in enumerate(indices):
            ax = axes[1, i]
            
            # Criar subplot com 3 imagens
            img_combined = np.zeros((y_test.shape[1], y_test.shape[2] * 3))
            
            # Ground truth
            img_combined[:, :y_test.shape[2]] = y_test[idx, :, :, 0]
            
            # Predicao
            img_combined[:, y_test.shape[2]:2*y_test.shape[2]] = y_pred[idx, :, :, 0]
            
            # Diferenca
            diff = np.abs(y_test[idx, :, :, 0] - y_pred[idx, :, :, 0])
            img_combined[:, 2*y_test.shape[2]:] = diff
            
            dice_score = metrics["dice"]["scores"][idx]
            ax.imshow(img_combined, cmap='viridis')
            ax.set_title(f'Exemplo {i+1}: GT | Pred | Diff - Dice: {dice_score:.3f}')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Salvar figura
        save_path = self.config.FIGURES_PATH / 'avaliacao_modelo.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figura salva em: {save_path}")
        plt.show()
    
    def generate_report(self, metrics):
        # Gera relatorio de avaliacao
        logger.info("Gerando relatorio...")
        
        dice_mean = metrics['dice']['mean']
        dice_std = metrics['dice']['std']
        iou_mean = metrics['iou']['mean']
        iou_std = metrics['iou']['std']
        precision_mean = metrics['precision']['mean']
        precision_std = metrics['precision']['std']
        recall_mean = metrics['recall']['mean']
        recall_std = metrics['recall']['std']
        
        q25 = np.percentile(metrics['dice']['scores'], 25)
        q50 = np.percentile(metrics['dice']['scores'], 50)
        q75 = np.percentile(metrics['dice']['scores'], 75)
        
        report_lines = []
        report_lines.append("========================================")
        report_lines.append("RELATORIO DE AVALIACAO DO MODELO")
        report_lines.append("========================================")
        report_lines.append("")
        report_lines.append("METRICAS GERAIS:")
        report_lines.append(f"- Dice Score:     {dice_mean:.4f} +/- {dice_std:.4f}")
        report_lines.append(f"- IoU Score:      {iou_mean:.4f} +/- {iou_std:.4f}")
        report_lines.append(f"- Precision:      {precision_mean:.4f} +/- {precision_std:.4f}")
        report_lines.append(f"- Recall:         {recall_mean:.4f} +/- {recall_std:.4f}")
        report_lines.append("")
        report_lines.append("PERFORMANCE POR QUARTIS (Dice Score):")
        report_lines.append(f"- Q1 (25%):       {q25:.4f}")
        report_lines.append(f"- Mediana (50%):  {q50:.4f}")
        report_lines.append(f"- Q3 (75%):       {q75:.4f}")
        report_lines.append("")
        report_lines.append("AMOSTRAS COM MELHOR PERFORMANCE:")
        
        # Top 5 melhores Dice scores
        best_indices = np.argsort(metrics['dice']['scores'])[-5:][::-1]
        for i, idx in enumerate(best_indices):
            dice_score = metrics['dice']['scores'][idx]
            iou_score = metrics['iou']['scores'][idx]
            report_lines.append(f"  {i+1}. Amostra {idx}: Dice={dice_score:.4f}, IoU={iou_score:.4f}")
        
        report_lines.append("")
        report_lines.append("AMOSTRAS COM PIOR PERFORMANCE:")
        worst_indices = np.argsort(metrics['dice']['scores'])[:5]
        for i, idx in enumerate(worst_indices):
            dice_score = metrics['dice']['scores'][idx]
            iou_score = metrics['iou']['scores'][idx]
            report_lines.append(f"  {i+1}. Amostra {idx}: Dice={dice_score:.4f}, IoU={iou_score:.4f}")
        
        performance_level = 'BOM' if dice_mean > 0.7 else 'MODERADO' if dice_mean > 0.5 else 'BAIXO'
        variability_level = 'BAIXA' if dice_std < 0.1 else 'MODERADA' if dice_std < 0.2 else 'ALTA'
        balance_level = 'BALANCEADO' if abs(precision_mean - recall_mean) < 0.1 else 'DESBALANCEADO'
        
        report_lines.append("")
        report_lines.append("ANALISE:")
        report_lines.append(f"- Modelo apresenta desempenho {performance_level}")
        report_lines.append(f"- Variabilidade {variability_level}")
        report_lines.append(f"- Balance Precision/Recall: {balance_level}")
        report_lines.append("")
        report_lines.append("========================================")
        
        report = '\n'.join(report_lines)
        
        # Salvar relatorio
        report_path = self.config.FIGURES_PATH / 'relatorio_avaliacao.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Relatorio salvo em: {report_path}")
        print(report)
        
        return report

def main():
    # Funcao principal de avaliacao
    print("=" * 70)
    print("AVALIACAO DO MODELO DE DETECCAO DE FRENTES FRIAS")
    print("=" * 70)
    
    try:
        # Configuracao
        config = FullDatasetConfig()
        config.ensure_dirs()
        set_seeds(config.RANDOM_SEED)
        
        # Inicializar avaliador
        evaluator = ModelEvaluator(config)
        
        # Carregar modelo
        model = evaluator.load_best_model()
        
        # Carregar dados de teste
        X_test, y_test = evaluator.load_test_data()
        
        # Avaliar modelo
        metrics, y_pred = evaluator.evaluate_model_metrics(model, X_test, y_test)
        
        # Visualizar resultados
        evaluator.plot_evaluation_results(metrics, y_test, y_pred)
        
        # Gerar relatorio
        evaluator.generate_report(metrics)
        
        print("=" * 70)
        print("AVALIACAO CONCLUIDA COM SUCESSO!")
        print("Verifique os arquivos gerados em:", config.FIGURES_PATH)
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Erro durante avaliacao: {e}")
        print("=" * 70)
        print("ERRO NA AVALIACAO")
        print("=" * 70)
        print(f"Erro: {e}")
        print("Certifique-se de ter executado o treinamento primeiro.")
        sys.exit(1)

if __name__ == "__main__":
    main()