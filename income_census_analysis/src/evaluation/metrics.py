"""
Comprehensive evaluation metrics module.

"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """
    Calcola metriche comprehensive per la valutazione, adatte a classificazione binaria.
    Corretto per il parametro 'zero_division'.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0), 
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1: 
        try:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba),
                'avg_precision': average_precision_score(y_true, y_proba)
            })
        except ValueError: 
            metrics.update({'roc_auc': np.nan, 'avg_precision': np.nan}) 
    else:
        metrics.update({'roc_auc': np.nan, 'avg_precision': np.nan}) 
    
    return metrics


def business_metrics_analysis(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """
    Analisi business-oriented con cost-sensitive metrics.
    Ottimizza la soglia per massimizzare il profitto.
    Cost matrix: [TN, FP], [FN, TP]
    """
    # Define costs/gains: TN=0, FP=1, FN=3, TP=5
    cost_fp = 1
    cost_fn = 3
    gain_tp = 5

    cm_default = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Ensure cm_default is 2x2. Pad if needed for cases where predictions are all one class.
    if cm_default.shape != (2,2):
        cm_default = np.zeros((2,2), dtype=int)
        unique_pred = np.unique(y_pred)
        if 0 in unique_pred:
            cm_default[0,0] = ((y_true == 0) & (y_pred == 0)).sum()
            cm_default[1,0] = ((y_true == 1) & (y_pred == 0)).sum()
        if 1 in unique_pred:
            cm_default[0,1] = ((y_true == 0) & (y_pred == 1)).sum(); cm_default[1,1] = ((y_true == 1) & (y_pred == 1)).sum()

    current_profit = (cm_default[1,1] * gain_tp) - (cm_default[0,1] * cost_fp) - (cm_default[1,0] * cost_fn)
    current_cost = (cm_default[0,1] * cost_fp) + (cm_default[1,0] * cost_fn) - (cm_default[1,1] * gain_tp) # sum of matrix with negative gain

    optimal_threshold = np.nan
    max_profit = np.nan
    profit_curve = []
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        thresholds = np.linspace(0.01, 0.99, 100) # More granular thresholds for optimization
        profits_at_thresholds = []
        
        for threshold in thresholds:
            pred_thresh = (y_proba >= threshold).astype(int)
            cm_thresh = confusion_matrix(y_true, pred_thresh, labels=[0, 1])

            # Ensure cm_thresh is 2x2, even if predictions are all one class for a given threshold
            if cm_thresh.shape != (2,2):
                temp_cm = np.zeros((2,2), dtype=int)
                unique_pred_thresh = np.unique(pred_thresh)
                if 0 in unique_pred_thresh: temp_cm[0,0] = ((y_true == 0) & (pred_thresh == 0)).sum(); temp_cm[1,0] = ((y_true == 1) & (pred_thresh == 0)).sum()
                if 1 in unique_pred_thresh: temp_cm[0,1] = ((y_true == 0) & (pred_thresh == 1)).sum(); temp_cm[1,1] = ((y_true == 1) & (pred_thresh == 1)).sum()
                cm_thresh = temp_cm

            current_profit_thresh = (cm_thresh[1,1] * gain_tp) - (cm_thresh[0,1] * cost_fp) - (cm_thresh[1,0] * cost_fn)
            profits_at_thresholds.append(current_profit_thresh)
        
        if profits_at_thresholds:
            optimal_threshold_idx = np.argmax(profits_at_thresholds)
            optimal_threshold = thresholds[optimal_threshold_idx]
            max_profit = profits_at_thresholds[optimal_threshold_idx]
            profit_curve = list(zip(thresholds, profits_at_thresholds))
        
    return {
        'current_cost': current_cost,
        'current_profit': current_profit,
        'optimal_threshold': optimal_threshold,
        'max_profit': max_profit,
        'profit_curve': profit_curve
    }
