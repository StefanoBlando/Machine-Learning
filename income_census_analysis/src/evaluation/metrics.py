"""
Comprehensive evaluation metrics module.
Extracted from Module 6 of the original notebook - metrics section.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score, make_scorer
)

from ..config.settings import COST_FP, COST_FN, GAIN_TP, THRESHOLD_RANGE, THRESHOLD_STEPS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for binary classification.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (Optional[np.ndarray]): Predicted probabilities
        
    Returns:
        Dict[str, float]: Dictionary of metric values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }
    
    # Add probability-based metrics if available
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba),
                'avg_precision': average_precision_score(y_true, y_proba)
            })
        except ValueError as e:
            logger.warning(f"Could not calculate probability-based metrics: {e}")
            metrics.update({'roc_auc': np.nan, 'avg_precision': np.nan})
    else:
        metrics.update({'roc_auc': np.nan, 'avg_precision': np.nan})
    
    return metrics


def business_metrics_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Business-oriented cost-sensitive metrics analysis with threshold optimization.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels  
        y_proba (Optional[np.ndarray]): Predicted probabilities
        
    Returns:
        Dict[str, Any]: Business metrics including optimal threshold and profit
    """
    # Calculate confusion matrix with proper handling
    cm_default = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Ensure confusion matrix is 2x2
    if cm_default.shape != (2, 2):
        cm_default = np.zeros((2, 2), dtype=int)
        unique_pred = np.unique(y_pred)
        if 0 in unique_pred:
            cm_default[0, 0] = ((y_true == 0) & (y_pred == 0)).sum()
            cm_default[1, 0] = ((y_true == 1) & (y_pred == 0)).sum()
        if 1 in unique_pred:
            cm_default[0, 1] = ((y_true == 0) & (y_pred == 1)).sum()
            cm_default[1, 1] = ((y_true == 1) & (y_pred == 1)).sum()
    
    # Calculate current profit/cost at default threshold (0.5)
    current_profit = (cm_default[1, 1] * GAIN_TP) - (cm_default[0, 1] * COST_FP) - (cm_default[1, 0] * COST_FN)
    current_cost = (cm_default[0, 1] * COST_FP) + (cm_default[1, 0] * COST_FN) - (cm_default[1, 1] * GAIN_TP)
    
    optimal_threshold = np.nan
    max_profit = np.nan
    profit_curve = []
    
    # Threshold optimization if probabilities are available
    if y_proba is not None and len(np.unique(y_true)) > 1:
        thresholds = np.linspace(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1], THRESHOLD_STEPS)
        profits_at_thresholds = []
        
        for threshold in thresholds:
            pred_thresh = (y_proba >= threshold).astype(int)
            cm_thresh = confusion_matrix(y_true, pred_thresh, labels=[0, 1])
            
            # Ensure 2x2 matrix
            if cm_thresh.shape != (2, 2):
                temp_cm = np.zeros((2, 2), dtype=int)
                unique_pred_thresh = np.unique(pred_thresh)
                if 0 in unique_pred_thresh:
                    temp_cm[0, 0] = ((y_true == 0) & (pred_thresh == 0)).sum()
                    temp_cm[1, 0] = ((y_true == 1) & (pred_thresh == 0)).sum()
                if 1 in unique_pred_thresh:
                    temp_cm[0, 1] = ((y_true == 0) & (pred_thresh == 1)).sum()
                    temp_cm[1, 1] = ((y_true == 1) & (pred_thresh == 1)).sum()
                cm_thresh = temp_cm
            
            profit_thresh = (cm_thresh[1, 1] * GAIN_TP) - (cm_thresh[0, 1] * COST_FP) - (cm_thresh[1, 0] * COST_FN)
            profits_at_thresholds.append(profit_thresh)
        
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
        'profit_curve': profit_curve,
        'confusion_matrix_default': cm_default.tolist()
    }


def create_custom_scorer():
    """
    Create a custom scorer that combines multiple metrics.
    Adapted for make_scorer compatibility.
    
    Returns:
        sklearn scorer: Custom scoring function
    """
    def combined_scorer_func(y_true, y_score, **kwargs):
        """Combined scoring function for hyperparameter optimization."""
        y_pred = (y_score >= 0.5).astype(int)
        
        # Ensure we have data to work with
        if len(y_true) == 0:
            return 0.0
        
        # Calculate individual metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        combined_score = 0.0
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_score)
                # Weighted combination: 40% F1, 40% ROC-AUC, 20% Balanced Accuracy
                combined_score = 0.4 * f1 + 0.4 * roc_auc + 0.2 * balanced_acc
            except ValueError:
                # If ROC-AUC cannot be calculated
                combined_score = 0.6 * f1 + 0.4 * balanced_acc
        else:
            # If probabilities not available: 60% F1, 40% Balanced Accuracy
            combined_score = 0.6 * f1 + 0.4 * balanced_acc
        
        return combined_score
    
    return make_scorer(combined_scorer_func, needs_proba=True)


def get_classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Get classification report as a dictionary.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        Dict[str, Any]: Classification report dictionary
    """
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return report
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        return {}


def calculate_profit_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                                threshold: float) -> Tuple[float, Dict[str, int]]:
    """
    Calculate profit at a specific threshold.
    
    Args:
        y_true (np.ndarray): True labels
        y_proba (np.ndarray): Predicted probabilities
        threshold (float): Classification threshold
        
    Returns:
        Tuple[float, Dict[str, int]]: Profit value and confusion matrix components
    """
    y_pred_thresh = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_thresh, labels=[0, 1])
    
    # Ensure 2x2 matrix
    if cm.shape != (2, 2):
        cm = np.zeros((2, 2), dtype=int)
        unique_pred = np.unique(y_pred_thresh)
        if 0 in unique_pred:
            cm[0, 0] = ((y_true == 0) & (y_pred_thresh == 0)).sum()
            cm[1, 0] = ((y_true == 1) & (y_pred_thresh == 0)).sum()
        if 1 in unique_pred:
            cm[0, 1] = ((y_true == 0) & (y_pred_thresh == 1)).sum()
            cm[1, 1] = ((y_true == 1) & (y_pred_thresh == 1)).sum()
    
    # Calculate profit
    profit = (cm[1, 1] * GAIN_TP) - (cm[0, 1] * COST_FP) - (cm[1, 0] * COST_FN)
    
    cm_dict = {
        'tn': cm[0, 0], 'fp': cm[0, 1],
        'fn': cm[1, 0], 'tp': cm[1, 1]
    }
    
    return profit, cm_dict


def find_optimal_threshold_for_metric(y_true: np.ndarray, y_proba: np.ndarray, 
                                    metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal threshold for a specific metric.
    
    Args:
        y_true (np.ndarray): True labels
        y_proba (np.ndarray): Predicted probabilities
        metric (str): Metric to optimize ('f1', 'precision', 'recall', 'profit')
        
    Returns:
        Tuple[float, float]: Optimal threshold and metric value
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    metric_values = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            value = f1_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'precision':
            value = precision_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            value = recall_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'profit':
            value, _ = calculate_profit_at_threshold(y_true, y_proba, threshold)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        metric_values.append(value)
    
    if metric_values:
        optimal_idx = np.argmax(metric_values)
        return thresholds[optimal_idx], metric_values[optimal_idx]
    else:
        return 0.5, 0.0


def get_roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get ROC curve data for plotting.
    
    Args:
        y_true (np.ndarray): True labels
        y_proba (np.ndarray): Predicted probabilities
        
    Returns:
        Dict[str, np.ndarray]: ROC curve data
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    except Exception as e:
        logger.error(f"Error calculating ROC curve: {e}")
        return {
            'fpr': np.array([0, 1]),
            'tpr': np.array([0, 1]),
            'thresholds': np.array([1, 0]),
            'auc': 0.5
        }


def get_precision_recall_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get Precision-Recall curve data for plotting.
    
    Args:
        y_true (np.ndarray): True labels
        y_proba (np.ndarray): Predicted probabilities
        
    Returns:
        Dict[str, np.ndarray]: PR curve data
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'avg_precision': avg_precision
        }
    except Exception as e:
        logger.error(f"Error calculating PR curve: {e}")
        return {
            'precision': np.array([1, 0]),
            'recall': np.array([0, 1]),
            'thresholds': np.array([0]),
            'avg_precision': 0.0
        }


def calculate_cost_sensitive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   cost_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate cost-sensitive metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        cost_matrix (Optional[np.ndarray]): 2x2 cost matrix [[TN_cost, FP_cost], [FN_cost, TP_cost]]
        
    Returns:
        Dict[str, float]: Cost-sensitive metrics
    """
    if cost_matrix is None:
        # Default cost matrix: [TN=0, FP=COST_FP], [FN=COST_FN, TP=-GAIN_TP] (negative because it's a gain)
        cost_matrix = np.array([[0, COST_FP], [COST_FN, -GAIN_TP]])
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Ensure 2x2 matrix
    if cm.shape != (2, 2):
        cm = np.zeros((2, 2), dtype=int)
        unique_pred = np.unique(y_pred)
        if 0 in unique_pred:
            cm[0, 0] = ((y_true == 0) & (y_pred == 0)).sum()
            cm[1, 0] = ((y_true == 1) & (y_pred == 0)).sum()
        if 1 in unique_pred:
            cm[0, 1] = ((y_true == 0) & (y_pred == 1)).sum()
            cm[1, 1] = ((y_true == 1) & (y_pred == 1)).sum()
    
    # Calculate total cost
    total_cost = np.sum(cm * cost_matrix)
    
    # Calculate cost per prediction
    total_predictions = np.sum(cm)
    cost_per_prediction = total_cost / total_predictions if total_predictions > 0 else 0
    
    return {
        'total_cost': total_cost,
        'cost_per_prediction': cost_per_prediction,
        'profit': -total_cost,  # Profit is negative cost
        'confusion_matrix': cm.tolist()
    }


def evaluate_model_stability(y_true_list: list, y_pred_list: list, 
                           y_proba_list: Optional[list] = None) -> Dict[str, Any]:
    """
    Evaluate model stability across multiple predictions (e.g., from bootstrap or CV).
    
    Args:
        y_true_list (list): List of true label arrays
        y_pred_list (list): List of predicted label arrays
        y_proba_list (Optional[list]): List of predicted probability arrays
        
    Returns:
        Dict[str, Any]: Stability metrics
    """
    metrics_across_folds = []
    
    for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
        y_proba = y_proba_list[i] if y_proba_list else None
        fold_metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba)
        metrics_across_folds.append(fold_metrics)
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(metrics_across_folds)
    
    stability_summary = {}
    for metric in metrics_df.columns:
        stability_summary[f'{metric}_mean'] = metrics_df[metric].mean()
        stability_summary[f'{metric}_std'] = metrics_df[metric].std()
        stability_summary[f'{metric}_cv'] = metrics_df[metric].std() / metrics_df[metric].mean() if metrics_df[metric].mean() != 0 else np.inf
    
    return {
        'metrics_per_fold': metrics_across_folds,
        'stability_summary': stability_summary,
        'overall_stability_score': 1 / (1 + metrics_df['f1'].std())  # Stability score (higher = more stable)
    }


def compare_models_statistical(results_list: list, metric: str = 'f1') -> Dict[str, Any]:
    """
    Statistical comparison of multiple models.
    
    Args:
        results_list (list): List of model result dictionaries
        metric (str): Metric to compare
        
    Returns:
        Dict[str, Any]: Statistical comparison results
    """
    model_scores = {}
    
    for result in results_list:
        model_name = result.get('model_name', 'Unknown')
        if 'validation_metrics_optimal_threshold' in result:
            score = result['validation_metrics_optimal_threshold'].get(metric, np.nan)
            if not np.isnan(score):
                model_scores[model_name] = score
    
    if len(model_scores) < 2:
        return {'error': 'Need at least 2 models for comparison'}
    
    # Sort models by performance
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate relative improvements
    best_score = sorted_models[0][1]
    comparisons = []
    
    for model_name, score in sorted_models[1:]:
        improvement = ((best_score - score) / score) * 100 if score > 0 else np.inf
        comparisons.append({
            'model': model_name,
            'score': score,
            'improvement_over_this': improvement
        })
    
    return {
        'best_model': sorted_models[0][0],
        'best_score': sorted_models[0][1],
        'model_ranking': sorted_models,
        'comparisons': comparisons,
        'score_range': max(model_scores.values()) - min(model_scores.values()),
        'mean_score': np.mean(list(model_scores.values())),
        'std_score': np.std(list(model_scores.values()))
    }


def get_metric_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all metrics used.
    
    Returns:
        Dict[str, str]: Metric descriptions
    """
    return {
        'accuracy': 'Overall correctness: (TP + TN) / (TP + TN + FP + FN)',
        'precision': 'Positive predictive value: TP / (TP + FP)',
        'recall': 'Sensitivity/True positive rate: TP / (TP + FN)',
        'f1': 'Harmonic mean of precision and recall: 2 * (precision * recall) / (precision + recall)',
        'balanced_accuracy': 'Average of sensitivity and specificity: (TPR + TNR) / 2',
        'matthews_corrcoef': 'Correlation between predicted and actual: (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))',
        'roc_auc': 'Area under ROC curve: measures discrimination ability',
        'avg_precision': 'Area under Precision-Recall curve: good for imbalanced datasets',
        'profit': f'Business profit: {GAIN_TP}*TP - {COST_FP}*FP - {COST_FN}*FN'
    }
