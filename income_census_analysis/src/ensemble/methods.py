"""
Advanced ensemble methods module.

"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from ..config.settings import RANDOM_STATE
from ..evaluation.metrics import calculate_comprehensive_metrics, business_metrics_analysis


def create_advanced_ensembles(all_results: list, balanced_datasets: dict,
                              X_val_preprocessed: np.ndarray, y_val: pd.Series) -> tuple[dict, list]:
    """
    Crea ensemble avanzati (Voting, Stacking, Weighted) con analisi di diversità.
    Seleziona i migliori modelli tra tutte le strategie di campionamento.
    """
    print("\n8. ADVANCED ENSEMBLE CREATION")
    print("-" * 50)
    
    # Filter for successful models and sort by F1-score (optimal threshold)
    # Ensure best_model is not None and F1 is not NaN
    top_successful_results = sorted([
        r for r in all_results 
        if r['best_model'] is not None and not np.isnan(r['validation_metrics_optimal_threshold'].get('f1', np.nan))
    ], 
    key=lambda x: x['validation_metrics_optimal_threshold']['f1'], 
    reverse=True)
    
    if len(top_successful_results) < 2:
        print("❌ Not enough successful models with valid F1 scores to create ensembles (need at least 2).")
        return {}, []
    
    # Select top 5 models for ensemble building (can be from different sampling strategies)
    models_for_ensemble = top_successful_results[:5]

    print(f"🔗 Creating ensembles from top {len(models_for_ensemble)} models across all sampling strategies:\n")
    for i, result in enumerate(models_for_ensemble, 1):
        print(f"   {i}. {result['model_name']} (Sampling: {result['sampling_strategy']}) - F1: {result['validation_metrics_optimal_threshold']['f1']:.4f}")
    
    ensemble_results = {}
    
    # Get training data from the sampling strategy of the overall best individual model for ensemble fitting
    # This ensures consistency, though in practice, training data for stacking can vary.
    overall_best_individual = models_for_ensemble[0]
    X_train_for_ensemble = balanced_datasets[overall_best_individual['sampling_strategy']]['X_train']
    y_train_for_ensemble = balanced_datasets[overall_best_individual['sampling_strategy']]['y_train']
    
    # 1. Voting Classifier (Soft voting)
    print(f"\n🗳️ Creating Voting Ensemble (Soft Voting)...\n")
    try:
        # Filter for models that have 'predict_proba' method
        voting_estimators = []
        for result in models_for_ensemble:
            if hasattr(result['best_model'], 'predict_proba'):
                voting_estimators.append((f"{result['model_name']}_{result['sampling_strategy']}", result['best_model']))
            else:
                print(f"   ⚠️ Skipping {result['model_name']} (Sampling: {result['sampling_strategy']}) for Voting Ensemble (no predict_proba).")

        if len(voting_estimators) < 2:
            print("   ❌ Not enough models with predict_proba for Voting Ensemble.")
        else:
            voting_ensemble = VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=-1)
            
            # Fit the VotingClassifier (uses the training data of the best individual model)
            voting_ensemble.fit(X_train_for_ensemble, y_train_for_ensemble)
            
            voting_pred = voting_ensemble.predict(X_val_preprocessed)
            voting_proba = voting_ensemble.predict_proba(X_val_preprocessed)[:, 1]
            
            voting_metrics = calculate_comprehensive_metrics(y_val, voting_pred, voting_proba)
            voting_business = business_metrics_analysis(y_val, voting_pred, voting_proba)
            
            ensemble_results['voting'] = {
                'model': voting_ensemble,
                'metrics': voting_metrics,
                'business_metrics': voting_business,
                'predictions': voting_pred,
                'probabilities': voting_proba,
                'type': 'voting'
            }
            
            print(f"   ✅ Voting Ensemble created")
            print(f"      F1-Score: {voting_metrics['f1']:.4f}")
            print(f"      ROC-AUC: {voting_metrics['roc_auc']:.4f}\n")
            
    except Exception as e:
        print(f"   ❌ Voting ensemble failed: {str(e)}\n")
    
    # 2. Stacking Classifier
    print(f"🏗️ Creating Stacking Ensemble...\n")
    try:
        # Base estimators (top 3-4 models, ensure they have predict_proba)
        base_estimators = []
        for result in models_for_ensemble[:4]: # Limit to top 4 for stacking, can adjust
            if hasattr(result['best_model'], 'predict_proba'):
                base_estimators.append((f"{result['model_name']}_{result['sampling_strategy']}", result['best_model']))
            else:
                print(f"   ⚠️ Skipping {result['model_name']} (Sampling: {result['sampling_strategy']}) for Stacking Ensemble (no predict_proba).")

        if len(base_estimators) < 2:
            print("   ❌ Not enough models with predict_proba for Stacking Ensemble.")
        else:
            # Meta-learner (Logistic Regression is a common choice for simplicity)
            meta_learner = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000)
            
            stacking_ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5, # Use a fixed CV for stacking, e.g., 5-fold
                stack_method='predict_proba', # Use probabilities from base models
                n_jobs=-1
            )
            
            stacking_ensemble.fit(X_train_for_ensemble, y_train_for_ensemble)
            
            stacking_pred = stacking_ensemble.predict(X_val_preprocessed)
            stacking_proba = stacking_ensemble.predict_proba(X_val_preprocessed)[:, 1]
            
            stacking_metrics = calculate_comprehensive_metrics(y_val, stacking_pred, stacking_proba)
            stacking_business = business_metrics_analysis(y_val, stacking_pred, stacking_proba)
            
            ensemble_results['stacking'] = {
                'model': stacking_ensemble,
                'metrics': stacking_metrics,
                'business_metrics': stacking_business,
                'predictions': stacking_pred,
                'probabilities': stacking_proba,
                'type': 'stacking'
            }
            
            print(f"   ✅ Stacking Ensemble created")
            print(f"      F1-Score: {stacking_metrics['f1']:.4f}")
            print(f"      ROC-AUC: {stacking_metrics['roc_auc']:.4f}\n")
            
    except Exception as e:
        print(f"   ❌ Stacking ensemble failed: {str(e)}\n")
    
    # 3. Weighted Average Ensemble (simple, probability-based)
    print(f"⚖️ Creating Weighted Average Ensemble...\n")
    try:
        # Collect probabilities and F1-scores from successful models that provide probas
        models_for_weighted_avg = []
        for result in models_for_ensemble:
            if 'y_val_proba' in result and result['y_val_proba'] is not None and not np.isnan(result['validation_metrics_optimal_threshold'].get('f1', np.nan)):
                models_for_weighted_avg.append(result)
            else:
                print(f"   ⚠️ Skipping {result['model_name']} (Sampling: {result['sampling_strategy']}) for Weighted Average Ensemble (no probabilities or invalid F1).")

        if not models_for_weighted_avg:
            print("   ❌ No models with valid probabilities for Weighted Average Ensemble.")
        else:
            # Calculate weights based on validation F1-score (optimal threshold)
            f1_scores_for_weights = np.array([r['validation_metrics_optimal_threshold']['f1'] for r in models_for_weighted_avg])
            
            # Normalize weights so they sum to 1. Add a small epsilon to avoid division by zero if all F1s are 0.
            if f1_scores_for_weights.sum() > 1e-6:
                weights = f1_scores_for_weights / f1_scores_for_weights.sum()
            else: # Fallback if all F1s are zero (unlikely with good models)
                weights = np.ones(len(models_for_weighted_avg)) / len(models_for_weighted_avg)
            
            print(f"   Model weights (based on F1-Score at Optimal Threshold):\n")
            for i, (result, weight) in enumerate(zip(models_for_weighted_avg, weights)):
                print(f"      {result['model_name']}: {weight:.3f}")
            
            # Weighted average of probabilities
            weighted_probabilities = np.zeros(len(y_val))
            for result, weight in zip(models_for_weighted_avg, weights):
                weighted_probabilities += weight * result['y_val_proba']
            
            # Determine predictions based on a default (e.g., 0.5) threshold for the weighted average
            weighted_pred = (weighted_probabilities >= 0.5).astype(int)
            
            weighted_metrics = calculate_comprehensive_metrics(y_val, weighted_pred, weighted_probabilities)
            weighted_business = business_metrics_analysis(y_val, weighted_pred, weighted_probabilities)
            
            ensemble_results['weighted'] = {
                'model': None,  # Not a scikit-learn model object
                'metrics': weighted_metrics,
                'business_metrics': weighted_business,
                'predictions': weighted_pred,
                'probabilities': weighted_probabilities,
                'weights': weights,
                'type': 'weighted'
            }
            
            print(f"   ✅ Weighted Average Ensemble created")
            print(f"      F1-Score: {weighted_metrics['f1']:.4f}")
            print(f"      ROC-AUC: {weighted_metrics['roc_auc']:.4f}\n")
            
    except Exception as e:
        print(f"   ❌ Weighted average ensemble failed: {str(e)}\n")
    
    return ensemble_results, models_for_ensemble # Return the list of top models actually used for ensemble
