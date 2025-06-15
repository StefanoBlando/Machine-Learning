"""
Model interpretability analysis module.

"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.inspection import permutation_importance

# SHAP imports (with availability check)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..config.settings import RANDOM_STATE
from ..evaluation.metrics import calculate_comprehensive_metrics


def model_interpretability_analysis(best_result: dict, top_results_for_ensemble: list, feature_names: list, 
                                    X_val_preprocessed: np.ndarray, y_val: pd.Series) -> dict:
    """
    Analisi di interpretabilità usando Feature Importance, Permutation Importance e SHAP.
    Si concentra sul migliore modello individuale.
    """
    print(f"\n8. MODEL INTERPRETABILITY ANALYSIS")
    print("-" * 50)
    
    interpretability_results = {}
    
    # Use the overall best individual model for interpretability
    best_individual_model = None
    if best_result and best_result['best_model'] is not None:
        best_individual_model = best_result['best_model']
        model_name = best_result['model_name']
    elif top_results_for_ensemble: # Fallback to the top model used for ensemble
        best_individual_model = top_results_for_ensemble[0]['best_model']
        model_name = top_results_for_ensemble[0]['model_name']
        print(f"⚠️ Best result not found, falling back to top model used for ensembles for interpretability ({model_name}).")
    else:
        print("❌ No valid model available for interpretability analysis.")
        return interpretability_results
    
    # 1. Feature Importance Analysis (for tree-based and linear models)
    print(f"📊 Feature Importance Analysis (from {model_name})...\n")
    if hasattr(best_individual_model, 'feature_importances_'):
        importances = best_individual_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"   ✅ Feature importances extracted from {model_name}")
        print(f"   📈 Top 15 most important features (by model's internal importance):\n")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"      {i:2d}. {str(row['feature']):<30} {row['importance']:.4f}")
        
        interpretability_results['feature_importance'] = importance_df
    elif hasattr(best_individual_model, 'coef_') and len(best_individual_model.coef_.shape) == 1: # For linear models
        importances = np.abs(best_individual_model.coef_) # Absolute coefficients for importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(f"   ✅ Coefficients extracted from {model_name} (as importance)")
        print(f"   📈 Top 15 most important features (by absolute coefficients):\n")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"      {i:2d}. {str(row['feature']):<30} {row['importance']:.4f}")
        interpretability_results['feature_importance'] = importance_df
    else:
        print(f"   ⚠️ Model {model_name} does not have standard feature_importances_ or coef_ attribute.")
    
    # 2. Permutation Importance (model-agnostic)
    print(f"\n🔄 Permutation Importance Analysis...\n")
    try:
        sample_size = min(2000, len(X_val_preprocessed)) 
        sample_idx = np.random.choice(len(X_val_preprocessed), sample_size, replace=False)
        X_sample = X_val_preprocessed[sample_idx]
        y_sample = y_val.iloc[sample_idx] if hasattr(y_val, 'iloc') else y_val[sample_idx]
        
        perm_importance = permutation_importance(
            best_individual_model, X_sample, y_sample,
            n_repeats=5, 
            random_state=RANDOM_STATE,
            scoring='f1', 
            n_jobs=-1 
        )
        
        perm_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print(f"   ✅ Permutation importance calculated on {sample_size} samples.")
        print(f"   📈 Top 15 features by permutation importance:\n")
        for i, (_, row) in enumerate(perm_importance_df.head(15).iterrows(), 1):
            print(f"      {i:2d}. {str(row['feature']):<30} {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        interpretability_results['permutation_importance'] = perm_importance_df
        
    except Exception as e:
        print(f"   ❌ Permutation importance failed: {str(e)}")
    
    # 3. SHAP Analysis (model-agnostic, for local interpretability)
    if SHAP_AVAILABLE:
        print(f"\n🎯 SHAP Analysis...\n")
        try:
            shap_sample_size = min(1000, len(X_val_preprocessed)) 
            shap_idx = np.random.choice(len(X_val_preprocessed), shap_sample_size, replace=False)
            X_shap = X_val_preprocessed[shap_idx]
            
            # Choose appropriate explainer based on model type
            if "XGB" in model_name or "LightGBM" in model_name or "Forest" in model_name or "Boosting" in model_name:
                explainer = shap.TreeExplainer(best_individual_model)
                shap_values = explainer.shap_values(X_shap)
                
                if isinstance(shap_values, list): # For multi-output models like binary classification
                    shap_values = shap_values[1] 
                
            else: # For linear models, SVMs, etc.
                # Use a small subset of training data as background for KernelExplainer
                # This would require access to balanced_datasets, which should be passed or made available
                # For now, we'll create a simple background from the validation data itself
                background_data_shap = X_val_preprocessed[np.random.choice(len(X_val_preprocessed), min(100, len(X_val_preprocessed)), replace=False)]
                
                # Ensure model has predict_proba, if not, use predict and wrap with lambda
                if hasattr(best_individual_model, 'predict_proba'):
                    model_predict_func = best_individual_model.predict_proba
                    shap_output_idx = 1 # Take probabilities for positive class
                else: # Fallback for models without predict_proba (less common for custom scorer)
                    model_predict_func = best_individual_model.predict
                    shap_output_idx = 0 # Take single output
                    print(f"   ⚠️ Model {model_name} does not have predict_proba. Using predict for SHAP (less ideal for classification).")

                explainer = shap.KernelExplainer(model_predict_func, background_data_shap)
                shap_values_raw = explainer.shap_values(X_shap)
                
                if isinstance(shap_values_raw, list): # For multi-output models like binary classification
                    shap_values = shap_values_raw[shap_output_idx] 
                else:
                    shap_values = shap_values_raw # For single output models

            # Calculate mean absolute SHAP values for global importance
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            shap_importance_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': mean_shap_values
            }).sort_values('shap_importance', ascending=False)
            
            print(f"   ✅ SHAP values calculated for {shap_sample_size} samples.")
            print(f"   📈 Top 15 features by SHAP importance (Mean Absolute SHAP Value):\n")
            for i, (_, row) in enumerate(shap_importance_df.head(15).iterrows(), 1):
                print(f"      {i:2d}. {str(row['feature']):<30} {row['shap_importance']:.4f}")
            
            interpretability_results['shap_values'] = shap_values
            interpretability_results['shap_importance'] = shap_importance_df
            interpretability_results['shap_explainer'] = explainer
            interpretability_results['X_shap'] = X_shap
            
        except Exception as e:
            print(f"   ❌ SHAP analysis failed: {str(e)}. Ensure SHAP is installed and compatible with the model/data.")
    else:
        print(f"\n⚠️ SHAP not available for interpretability analysis.")
    
    # 4. Model Performance Stability Analysis (Bootstrap on validation set)
    print(f"\n📈 Model Stability Analysis (Bootstrap)... \n")
    try:
        stability_scores = []
        n_bootstrap_samples = 20 
        
        for i in range(n_bootstrap_samples):
            bootstrap_idx = np.random.choice(len(X_val_preprocessed), len(X_val_preprocessed), replace=True)
            X_bootstrap = X_val_preprocessed[bootstrap_idx]
            y_bootstrap = y_val.iloc[bootstrap_idx] if hasattr(y_val, 'iloc') else y_val[bootstrap_idx]
            
            y_pred_bootstrap = best_individual_model.predict(X_bootstrap)
            f1_bootstrap = f1_score(y_bootstrap, y_pred_bootstrap, zero_division=0)
            stability_scores.append(f1_bootstrap)
        
        stability_mean = np.mean(stability_scores)
        stability_std = np.std(stability_scores)
        
        stability_cv = (stability_std / stability_mean) if stability_mean != 0 else np.inf 
        
        print(f"   ✅ Stability analysis completed ({n_bootstrap_samples} bootstrap samples).")
        print(f"   📊 F1-Score stability: {stability_mean:.4f} ± {stability_std:.4f}")
        print(f"   📊 Coefficient of Variation (CV): {stability_cv:.3f}")
        
        interpretability_results['stability'] = {
            'mean': stability_mean,
            'std': stability_std,
            'scores': stability_scores,
            'cv': stability_cv
        }
        
    except Exception as e:
        print(f"   ❌ Stability analysis failed: {str(e)}")
    
    return interpretability_results
