"""
Hyperparameter optimization module.

"""
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any
from sklearn.model_selection import cross_val_score

# Optimization imports (with availability checks)
try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    from sklearn.model_selection import RandomizedSearchCV

from ..config.settings import RANDOM_STATE
from ..evaluation.metrics import calculate_comprehensive_metrics, business_metrics_analysis


def get_smart_iterations_per_model(model_name: str) -> int:
    """
    Restituisce un numero 'smart' di iterazioni per la ricerca degli iperparametri
    basato sulla complessità del modello e la dimensione dello spazio.
    """
    # These values are tuned for speed and reasonable performance, not exhaustive search
    smart_iterations = {
        'Logistic_Regression': 5,
        'SVM': 8,                   
        'Random_Forest': 10,       
        'Gradient_Boosting': 12,    
        'XGBoost': 15,              
        'LightGBM': 12              
    }
    return smart_iterations.get(model_name, 10) 


def get_focused_param_space(model_name: str, model_config: dict, skopt_available: bool) -> dict:
    """
    Restituisce spazi parametri focalizzati per una maggiore velocità della ricerca.
    Questi spazi sono usati per la ricerca non-nested di default.
    """
    if model_name == 'Logistic_Regression':
        if skopt_available:
            from skopt.space import Real, Categorical
            return {'C': Real(0.1, 10, prior='log-uniform'), 'penalty': Categorical(['l1', 'l2'])}
        else:
            return {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
    
    elif model_name == 'SVM':
        if skopt_available:
            from skopt.space import Real, Categorical
            return {'C': Real(0.1, 10, prior='log-uniform'), 'kernel': Categorical(['rbf']), 'gamma': Categorical(['scale', 'auto'])}
        else:
            return {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}
    
    elif model_name == 'Random_Forest':
        if skopt_available:
            from skopt.space import Integer
            return {'n_estimators': Integer(100, 300), 'max_depth': Integer(10, 30), 'min_samples_split': Integer(2, 10)}
        else:
            return {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    
    elif model_name == 'Gradient_Boosting':
        if skopt_available:
            from skopt.space import Integer, Real
            return {'n_estimators': Integer(100, 300), 'learning_rate': Real(0.05, 0.2, prior='log-uniform'), 'max_depth': Integer(3, 8)}
        else:
            return {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5, 8]}
    
    # For XGBoost and LightGBM, use the original (broader) parameter space defined in Modulo 5
    return model_config['param_space']


# --- Global constant to control Nested CV ---
USE_NESTED_CV = False # Set to True to enable Nested CV (computationaly intensive)


def perform_hyperparameter_optimization(models_config: dict, balanced_datasets: dict, 
                                        cv_strategies: dict, custom_scorer, 
                                        X_val_preprocessed: np.ndarray, y_val: pd.Series,
                                        sampling_strategy_key: str) -> list:
    """
    Esegue l'ottimizzazione degli iperparametri per tutti i modelli su una specifica strategia di campionamento.
    Incorpora l'ottimizzazione della soglia.
    """
    print(f"\n--- Starting Hyperparameter Optimization for {sampling_strategy_key.upper()} Sampling Strategy ---")
    
    data_for_training = balanced_datasets[sampling_strategy_key]
    X_train_data = data_for_training['X_train']
    y_train_data = data_for_training['y_train']
    cv_strategy = cv_strategies['stratified_5fold'] # Standard stratified K-Fold for inner CV
    
    results_for_strategy = []
    
    SearchCV_Class = BayesSearchCV if SKOPT_AVAILABLE else RandomizedSearchCV
    
    print(f"   Dataset size: {len(y_train_data):,} samples, {y_train_data.mean()*100:.1f}% positive class\n")
    
    for model_name, model_config in models_config.items():
        print(f"\n{'='*50}\n🤖 TRAINING: {model_name} with {sampling_strategy_key.upper()} Data\n{'='*50}\n")
        
        try:
            start_time = time.time()
            
            n_iter = get_smart_iterations_per_model(model_name)
            focused_param_space = get_focused_param_space(model_name, model_config, SKOPT_AVAILABLE) 
            
            print(f"   🎯 Using {n_iter} iterations with focused parameter space.\n")
            
            # Create a fresh model instance for each search
            base_model_instance = model_config['model'].__class__(**model_config['model'].get_params())
            
            # --- CORREZIONE CRUCIALE: Applicazione condizionale di class_weight ---
            # Applica class_weight solo se il modello lo supporta E non è già gestito internamente
            # (XGBoost gestisce scale_pos_weight, LightGBM gestisce class_weight='balanced')
            if model_name == 'XGBoost':
                scale_pos_weight_current = (y_train_data == 0).sum() / (y_train_data == 1).sum()
                base_model_instance.set_params(scale_pos_weight=scale_pos_weight_current)
            elif model_name == 'LightGBM':
                base_model_instance.set_params(class_weight='balanced') # LightGBM può prendere 'balanced' string
            else: 
                # Per altri modelli scikit-learn (LogReg, RF, SVM)
                # verifica se il parametro class_weight è valido prima di impostarlo
                if 'class_weight' in base_model_instance.get_params():
                    from sklearn.utils.class_weight import compute_class_weight
                    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_data), y=y_train_data)
                    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
                    base_model_instance.set_params(class_weight=class_weight_dict)
                else:
                    print(f"   ⚠️ Model {model_name} does not support 'class_weight' parameter. Skipping.")

            search_params = {
                'estimator': base_model_instance,
                'scoring': custom_scorer,
                'cv': cv_strategy,
                'n_jobs': -1,
                'random_state': RANDOM_STATE,
            }

            if SKOPT_AVAILABLE:
                search_params['search_spaces'] = focused_param_space
                search_params['n_iter'] = n_iter
            else:
                search_params['param_distributions'] = focused_param_space
                search_params['n_iter'] = n_iter
                search_params['return_train_score'] = True
                
            search = SearchCV_Class(**search_params)
            search.fit(X_train_data, y_train_data)
            
            training_duration = time.time() - start_time
            
            best_model = search.best_estimator_
            
            y_val_pred_default = best_model.predict(X_val_preprocessed)
            y_val_proba = None
            if hasattr(best_model, 'predict_proba'): 
                y_val_proba = best_model.predict_proba(X_val_preprocessed)[:, 1]
            
            # --- Business Metrics and Threshold Optimization ---
            business_metrics_at_default = business_metrics_analysis(y_val, y_val_pred_default, y_val_proba)
            
            # Calculate metrics using the optimal threshold found
            optimal_threshold = business_metrics_at_default.get('optimal_threshold', 0.5)
            y_val_pred_optimal_threshold = (y_val_proba >= optimal_threshold).astype(int) if y_val_proba is not None else y_val_pred_default
            
            # Recalculate all comprehensive metrics using the optimal threshold
            validation_metrics_optimal_threshold = calculate_comprehensive_metrics(y_val, y_val_pred_optimal_threshold, y_val_proba)
            
            # Cross-validation metrics (from the search object)
            cv_score_mean = search.best_score_
            cv_scores_recheck = cross_val_score(best_model, X_train_data, y_train_data, 
                                                cv=cv_strategy, scoring=custom_scorer, n_jobs=-1)
            cv_score_std = cv_scores_recheck.std()
            
            result = {
                'model_name': model_name,
                'sampling_strategy': sampling_strategy_key, # Store the sampling strategy used
                'best_params': search.best_params_,
                'cv_score_mean': cv_score_mean,
                'cv_score_std': cv_score_std,
                'validation_metrics_default_threshold': calculate_comprehensive_metrics(y_val, y_val_pred_default, y_val_proba), # Keep metrics at default
                'validation_metrics_optimal_threshold': validation_metrics_optimal_threshold, # Metrics at optimal threshold
                'business_metrics': business_metrics_at_default, # Contains optimal threshold & max profit
                'training_time': training_duration,
                'best_model': best_model,
                'search_object': search, 
                'training_samples': len(y_train_data),
                'y_val_pred_default': y_val_pred_default,
                'y_val_pred_optimal': y_val_pred_optimal_threshold,
                'y_val_proba': y_val_proba,
                'n_iterations_used': n_iter
            }
            results_for_strategy.append(result)
            
            print(f"   ✅ Training completed in {training_duration:.1f}s")
            print(f"   📊 CV Score (Optimal Scorer): {cv_score_mean:.4f} ± {cv_score_std:.4f}")
            print(f"   🎯 Validation Metrics (Optimal Threshold={optimal_threshold:.3f}):")
            print(f"      F1-Score: {validation_metrics_optimal_threshold['f1']:.4f}")
            print(f"      ROC-AUC: {validation_metrics_optimal_threshold['roc_auc']:.4f}")
            print(f"      Precision: {validation_metrics_optimal_threshold['precision']:.4f}")
            print(f"      Recall: {validation_metrics_optimal_threshold['recall']:.4f}")
            print(f"      Accuracy: {validation_metrics_optimal_threshold['accuracy']:.4f}")
            print(f"      Balanced Accuracy: {validation_metrics_optimal_threshold['balanced_accuracy']:.4f}")
            if not np.isnan(business_metrics_at_default.get('max_profit', np.nan)):
                print(f"   💰 Max profit (at optimal threshold): {business_metrics_at_default['max_profit']:,}")
            print(f"   ⚙️ Best params: {str(search.best_params_)[:150]}{'...' if len(str(search.best_params_)) > 150 else ''}\n")
            
        except Exception as e:
            print(f"   ❌ Training failed for {model_name}: {str(e)}")
            results_for_strategy.append({
                'model_name': model_name, 'sampling_strategy': sampling_strategy_key,
                'best_params': {}, 'cv_score_mean': np.nan, 'cv_score_std': np.nan,
                'validation_metrics_default_threshold': {}, 'validation_metrics_optimal_threshold': {}, 
                'business_metrics': {'current_cost':np.nan, 'current_profit':np.nan, 'optimal_threshold':np.nan, 'max_profit':np.nan},
                'training_time': 0, 'best_model': None, 'search_object': None, 'training_samples': 0,
                'y_val_pred_default': None, 'y_val_pred_optimal': None, 'y_val_proba': None, 'n_iterations_used': 0
            })
            continue
    
    return results_for_strategy


def create_results_summary(all_results: list):
    """
    Crea un summary dettagliato dei risultati di tutti i modelli addestrati.
    I dati delle metriche useranno la soglia ottimale.
    """
    print(f"\n7. RESULTS ANALYSIS & COMPARISON")
    print("-" * 60)
    
    if not all_results:
        print("❌ No successful results to analyze!")
        return pd.DataFrame(), None 
    
    comparison_data = []
    for result in all_results:
        metrics_to_use = result.get('validation_metrics_optimal_threshold', {})
        
        row = {
            'Model': result['model_name'],
            'Sampling_Strategy': result['sampling_strategy'], # Use a more descriptive name
            'CV_Score_Mean': result['cv_score_mean'],
            'CV_Score_Std': result['cv_score_std'],
            'Training_Time': result['training_time']
        }
        # Add metrics, using .get() with np.nan as default for robustness
        for metric_name in ['f1', 'roc_auc', 'precision', 'recall', 'accuracy', 'balanced_accuracy', 'matthews_corrcoef']:
            row[metric_name] = metrics_to_use.get(metric_name, np.nan)

        row['Optimal_Threshold'] = result['business_metrics'].get('optimal_threshold', np.nan)
        row['Max_Profit'] = result['business_metrics'].get('max_profit', np.nan)
        row['Current_Cost_Default_Thresh'] = result['business_metrics'].get('current_cost', np.nan) 
        row['Current_Profit_Default_Thresh'] = result['business_metrics'].get('current_profit', np.nan)
        
        comparison_data.append(row)
    
    results_df = pd.DataFrame(comparison_data)
    # Ensure sorting handles NaNs gracefully, putting them at the end
    results_df = results_df.sort_values('f1', ascending=False, na_position='last').reset_index(drop=True)
    
    print("📊 COMPREHENSIVE RESULTS COMPARISON (Metrics at Optimal Threshold):\n")
    print("=" * 150) # Adjusted width for new column
    
    display_cols = ['Model', 'Sampling_Strategy', 'f1', 'roc_auc', 'precision', 'recall', 
                    'accuracy', 'balanced_accuracy', 'matthews_corrcoef', 
                    'Optimal_Threshold', 'Max_Profit', 'Training_Time']
    print(results_df[display_cols].round(4).to_string(index=False))
    print("=" * 150)
    
    # Identify the best overall model based on F1-score (at optimal threshold)
    # CORRECTION: Find the best_result_info by matching Model and Sampling_Strategy
    best_result_info = None
    if not results_df.empty and not pd.isna(results_df.iloc[0]['f1']): # Check if there's a valid best F1
        best_model_name_from_df = results_df.iloc[0]['Model']
        best_sampling_strategy_from_df = results_df.iloc[0]['Sampling_Strategy']
        
        # Find the full result dictionary in all_results list
        for res in all_results:
            if res['model_name'] == best_model_name_from_df and res['sampling_strategy'] == best_sampling_strategy_from_df:
                best_result_info = res
                break

    if best_result_info and best_result_info['best_model'] is not None: # Ensure the best model found is not None (e.g. from NestedCV)
        print(f"\n🥇 BEST OVERALL MODEL (based on F1-Score at Optimal Threshold):\n")
        print(f"   Model: {best_result_info['model_name']} (Sampling: {best_result_info['sampling_strategy']} sampling)")
        print(f"   F1-Score: {best_result_info['validation_metrics_optimal_threshold']['f1']:.4f}")
        print(f"   ROC-AUC: {best_result_info['validation_metrics_optimal_threshold']['roc_auc']:.4f}")
        print(f"   Precision: {best_result_info['validation_metrics_optimal_threshold']['precision']:.4f}")
        print(f"   Recall: {best_result_info['validation_metrics_optimal_threshold']['recall']:.4f}")
        print(f"   Accuracy: {best_result_info['validation_metrics_optimal_threshold']['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {best_result_info['validation_metrics_optimal_threshold']['balanced_accuracy']:.4f}")
        print(f"   Matthews Correlation: {best_result_info['validation_metrics_optimal_threshold']['matthews_corrcoef']:.4f}")
        print(f"   Optimal Threshold: {best_result_info['business_metrics'].get('optimal_threshold', np.nan):.3f}")
        print(f"   Max Profit: {best_result_info['business_metrics'].get('max_profit', np.nan):,}")
        print(f"   Training Time: {best_result_info['training_time']:.1f}s\n")
        
    else: # If no valid best model was found in results_df
        print("\n⚠️ No valid overall best model with F1-Score found to display detailed metrics.")
        best_result_info = None # Ensure it's None if no valid model was found
    
    print(f"\n🏆 MODEL RANKING (by F1-Score at Optimal Threshold):\n")
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        if pd.isna(row['f1']):
            print(f"   {i}. {row['Model']:<20} (Sampling: {row['Sampling_Strategy']:<10}) F1: N/A, ROC-AUC: N/A, Time: N/A")
        else:
            print(f"   {i}. {row['Model']:<20} (Sampling: {row['Sampling_Strategy']:<10}) F1: {row['f1']:.4f}, ROC-AUC: {row['roc_auc']:.4f}, Profit: {row['Max_Profit']:,}, Time: {row['Training_Time']:.1f}s")
    
    print(f"\n📊 PERFORMANCE STATISTICS SUMMARY:\n")
    if len(results_df) > 0 and results_df['f1'].count() > 0:
        print(f"   F1-Score Range: {results_df['f1'].min():.4f} - {results_df['f1'].max():.4f}")
        print(f"   ROC-AUC Range: {results_df['roc_auc'].min():.4f} - {results_df['roc_auc'].max():.4f}")
        print(f"   Accuracy Range: {results_df['accuracy'].min():.4f} - {results_df['accuracy'].max():.4f}\n")
    else:
        print("   No valid performance data for summary statistics.\n")

    total_training_time_all_models = results_df['Training_Time'].sum() if 'Training_Time' in results_df.columns else 0
    average_training_time_per_model = results_df['Training_Time'].mean() if 'Training_Time' in results_df.columns else 0
    print(f"   Total Training Time for all models: {total_training_time_all_models:.1f}s ({total_training_time_all_models/60:.1f} min)")
    print(f"   Average Training Time per successful model: {average_training_time_per_model:.1f}s\n")
    
    if 'Max_Profit' in results_df.columns and results_df['Max_Profit'].count() > 0:
        best_business_model = results_df.loc[results_df['Max_Profit'].idxmax()]
        print(f"💰 BEST BUSINESS MODEL (Highest Max Profit):\n")
        print(f"   Model: {best_business_model['Model']} (Sampling: {best_business_model['Sampling_Strategy']})")
        print(f"   Max Profit: {best_business_model['Max_Profit']:,}")
        print(f"   F1-Score (at Optimal Thresh): {best_business_model['f1']:.4f}")
        print(f"   Optimal Threshold: {best_business_model['Optimal_Threshold']:.3f}")
        print("\n")
    
    return results_df, best_result_info
