"""
Optuna optimization module.

"""
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Optuna for advanced methodological comparison (checking availability)
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Advanced models (checking availability)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..config.settings import RANDOM_STATE
from ..evaluation.metrics import calculate_comprehensive_metrics, business_metrics_analysis


# --- Re-use metric functions from Modulo 6 ---
def calculate_comprehensive_metrics_optuna(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    return calculate_comprehensive_metrics(y_true, y_pred, y_proba)

def business_metrics_analysis_optuna(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    return business_metrics_analysis(y_true, y_pred, y_proba)

# --- Expanded parameter suggestion functions for Optuna ---
# These offer a wider search space than the focused spaces in Modulo 6
def suggest_xgboost_params_expanded(trial: optuna.Trial, scale_pos_weight_val: float):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 1e-8, 1, log=True),
        'scale_pos_weight': scale_pos_weight_val 
    }

def suggest_lightgbm_params_expanded(trial: optuna.Trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True),
        'class_weight': 'balanced'
    }

def suggest_rf_params_expanded(trial: optuna.Trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20, 30, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.params.get('bootstrap', True) else None
    }

def suggest_gb_params_expanded(trial: optuna.Trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]), # Categorical param
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
    }

def suggest_svm_params_expanded(trial: optuna.Trial):
    return {
        'C': trial.suggest_float('C', 0.01, 100, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']), # Categorical param
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) # Categorical param
    }

def suggest_lr_params_expanded(trial: optuna.Trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2']) # Categorical param
    params = {
        'C': trial.suggest_float('C', 1e-4, 100, log=True),
        'penalty': penalty,
        'solver': 'liblinear' # Hardcode for compatibility with l1/l2
    }
    return params

def run_optuna_methodology_complete(model_name: str, X_train_optuna: np.ndarray, y_train_optuna: pd.Series, 
                                  cv_strategy_optuna, custom_scorer_optuna, X_val_optuna: np.ndarray, y_val_optuna: pd.Series, 
                                  sampler_type: str = 'TPE', n_trials: int = 25, use_pruning: bool = False, 
                                  timeout_seconds: int = 300, class_weight_dict: dict = None) -> dict:
    """
    Esegue una specifica metodologia Optuna con validazione completa e gestisce il timeout.
    """
    start_time = time.time()
    
    # Setup Sampler
    sampler = None
    if sampler_type == 'TPE':
        sampler = optuna.samplers.TPESampler(n_startup_trials=min(8, n_trials//3), multivariate=True, seed=RANDOM_STATE)
    elif sampler_type == 'CMA-ES':
        # CORREZIONE: Saltare CMA-ES per modelli con parametri categorici nell'expanded space
        # CMA-ES non gestisce parametri categorici.
        models_with_categorical_params_in_expanded_space = [
            'Random_Forest', 'Gradient_Boosting', 'SVM', 'Logistic_Regression'
        ]
        
        if model_name in models_with_categorical_params_in_expanded_space:
            print(f"   ⚠️ CMA-ES skipped for {model_name} because its expanded search space contains categorical parameters.")
            return None
        
        sampler = optuna.samplers.CmaEsSampler(seed=RANDOM_STATE)

    elif sampler_type == 'Random':
        sampler = optuna.samplers.RandomSampler(seed=RANDOM_STATE)
    else: # Default to TPE
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    
    # Setup Pruner
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3) if use_pruning else optuna.pruners.NopPruner()
    
    # Create Study
    # Added timestamp for unique name to avoid conflicts if multiple runs/restarts
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name}_{sampler_type}_{'Pruned' if use_pruning else 'NoPrune'}_{datetime.now().strftime('%H%M%S%f')}" 
    )
    
    # Define Objective Function for Optuna
    def objective(trial: optuna.Trial):
        model = None
        params = {}
        if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            scale_pos_weight_for_xgb = (y_train_optuna == 0).sum() / (y_train_optuna == 1).sum() 
            params = suggest_xgboost_params_expanded(trial, scale_pos_weight_for_xgb)
            model = XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, use_label_encoder=False)
        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            params = suggest_lightgbm_params_expanded(trial)
            model = LGBMClassifier(**params, random_state=RANDOM_STATE, verbosity=-1)
        elif model_name == 'Random_Forest':
            params = suggest_rf_params_expanded(trial)
            model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=1) # n_jobs=1 for Optuna CV
        elif model_name == 'Gradient_Boosting':
            params = suggest_gb_params_expanded(trial)
            model = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
        elif model_name == 'SVM':
            params = suggest_svm_params_expanded(trial)
            model = SVC(**params, random_state=RANDOM_STATE, probability=True)
        elif model_name == 'Logistic_Regression':
            # CORREZIONE: Passare 'solver' solo tramite **params, poiché è già incluso in suggest_lr_params_expanded
            params = suggest_lr_params_expanded(trial)
            model = LogisticRegression(**params, random_state=RANDOM_STATE, max_iter=2000) 
        else:
            raise ValueError(f"Model {model_name} not supported or not available.")
        
        # --- CORREZIONE: Applicazione condizionale di class_weight ---
        if model_name not in ['XGBoost', 'LightGBM']:
            if 'class_weight' in model.get_params() and class_weight_dict is not None:
                model.set_params(class_weight=class_weight_dict)

        # Cross-validation within Optuna objective
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy_optuna.split(X_train_optuna, y_train_optuna)):
            X_fold_train = X_train_optuna[train_idx]
            X_fold_val = X_train_optuna[val_idx]
            y_fold_train = y_train_optuna.iloc[train_idx] if hasattr(y_train_optuna, 'iloc') else y_train_optuna[train_idx]
            y_fold_val = y_train_optuna.iloc[val_idx] if hasattr(y_train_optuna, 'iloc') else y_train_optuna[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            score = custom_scorer_optuna(model, X_fold_val, y_fold_val) # Use the custom scorer
            scores.append(score)
            
            # Report for pruning
            trial.report(score, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return np.mean(scores)
    
    # Optimize the objective function
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)
        
        training_time = time.time() - start_time
        
        # Get best model and validate on holdout
        best_params = study.best_params
        best_cv_score = study.best_value
        
        # Create final model with best parameters
        if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            scale_pos_weight_for_xgb = (y_train_optuna == 0).sum() / (y_train_optuna == 1).sum()
            final_model = XGBClassifier(**best_params, random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, use_label_encoder=False)
        elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            final_model = LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbosity=-1)
        elif model_name == 'Random_Forest':
            final_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
        elif model_name == 'Gradient_Boosting':
            final_model = GradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
        elif model_name == 'SVM':
            final_model = SVC(**best_params, random_state=RANDOM_STATE, probability=True, class_weight='balanced')
        elif model_name == 'Logistic_Regression':
            # Ensures class_weight is passed here too
            final_model = LogisticRegression(**best_params, random_state=RANDOM_STATE, max_iter=2000, class_weight='balanced') 
        
        # Train on full training set (X_train_optuna, y_train_optuna)
        final_model.fit(X_train_optuna, y_train_optuna)
        
        # Validate on holdout set (X_val_optuna, y_val_optuna)
        y_val_pred = final_model.predict(X_val_optuna)
        y_val_proba = final_model.predict_proba(X_val_optuna)[:, 1] if hasattr(final_model, 'predict_proba') else None
        
        # Calculate comprehensive metrics
        validation_metrics = calculate_comprehensive_metrics_optuna(y_val_optuna, y_val_pred, y_val_proba)
        business_metrics = business_metrics_analysis_optuna(y_val_optuna, y_val_pred, y_val_proba)
        
        # Results
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        print(f"   ✅ {sampler_type} (Pruning: {'Yes' if use_pruning else 'No'}): Completed in {training_time:.1f}s")
        print(f"      CV Score: {best_cv_score:.4f}")
        print(f"      Validation F1: {validation_metrics['f1']:.4f}")
        print(f"      Validation ROC-AUC: {validation_metrics['roc_auc']:.4f}")
        print(f"      Trials: {n_completed} completed, {n_pruned} pruned")
        
        return {
            'approach': f'Optuna {sampler_type}',
            'validation_metrics': validation_metrics,
            'business_metrics': business_metrics,
            'cv_score': best_cv_score,
            'time': training_time,
            'trials': n_completed,
            'pruned_trials': n_pruned,
            'best_params': best_params,
            'study': study, 
            'final_model': final_model,
            'description': f'{sampler_type} sampler + {"Hyperband pruning" if use_pruning else "No pruning"}',
            'X_train_data_shape': X_train_optuna.shape, 
            'y_val_pred': y_val_pred,
            'y_val_proba': y_val_proba
        }
        
    except optuna.exceptions.TrialPruned:
        print(f"   ⚠️ {sampler_type} (Pruning: {'Yes' if use_pruning else 'No'}): Trial pruned.")
        return None 
    except Exception as e:
        print(f"   ❌ {sampler_type} failed: {str(e)}")
        return None

def methodological_comparison_study(best_result_from_modulo6: dict, balanced_datasets: dict, cv_strategies: dict, custom_scorer, X_val_preprocessed: np.ndarray, y_val: pd.Series, class_weight_dict: dict = None) -> dict:
    """
    Studio comparativo metodologico: Traditional vs Optuna approaches.
    Si concentra sul modello che ha ottenuto le migliori performance nel Modulo 6.
    """
    print("\n6B. METHODOLOGICAL DEEP DIVE WITH OPTUNA")
    print("=" * 60)
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna non disponibile. Skipping methodological analysis.")
        return None
    
    if not best_result_from_modulo6 or best_result_from_modulo6['best_model'] is None:
        print("❌ Nessun modello valido trovato nel Modulo 6 per l'analisi metodologica. Esegui prima il modulo 6.")
        return None
    
    best_model_name = best_result_from_modulo6['model_name']
    
    print(f"🎯 Approfondimento metodologico su: {best_model_name} (Best Model from Module 6)")
    print(f"📈 Performance baseline dal Modulo 6 (Optimal Threshold Metrics):\n")
    print(f"   F1-Score: {best_result_from_modulo6['validation_metrics_optimal_threshold']['f1']:.4f}")
    print(f"   ROC-AUC: {best_result_from_modulo6['validation_metrics_optimal_threshold']['roc_auc']:.4f}")
    print(f"   Accuracy: {best_result_from_modulo6['validation_metrics_optimal_threshold']['accuracy']:.4f}")
    print(f"   Training Time: {best_result_from_modulo6['training_time']:.1f}s")
    print(f"   Iterations Used: {best_result_from_modulo6.get('n_iterations_used', 'N/A')}\n")
    
    # Prepare data for Optuna: Use the training data that the best model from Module 6 was trained on
    # It's crucial to use the SAME preprocessed (e.g., feature-selected) data
    sampling_key_for_best_model = best_result_from_modulo6['sampling_strategy']
    
    X_train_for_optuna = balanced_datasets[sampling_key_for_best_model]['X_train']
    y_train_for_optuna = balanced_datasets[sampling_key_for_best_model]['y_train']
    
    # Use validation data for final evaluation
    X_val_for_optuna = X_val_preprocessed
    y_val_for_optuna = y_val
    
    cv_strategy_optuna = cv_strategies['stratified_5fold']
    
    methodological_results = {}
    
    # 1. BASELINE (from Modulo 6 - the best result already obtained)
    methodological_results['Traditional_Optimized'] = {
        'approach': f'Traditional ({best_result_from_modulo6.get("optimization_method", "Search")})',
        'validation_metrics': best_result_from_modulo6['validation_metrics_optimal_threshold'],
        'business_metrics': best_result_from_modulo6['business_metrics'],
        'cv_score': best_result_from_modulo6['cv_score_mean'],
        'time': best_result_from_modulo6['training_time'],
        'trials': best_result_from_modulo6.get('n_iterations_used', 10),
        'pruned_trials': 0, # Not applicable for non-Optuna search
        'best_params': best_result_from_modulo6['best_params'],
        'study': None,
        'final_model': best_result_from_modulo6['best_model'],
        'description': f'Focused parameters + {best_result_from_modulo6.get("n_iterations_used", 10)} iterations on {sampling_key_for_best_model} data',
        'X_train_data_shape': X_train_for_optuna.shape,
        'y_val_pred': best_result_from_modulo6['y_val_pred_optimal'],
        'y_val_proba': best_result_from_modulo6['y_val_proba']
    }
    
    # 2. OPTUNA TPE
    print(f"\n🧠 Testing TPE (Tree-structured Parzen Estimator)...\n")
    tpe_result = run_optuna_methodology_complete(
        best_model_name, X_train_for_optuna, y_train_for_optuna, cv_strategy_optuna, custom_scorer,
        X_val_for_optuna, y_val_for_optuna, sampler_type='TPE', n_trials=25, class_weight_dict=class_weight_dict # Reduced trials for faster execution
    )
    if tpe_result:
        methodological_results['Optuna_TPE'] = tpe_result
    
    # 3. OPTUNA CMA-ES (only for models with no categorical parameters in the expanded space)
    # This check is done inside run_optuna_methodology_complete now.
    print(f"\n🔬 Testing CMA-ES (Covariance Matrix Adaptation)...\n")
    cmaes_result = run_optuna_methodology_complete(
        best_model_name, X_train_for_optuna, y_train_for_optuna, cv_strategy_optuna, custom_scorer,
        X_val_for_optuna, y_val_for_optuna, sampler_type='CMA-ES', n_trials=25, class_weight_dict=class_weight_dict
    )
    if cmaes_result:
        methodological_results['Optuna_CMAES'] = cmaes_result
    
    # 4. OPTUNA Random (baseline for Optuna)
    print(f"\n🎲 Testing Random Search (Optuna baseline)...\n")
    random_result = run_optuna_methodology_complete(
        best_model_name, X_train_for_optuna, y_train_for_optuna, cv_strategy_optuna, custom_scorer,
        X_val_for_optuna, y_val_for_optuna, sampler_type='Random', n_trials=25, class_weight_dict=class_weight_dict
    )
    if random_result:
        methodological_results['Optuna_Random'] = random_result
    
    # 5. OPTUNA TPE with advanced Pruning
    print(f"\n✂️ Testing TPE + Hyperband Pruning...\n")
    tpe_pruned_result = run_optuna_methodology_complete(
        best_model_name, X_train_for_optuna, y_train_for_optuna, cv_strategy_optuna, custom_scorer,
        X_val_for_optuna, y_val_for_optuna, sampler_type='TPE', n_trials=40, use_pruning=True, class_weight_dict=class_weight_dict # More trials with pruning
    )
    if tpe_pruned_result:
        methodological_results['Optuna_TPE_Pruned'] = tpe_pruned_result
    
    return methodological_results

def create_methodological_analysis_final(methodological_results: dict):
    """
    Analisi finale completa dei risultati metodologici di ottimizzazione.
    """
    if not methodological_results or not OPTUNA_AVAILABLE:
        print("❌ No methodological results to analyze or Optuna not available.")
        return None
    
    print(f"\n🔬 FINAL METHODOLOGICAL ANALYSIS")
    print("=" * 80)
    
    comparison_data = []
    for method_name, result in methodological_results.items():
        if result is None: # Skip failed or pruned trials
            continue
        row = {
            'Method': result['approach'],
            'F1_Score': result['validation_metrics']['f1'],
            'ROC_AUC': result['validation_metrics']['roc_auc'],
            'Accuracy': result['validation_metrics']['accuracy'],
            'Precision': result['validation_metrics']['precision'],
            'Recall': result['validation_metrics']['recall'],
            'Time_Minutes': result['time'] / 60,
            'Trials': result['trials'],
            'Efficiency': result['validation_metrics']['f1'] / (result['time'] / 60) if result['time'] > 0 else 0, 
            'Description': result['description']
        }
        if 'optimal_threshold' in result['business_metrics'] and not np.isnan(result['business_metrics']['optimal_threshold']):
            row['Optimal_Threshold'] = result['business_metrics']['optimal_threshold']
        if 'max_profit' in result['business_metrics'] and not np.isnan(result['business_metrics']['max_profit']):
            row['Max_Profit'] = result['business_metrics']['max_profit']
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    if comparison_df.empty:
        print("No valid methodological comparison data generated.")
        return None

    comparison_df = comparison_df.sort_values('F1_Score', ascending=False).reset_index(drop=True)
    
    print("📊 COMPREHENSIVE METHODOLOGICAL COMPARISON:\n")
    print("-" * 120)
    display_cols = ['Method', 'F1_Score', 'ROC_AUC', 'Accuracy', 'Time_Minutes', 'Trials', 'Efficiency']
    print(comparison_df[display_cols].round(4).to_string(index=False))
    print("-" * 120)
    
    # Best method analysis
    best_method = comparison_df.iloc[0]
    baseline_method = comparison_df[comparison_df['Method'].str.contains('Traditional')].iloc[0] # Assumes 'Traditional_Optimized' exists
    
    print(f"\n🏆 METHODOLOGICAL INSIGHTS:\n")
    print(f"   🥇 Best Method: {best_method['Method']}")
    print(f"   📈 Performance vs Traditional Baseline ({baseline_method['Method']}):")
    
    f1_improvement = ((best_method['F1_Score'] - baseline_method['F1_Score']) / baseline_method['F1_Score']) * 100 if baseline_method['F1_Score'] > 0 else 0
    time_change = ((best_method['Time_Minutes'] - baseline_method['Time_Minutes']) / baseline_method['Time_Minutes']) * 100 if baseline_method['Time_Minutes'] > 0 else 0
    
    print(f"      F1-Score Improvement: {f1_improvement:+.2f}%")
    print(f"      Time Change: {time_change:+.1f}%")
    print(f"      Trials: {best_method['Trials']} vs {int(baseline_method['Trials'])}\n")
    
    # Efficiency analysis
    efficiency_leader = comparison_df.loc[comparison_df['Efficiency'].idxmax()]
    print(f"   ⚡ Most Efficient Method (F1/minute): {efficiency_leader['Method']}")
    print(f"      Efficiency: {efficiency_leader['Efficiency']:.4f} F1/minute\n")
    
    # Methodology recommendations
    print(f"💡 METHODOLOGY RECOMMENDATIONS:\n")
    if best_method['Method'] != baseline_method['Method'] and f1_improvement > 0.5: 
        print(f"   ✅ Recommended: {best_method['Method']} for its balance of performance and efficiency.")
        print(f"   🎯 Justification: Achieved a {f1_improvement:+.2f}% F1 improvement with a {time_change:+.1f}% change in time.")
    else:
        print(f"   ✅ Traditional optimization (e.g., Randomized Search / BayesSearchCV) is sufficient for this problem, as Optuna did not show significant gains in this test.")
    
    # Statistical significance estimate
    if comparison_df['F1_Score'].count() > 1:
        f1_std = comparison_df['F1_Score'].std()
        print(f"\n📊 PERFORMANCE STATISTICS (across methodologies):\n")
        print(f"   F1-Score Range: {comparison_df['F1_Score'].min():.4f} - {comparison_df['F1_Score'].max():.4f}")
        print(f"   F1-Score Std Dev: {f1_std:.4f}")
        print(f"   Time Range: {comparison_df['Time_Minutes'].min():.1f} - {comparison_df['Time_Minutes'].max():.1f} minutes")
    
    return comparison_df


# --- Global constant to control Nested CV ---
USE_NESTED_CV = False # Set to True to enable Nested CV (computationaly intensive)

# --- Execute methodological study if previous results available ---
def execute_optuna_study_if_available(best_result, balanced_datasets, cv_strategies, custom_scorer, X_val_preprocessed, y_val, class_weight_dict=None):
    """
    Execute methodological study if previous results available.
    Estratto ESATTAMENTE dal blocco di esecuzione del MODULO 6B nel notebook.
    """
    # Ensure best_result, balanced_datasets, cv_strategies, custom_scorer, X_val_preprocessed, y_val are available
    if OPTUNA_AVAILABLE and best_result and \
       balanced_datasets and cv_strategies and \
       custom_scorer and X_val_preprocessed is not None and y_val is not None:
        
        # Check if a best_result was actually found (not None due to training failures)
        if best_result['best_model'] is not None:
            print("\n🚀 Starting methodological comparison with Optuna...")
            
            methodological_results = methodological_comparison_study(
                best_result, balanced_datasets, cv_strategies, custom_scorer, 
                X_val_preprocessed, y_val, class_weight_dict
            )
            
            if methodological_results:
                comparison_df_methodology = create_methodological_analysis_final(methodological_results)
                print(f"\n✅ Methodological analysis completed!")
                print(f"🎯 Results available in 'methodological_results' and 'comparison_df_methodology' variables.")
                return methodological_results, comparison_df_methodology
            else:
                print(f"⚠️ Methodological analysis could not be completed.")
                return None, None
        else:
            print(f"⚠️ No best model found from Module 6 to perform methodological analysis.")
            return None, None
    else:
        print(f"⚠️ Optuna not available or Module 6 results not available for methodological analysis.")
        return None, None
