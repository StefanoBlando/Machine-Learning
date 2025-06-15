"""
Model configuration module.

"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, balanced_accuracy_score

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

# Bayesian optimization (with availability check)
try:
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from ..config.settings import RANDOM_STATE


def create_custom_scorer():
    """
    Crea uno scorer personalizzato che combina multiple metriche (F1, ROC-AUC, Balanced Accuracy).
    Adatta la firma della funzione per make_scorer.
    """
    # La funzione passata a make_scorer, quando needs_proba=True, riceve y_true e y_score (che sono le probabilità)
    def combined_scorer_func(y_true, y_score, **kwargs): # y_score qui saranno le probabilità
        y_pred = (y_score >= 0.5).astype(int) # Default threshold for F1/Accuracy
        
        # Ensure y_true and y_pred are not empty
        if len(y_true) == 0:
            return 0.0 # Or np.nan, depending on how you want to handle empty folds

        # Use zero_division=0 to avoid warnings/errors if a class is not predicted
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        combined_score = 0.0
        if y_score is not None and len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_score) # ROC-AUC uses probabilities
            # Weighted combination: 40% F1, 40% ROC-AUC, 20% Balanced Accuracy
            combined_score = 0.4 * f1 + 0.4 * roc_auc + 0.2 * balanced_acc
        else:
            # If probabilities not available or ROC-AUC not computable: 60% F1, 40% Balanced Accuracy
            combined_score = 0.6 * f1 + 0.4 * balanced_acc
        
        return combined_score
    
    # needs_proba=True ensures that y_score (probabilities) is passed to the scorer function
    return make_scorer(combined_scorer_func, needs_proba=True) 


def setup_advanced_models(class_weight_dict: dict, y_train_for_xgb_weight: pd.Series):
    """
    Setup di modelli tradizionali e avanzati con parametri ottimizzati
    per la ricerca degli iperparametri.
    Args:
        class_weight_dict (dict): Dizionario dei pesi delle classi per modelli sklearn.
        y_train_for_xgb_weight (pd.Series): y_train per calcolare scale_pos_weight specifico per XGBoost.
    """
    print("\n5. ADVANCED MODEL CONFIGURATION")
    print("-" * 50)
    
    models_config = {}
    
    # --- Traditional Models ---
    print("🏛️ Traditional Models:\n")
    
    # Logistic Regression
    models_config['Logistic_Regression'] = {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, class_weight=class_weight_dict, solver='liblinear'),
        'param_space': {
            'C': Real(0.001, 100, prior='log-uniform') if SKOPT_AVAILABLE else [0.01, 0.1, 1, 10, 100],
            'penalty': Categorical(['l1', 'l2']) if SKOPT_AVAILABLE else ['l1', 'l2'],
        },
        'category': 'traditional'
    }
    print("   ✅ Logistic Regression configured")
    
    # Random Forest
    models_config['Random_Forest'] = {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight=class_weight_dict, n_jobs=-1),
        'param_space': {
            'n_estimators': Integer(50, 500) if SKOPT_AVAILABLE else [100, 200, 300, 500],
            'max_depth': Integer(5, 50) if SKOPT_AVAILABLE else [None, 10, 20, 30],
            'min_samples_split': Integer(2, 20) if SKOPT_AVAILABLE else [2, 5, 10],
            'min_samples_leaf': Integer(1, 10) if SKOPT_AVAILABLE else [1, 2, 4],
            'max_features': Categorical(['sqrt', 'log2', None]) if SKOPT_AVAILABLE else ['sqrt', 'log2', None]
        },
        'category': 'traditional'
    }
    print("   ✅ Random Forest configured")
    
    # Gradient Boosting (CORREZIONE: rimosso class_weight)
    models_config['Gradient_Boosting'] = {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'param_space': {
            'n_estimators': Integer(50, 500) if SKOPT_AVAILABLE else [100, 200, 300, 500],
            'learning_rate': Real(0.01, 0.3, prior='log-uniform') if SKOPT_AVAILABLE else [0.01, 0.05, 0.1, 0.2],
            'max_depth': Integer(3, 10) if SKOPT_AVAILABLE else [3, 5, 7, 9],
            'subsample': Real(0.6, 1.0) if SKOPT_AVAILABLE else [0.8, 0.9, 1.0],
            'max_features': Categorical(['sqrt', 'log2', None]) if SKOPT_AVAILABLE else ['sqrt', 'log2', None]
        },
        'category': 'traditional'
    }
    print("   ✅ Gradient Boosting configured")
    
    # Support Vector Machine (SVC)
    models_config['SVM'] = {
        'model': SVC(random_state=RANDOM_STATE, probability=True, class_weight=class_weight_dict),
        'param_space': {
            'C': Real(0.1, 100, prior='log-uniform') if SKOPT_AVAILABLE else [0.1, 1, 10, 100],
            'kernel': Categorical(['linear', 'rbf']) if SKOPT_AVAILABLE else ['linear', 'rbf'], 
            'gamma': Categorical(['scale', 'auto']) if SKOPT_AVAILABLE else ['scale', 'auto']
        },
        'category': 'traditional'
    }
    print("   ✅ SVM configured")
    
    # --- Advanced Models ---
    print("\n🚀 Advanced Models:\n")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        scale_pos_weight_val = (y_train_for_xgb_weight == 0).sum() / (y_train_for_xgb_weight == 1).sum() 
        
        models_config['XGBoost'] = {
            'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', 
                                   use_label_encoder=False, 
                                   scale_pos_weight=scale_pos_weight_val,
                                   n_jobs=-1, verbosity=0), 
            'param_space': {
                'n_estimators': Integer(50, 1000) if SKOPT_AVAILABLE else [100, 200, 300, 500],
                'learning_rate': Real(0.01, 0.3, prior='log-uniform') if SKOPT_AVAILABLE else [0.01, 0.05, 0.1, 0.2],
                'max_depth': Integer(3, 12) if SKOPT_AVAILABLE else [3, 5, 7, 9],
                'subsample': Real(0.6, 1.0) if SKOPT_AVAILABLE else [0.8, 0.9, 1.0],
                'colsample_bytree': Real(0.6, 1.0) if SKOPT_AVAILABLE else [0.8, 0.9, 1.0],
                'reg_alpha': Real(1e-8, 10, prior='log-uniform') if SKOPT_AVAILABLE else [0, 0.1, 1], 
                'reg_lambda': Real(1e-8, 10, prior='log-uniform') if SKOPT_AVAILABLE else [1, 1.5, 2] 
            },
            'category': 'advanced'
        }
        print("   ✅ XGBoost configured")
    else:
        print("   ❌ XGBoost not available, skipping configuration.")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        models_config['LightGBM'] = {
            'model': LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced', verbosity=-1, n_jobs=-1), 
            'param_space': {
                'n_estimators': Integer(50, 1000) if SKOPT_AVAILABLE else [100, 200, 300, 500],
                'learning_rate': Real(0.01, 0.3, prior='log-uniform') if SKOPT_AVAILABLE else [0.01, 0.05, 0.1, 0.2],
                'max_depth': Integer(3, 15) if SKOPT_AVAILABLE else [3, 5, 7, 9],
                'num_leaves': Integer(10, 200) if SKOPT_AVAILABLE else [15, 31, 63, 127], 
                'subsample': Real(0.6, 1.0) if SKOPT_AVAILABLE else [0.8, 0.9, 1.0],
                'colsample_bytree': Real(0.6, 1.0) if SKOPT_AVAILABLE else [0.8, 0.9, 1.0],
                'reg_alpha': Real(1e-8, 10, prior='log-uniform') if SKOPT_AVAILABLE else [0, 0.1, 1],
                'reg_lambda': Real(1e-8, 10, prior='log-uniform') if SKOPT_AVAILABLE else [1, 1.5, 2]
            },
            'category': 'advanced'
        }
        print("   ✅ LightGBM configured")
    else:
        print("   ❌ LightGBM not available, skipping configuration.")
    
    # --- Overall Summary ---
    traditional_count = sum(1 for config in models_config.values() if config['category'] == 'traditional')
    advanced_count = sum(1 for config in models_config.values() if config['category'] == 'advanced')
    
    print(f"\n📊 Model Configuration Summary:\n")
    print(f"   Traditional models: {traditional_count}")
    print(f"   Advanced models: {advanced_count}")
    print(f"   Total models configured: {len(models_config)}")
    
    optimization_strategy = "BayesianOptimization" if SKOPT_AVAILABLE else "RandomizedSearch"
    print(f"   Optimization strategy: {optimization_strategy}")
    
    return models_config


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
