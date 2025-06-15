"""
Traditional machine learning models configuration.
Extracted from Module 5 of the original notebook - Traditional Models section.
"""
import logging
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Bayesian optimization imports (with availability check)
try:
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from ..config.settings import RANDOM_STATE, TRADITIONAL_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_traditional_models(class_weight_dict: Dict[int, float]) -> Dict[str, Dict[str, Any]]:
    """
    Setup traditional machine learning models with their configurations.
    
    Args:
        class_weight_dict (Dict[int, float]): Class weights for balancing
        
    Returns:
        Dict[str, Dict[str, Any]]: Model configurations
    """
    logger.info("🏛️ Setting up Traditional Models")
    logger.info("-" * 30)
    
    models_config = {}
    
    # Logistic Regression
    if 'Logistic_Regression' in TRADITIONAL_MODELS:
        models_config['Logistic_Regression'] = {
            'model': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=2000,
                class_weight=class_weight_dict,
                solver='liblinear'
            ),
            'param_space': _get_logistic_regression_params(),
            'category': 'traditional',
            'description': 'Linear model with L1/L2 regularization'
        }
        logger.info("   ✅ Logistic Regression configured")
    
    # Random Forest
    if 'Random_Forest' in TRADITIONAL_MODELS:
        models_config['Random_Forest'] = {
            'model': RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight=class_weight_dict,
                n_jobs=-1
            ),
            'param_space': _get_random_forest_params(),
            'category': 'traditional',
            'description': 'Ensemble of decision trees'
        }
        logger.info("   ✅ Random Forest configured")
    
    # Gradient Boosting
    if 'Gradient_Boosting' in TRADITIONAL_MODELS:
        models_config['Gradient_Boosting'] = {
            'model': GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),
            'param_space': _get_gradient_boosting_params(),
            'category': 'traditional',
            'description': 'Sequential boosting algorithm'
        }
        logger.info("   ✅ Gradient Boosting configured")
    
    # Support Vector Machine
    if 'SVM' in TRADITIONAL_MODELS:
        models_config['SVM'] = {
            'model': SVC(
                random_state=RANDOM_STATE,
                probability=True,
                class_weight=class_weight_dict
            ),
            'param_space': _get_svm_params(),
            'category': 'traditional',
            'description': 'Support Vector Machine with RBF/Linear kernels'
        }
        logger.info("   ✅ SVM configured")
    
    logger.info(f"📊 Traditional Model Summary:")
    logger.info(f"   Models configured: {len(models_config)}")
    logger.info(f"   Optimization strategy: {'Bayesian' if SKOPT_AVAILABLE else 'RandomizedSearch'}")
    
    return models_config


def _get_logistic_regression_params() -> Dict[str, Any]:
    """Get parameter space for Logistic Regression."""
    if SKOPT_AVAILABLE:
        return {
            'C': Real(0.001, 100, prior='log-uniform'),
            'penalty': Categorical(['l1', 'l2'])
        }
    else:
        return {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }


def _get_random_forest_params() -> Dict[str, Any]:
    """Get parameter space for Random Forest."""
    if SKOPT_AVAILABLE:
        return {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None])
        }
    else:
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }


def _get_gradient_boosting_params() -> Dict[str, Any]:
    """Get parameter space for Gradient Boosting."""
    if SKOPT_AVAILABLE:
        return {
            'n_estimators': Integer(50, 500),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.6, 1.0),
            'max_features': Categorical(['sqrt', 'log2', None])
        }
    else:
        return {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }


def _get_svm_params() -> Dict[str, Any]:
    """Get parameter space for SVM."""
    if SKOPT_AVAILABLE:
        return {
            'C': Real(0.1, 100, prior='log-uniform'),
            'kernel': Categorical(['linear', 'rbf']),
            'gamma': Categorical(['scale', 'auto'])
        }
    else:
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }


def get_focused_traditional_params(model_name: str, model_config: Dict) -> Dict[str, Any]:
    """
    Get focused parameter spaces for faster hyperparameter search.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict): Original model configuration
        
    Returns:
        Dict[str, Any]: Focused parameter space
    """
    if model_name == 'Logistic_Regression':
        if SKOPT_AVAILABLE:
            return {
                'C': Real(0.1, 10, prior='log-uniform'),
                'penalty': Categorical(['l1', 'l2'])
            }
        else:
            return {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
    
    elif model_name == 'Random_Forest':
        if SKOPT_AVAILABLE:
            return {
                'n_estimators': Integer(100, 300),
                'max_depth': Integer(10, 30),
                'min_samples_split': Integer(2, 10)
            }
        else:
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
    
    elif model_name == 'Gradient_Boosting':
        if SKOPT_AVAILABLE:
            return {
                'n_estimators': Integer(100, 300),
                'learning_rate': Real(0.05, 0.2, prior='log-uniform'),
                'max_depth': Integer(3, 8)
            }
        else:
            return {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 8]
            }
    
    elif model_name == 'SVM':
        if SKOPT_AVAILABLE:
            return {
                'C': Real(0.1, 10, prior='log-uniform'),
                'kernel': Categorical(['rbf']),
                'gamma': Categorical(['scale', 'auto'])
            }
        else:
            return {
                'C': [0.1, 1, 10],
                'kernel': ['rbf'],
                'gamma': ['scale', 'auto']
            }
    
    # Return original parameter space if no focused version available
    return model_config['param_space']


def validate_traditional_model_config(models_config: Dict[str, Dict]) -> bool:
    """
    Validate that all traditional models are properly configured.
    
    Args:
        models_config (Dict[str, Dict]): Model configurations
        
    Returns:
        bool: True if all configurations are valid
    """
    required_keys = ['model', 'param_space', 'category', 'description']
    
    for model_name, config in models_config.items():
        # Check required keys
        for key in required_keys:
            if key not in config:
                logger.error(f"❌ Missing key '{key}' in {model_name} configuration")
                return False
        
        # Check model instance
        if not hasattr(config['model'], 'fit'):
            logger.error(f"❌ Invalid model instance for {model_name}")
            return False
        
        # Check parameter space
        if not isinstance(config['param_space'], dict):
            logger.error(f"❌ Invalid parameter space for {model_name}")
            return False
        
        # Check category
        if config['category'] != 'traditional':
            logger.warning(f"⚠️ Unexpected category for {model_name}: {config['category']}")
    
    logger.info("✅ All traditional model configurations are valid")
    return True


def get_traditional_model_info() -> Dict[str, str]:
    """
    Get information about traditional models.
    
    Returns:
        Dict[str, str]: Model information
    """
    return {
        'Logistic_Regression': {
            'description': 'Linear classifier with L1/L2 regularization',
            'pros': ['Fast training', 'Interpretable', 'Good baseline'],
            'cons': ['Assumes linear relationships', 'May need feature scaling'],
            'best_for': 'Linear separable data, baseline model'
        },
        'Random_Forest': {
            'description': 'Ensemble of decision trees with bagging',
            'pros': ['Handles non-linear data', 'Feature importance', 'Robust to outliers'],
            'cons': ['Can overfit', 'Less interpretable than single tree'],
            'best_for': 'Non-linear data, mixed feature types'
        },
        'Gradient_Boosting': {
            'description': 'Sequential ensemble that corrects previous errors',
            'pros': ['High accuracy', 'Handles complex patterns', 'Feature importance'],
            'cons': ['Prone to overfitting', 'Sensitive to hyperparameters'],
            'best_for': 'Complex patterns, when accuracy is priority'
        },
        'SVM': {
            'description': 'Finds optimal separating hyperplane',
            'pros': ['Effective in high dimensions', 'Memory efficient', 'Kernel trick'],
            'cons': ['Slow on large datasets', 'Sensitive to scaling'],
            'best_for': 'High-dimensional data, complex boundaries'
        }
    }
