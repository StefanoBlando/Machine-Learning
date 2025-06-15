"""
Advanced machine learning models configuration.
Extracted from Module 5 of the original notebook - Advanced Models section.
"""
import logging
import pandas as pd
from typing import Dict, Any, Optional

# Advanced model imports (with availability checks)
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

# Bayesian optimization imports (with availability check)
try:
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from ..config.settings import RANDOM_STATE, ADVANCED_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_advanced_models(y_train_for_xgb_weight: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Setup advanced machine learning models with their configurations.
    
    Args:
        y_train_for_xgb_weight (pd.Series): Training target for XGBoost weight calculation
        
    Returns:
        Dict[str, Dict[str, Any]]: Model configurations
    """
    logger.info("🚀 Setting up Advanced Models")
    logger.info("-" * 30)
    
    models_config = {}
    
    # XGBoost
    if 'XGBoost' in ADVANCED_MODELS and XGBOOST_AVAILABLE:
        scale_pos_weight_val = (y_train_for_xgb_weight == 0).sum() / (y_train_for_xgb_weight == 1).sum()
        
        models_config['XGBoost'] = {
            'model': XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight_val,
                n_jobs=-1,
                verbosity=0
            ),
            'param_space': _get_xgboost_params(),
            'category': 'advanced',
            'description': 'Extreme Gradient Boosting with optimized performance',
            'scale_pos_weight': scale_pos_weight_val
        }
        logger.info(f"   ✅ XGBoost configured (scale_pos_weight: {scale_pos_weight_val:.3f})")
    elif 'XGBoost' in ADVANCED_MODELS and not XGBOOST_AVAILABLE:
        logger.warning("   ❌ XGBoost not available, skipping configuration.")
    
    # LightGBM
    if 'LightGBM' in ADVANCED_MODELS and LIGHTGBM_AVAILABLE:
        models_config['LightGBM'] = {
            'model': LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight='balanced',
                verbosity=-1,
                n_jobs=-1
            ),
            'param_space': _get_lightgbm_params(),
            'category': 'advanced',
            'description': 'Microsoft\'s efficient gradient boosting framework'
        }
        logger.info("   ✅ LightGBM configured")
    elif 'LightGBM' in ADVANCED_MODELS and not LIGHTGBM_AVAILABLE:
        logger.warning("   ❌ LightGBM not available, skipping configuration.")
    
    logger.info(f"📊 Advanced Model Summary:")
    logger.info(f"   Models configured: {len(models_config)}")
    logger.info(f"   XGBoost available: {'✅' if XGBOOST_AVAILABLE else '❌'}")
    logger.info(f"   LightGBM available: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
    logger.info(f"   Optimization strategy: {'Bayesian' if SKOPT_AVAILABLE else 'RandomizedSearch'}")
    
    return models_config


def _get_xgboost_params() -> Dict[str, Any]:
    """Get parameter space for XGBoost."""
    if SKOPT_AVAILABLE:
        return {
            'n_estimators': Integer(50, 1000),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 12),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'reg_alpha': Real(1e-8, 10, prior='log-uniform'),
            'reg_lambda': Real(1e-8, 10, prior='log-uniform')
        }
    else:
        return {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }


def _get_lightgbm_params() -> Dict[str, Any]:
    """Get parameter space for LightGBM."""
    if SKOPT_AVAILABLE:
        return {
            'n_estimators': Integer(50, 1000),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 15),
            'num_leaves': Integer(10, 200),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'reg_alpha': Real(1e-8, 10, prior='log-uniform'),
            'reg_lambda': Real(1e-8, 10, prior='log-uniform')
        }
    else:
        return {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }


def get_focused_advanced_params(model_name: str, model_config: Dict) -> Dict[str, Any]:
    """
    Get focused parameter spaces for faster hyperparameter search.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict): Original model configuration
        
    Returns:
        Dict[str, Any]: Focused parameter space
    """
    if model_name == 'XGBoost':
        if SKOPT_AVAILABLE:
            return {
                'n_estimators': Integer(100, 500),
                'learning_rate': Real(0.05, 0.2, prior='log-uniform'),
                'max_depth': Integer(3, 8),
                'subsample': Real(0.8, 1.0)
            }
        else:
            return {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 6, 8],
                'subsample': [0.8, 0.9, 1.0]
            }
    
    elif model_name == 'LightGBM':
        if SKOPT_AVAILABLE:
            return {
                'n_estimators': Integer(100, 500),
                'learning_rate': Real(0.05, 0.2, prior='log-uniform'),
                'max_depth': Integer(3, 10),
                'num_leaves': Integer(20, 100)
            }
        else:
            return {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 10],
                'num_leaves': [31, 63, 100]
            }
    
    # Return original parameter space if no focused version available
    return model_config['param_space']


def update_xgboost_weight(model_config: Dict, y_train: pd.Series) -> Dict:
    """
    Update XGBoost scale_pos_weight based on current training data.
    
    Args:
        model_config (Dict): XGBoost model configuration
        y_train (pd.Series): Training target data
        
    Returns:
        Dict: Updated model configuration
    """
    if 'XGBoost' in model_config:
        new_scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model_config['XGBoost']['model'].set_params(scale_pos_weight=new_scale_pos_weight)
        model_config['XGBoost']['scale_pos_weight'] = new_scale_pos_weight
        logger.info(f"🔄 Updated XGBoost scale_pos_weight: {new_scale_pos_weight:.3f}")
    
    return model_config


def validate_advanced_model_config(models_config: Dict[str, Dict]) -> bool:
    """
    Validate that all advanced models are properly configured.
    
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
        if config['category'] != 'advanced':
            logger.warning(f"⚠️ Unexpected category for {model_name}: {config['category']}")
        
        # Specific validation for XGBoost
        if model_name == 'XGBoost' and 'scale_pos_weight' not in config:
            logger.warning(f"⚠️ Missing scale_pos_weight for XGBoost")
    
    logger.info("✅ All advanced model configurations are valid")
    return True


def get_advanced_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about advanced models.
    
    Returns:
        Dict[str, Dict[str, Any]]: Model information
    """
    return {
        'XGBoost': {
            'description': 'Extreme Gradient Boosting with optimized performance',
            'pros': ['High accuracy', 'Feature importance', 'Handles missing values', 'Fast training'],
            'cons': ['Many hyperparameters', 'Can overfit', 'Memory intensive'],
            'best_for': 'Structured data, competitions, high accuracy requirements',
            'special_features': ['Built-in regularization', 'Tree pruning', 'Parallel processing'],
            'available': XGBOOST_AVAILABLE
        },
        'LightGBM': {
            'description': 'Microsoft\'s efficient gradient boosting framework',
            'pros': ['Very fast training', 'Low memory usage', 'High accuracy', 'Categorical support'],
            'cons': ['Can overfit on small datasets', 'Sensitive to parameters'],
            'best_for': 'Large datasets, speed requirements, categorical features',
            'special_features': ['Leaf-wise tree growth', 'Categorical feature support', 'Network training'],
            'available': LIGHTGBM_AVAILABLE
        }
    }


def create_advanced_model_instance(model_name: str, params: Dict[str, Any], 
                                 y_train: Optional[pd.Series] = None) -> Any:
    """
    Create an instance of an advanced model with given parameters.
    
    Args:
        model_name (str): Name of the model
        params (Dict[str, Any]): Model parameters
        y_train (Optional[pd.Series]): Training target for weight calculation
        
    Returns:
        Any: Model instance
    """
    if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight if y_train is provided
        scale_pos_weight = 1.0
        if y_train is not None:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            verbosity=0
        )
        return model
    
    elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
        model = LGBMClassifier(
            **params,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            verbosity=-1,
            n_jobs=-1
        )
        return model
    
    else:
        raise ValueError(f"Model {model_name} not available or not supported")


def get_model_specific_preprocessing_notes() -> Dict[str, List[str]]:
    """
    Get preprocessing notes specific to each advanced model.
    
    Returns:
        Dict[str, List[str]]: Preprocessing recommendations
    """
    return {
        'XGBoost': [
            'Handles missing values internally',
            'No need for feature scaling',
            'Can handle mixed data types',
            'Benefits from feature engineering',
            'Use scale_pos_weight for class imbalance'
        ],
        'LightGBM': [
            'Excellent categorical feature support',
            'Handles missing values well',
            'No scaling required',
            'Fast with large datasets',
            'Use class_weight for balancing'
        ]
    }
