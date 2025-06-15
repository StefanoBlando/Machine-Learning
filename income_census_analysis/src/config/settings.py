"""
Global configuration settings for the Advanced Income Classification project.
Extracted from Module 1 of the original notebook.
"""
import os
from pathlib import Path
from datetime import datetime

# =============================================================================
# PROJECT CONSTANTS (From Module 1)
# =============================================================================

# Random seed for reproducibility
RANDOM_STATE = 123

# Data splitting configuration
TEST_SIZE = 0.3
TRAIN_SIZE = 0.7

# Cross-validation configuration
CV_FOLDS = 5
CV_REPEATS = 3

# File paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file configuration
DATA_FILEPATH = RAW_DATA_DIR / "data.csv"

# =============================================================================
# MODEL AVAILABILITY FLAGS (From Module 1)
# =============================================================================

# These would be dynamically checked in actual implementation
XGBOOST_AVAILABLE = True
LIGHTGBM_AVAILABLE = True
IMBLEARN_AVAILABLE = True
SKOPT_AVAILABLE = True
SHAP_AVAILABLE = True
OPTUNA_AVAILABLE = True

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION (From Module 3)
# =============================================================================

# Columns to remove as per task requirements
COLUMNS_TO_DROP = ['education', 'native-country']

# Occupation mapping to 5 categories (From Module 3)
OCCUPATION_MAPPING = {
    # High Income / Professional/Managerial
    'Exec-managerial': 'Professional_HighIncome',
    'Prof-specialty': 'Professional_HighIncome',
    'Protective-serv': 'Professional_HighIncome',
    
    # Skilled / Technical / Sales (mid-high income potential)
    'Tech-support': 'Technical_Skilled',
    'Sales': 'Technical_Skilled',
    
    # Skilled Manual / Craft (mid-range income)
    'Craft-repair': 'Skilled_Manual',
    'Transport-moving': 'Skilled_Manual',
    'Machine-op-inspct': 'Skilled_Manual',
    'Farming-fishing': 'Skilled_Manual',
    
    # Operational / Administrative (mid-low income)
    'Adm-clerical': 'Operational',
    'Priv-house-serv': 'Operational',
    'Handlers-cleaners': 'Operational',
    
    # Low Income / Other Service / Basic (lowest income categories)
    'Other-service': 'Service_Basic',
    'Armed-Forces': 'Service_Basic',
}

# Age group bins
AGE_GROUP_BINS = [0, 25, 35, 45, 55, 100]
AGE_GROUP_LABELS = ['Young_Adult', 'Early_Career', 'Mid_Career', 'Senior_Career', 'Pre_Retirement']

# Work intensity bins
WORK_INTENSITY_BINS = [0, 20, 35, 40, 50, 100]
WORK_INTENSITY_LABELS = ['Part_Time', 'Reduced_Hours', 'Standard', 'Extended', 'Intensive']

# Education level bins
EDUCATION_LEVEL_BINS = [0, 9, 12, 13, 16, 20]
EDUCATION_LEVEL_LABELS = ['Basic', 'High_School', 'Some_College', 'Bachelor', 'Advanced']

# =============================================================================
# MODEL CONFIGURATION (From Module 5)
# =============================================================================

# Traditional models category
TRADITIONAL_MODELS = [
    'Logistic_Regression',
    'Random_Forest', 
    'Gradient_Boosting',
    'SVM'
]

# Advanced models category
ADVANCED_MODELS = [
    'XGBoost',
    'LightGBM'
]

# Smart iterations per model (From Module 6)
SMART_ITERATIONS = {
    'Logistic_Regression': 5,
    'SVM': 8,
    'Random_Forest': 10,
    'Gradient_Boosting': 12,
    'XGBoost': 15,
    'LightGBM': 12
}

# =============================================================================
# BUSINESS METRICS CONFIGURATION (From Module 6)
# =============================================================================

# Cost matrix for business analysis
COST_FP = 1  # Cost of False Positive
COST_FN = 3  # Cost of False Negative
GAIN_TP = 5  # Gain from True Positive

# Threshold optimization
THRESHOLD_RANGE = (0.01, 0.99)
THRESHOLD_STEPS = 100

# =============================================================================
# VISUALIZATION CONFIGURATION (From Module 9)
# =============================================================================

# Plot styling
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE_LARGE = (24, 18)
FIGURE_SIZE_MEDIUM = (18, 12)
FIGURE_SIZE_SMALL = (12, 8)

# Color palettes
PALETTE_MAIN = 'viridis'
PALETTE_SECONDARY = 'Set2'

# =============================================================================
# SAMPLING STRATEGIES (From Module 4)
# =============================================================================

SAMPLING_STRATEGIES = [
    'original',
    'smote',
    'adasyn', 
    'smoteenn'
]

# =============================================================================
# FEATURE SELECTION CONFIGURATION (From Module 4)
# =============================================================================

# SelectKBest configuration
FEATURE_SELECTION_SCORE_FUNC = 'f_classif'
FEATURE_SELECTION_K_RATIO = 0.70  # Select 70% of features

# =============================================================================
# OPTUNA CONFIGURATION (From Module 6B)
# =============================================================================

# Optuna study configuration
OPTUNA_N_TRIALS = 25
OPTUNA_TIMEOUT_SECONDS = 300
OPTUNA_N_STARTUP_TRIALS = 8

# Optuna samplers
OPTUNA_SAMPLERS = ['TPE', 'CMA-ES', 'Random']

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_dir_exists(path):
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_experiment_dir():
    """Get experiment directory with timestamp."""
    experiment_dir = RESULTS_DIR / "experiments" / get_timestamp()
    ensure_dir_exists(experiment_dir)
    return experiment_dir

# Environment-specific overrides
if os.getenv('INCOME_CLASSIFICATION_ENV') == 'development':
    # Reduce iterations for faster development
    SMART_ITERATIONS = {k: max(2, v // 3) for k, v in SMART_ITERATIONS.items()}
    OPTUNA_N_TRIALS = 5
    CV_FOLDS = 3
    CV_REPEATS = 1

# Print configuration on import (optional)
if __name__ == "__main__":
    print("🚀 ADVANCED INCOME CLASSIFICATION - CONFIGURATION")
    print("=" * 60)
    print(f"Random State: {RANDOM_STATE}")
    print(f"Train/Test Split: {int(TRAIN_SIZE*100)}/{int(TEST_SIZE*100)}%")
    print(f"Cross-Validation: {CV_FOLDS}-fold × {CV_REPEATS} repeats")
    print(f"Traditional Models: {len(TRADITIONAL_MODELS)}")
    print(f"Advanced Models: {len(ADVANCED_MODELS)}")
    print(f"Data File: {DATA_FILEPATH}")
    print("=" * 60)
