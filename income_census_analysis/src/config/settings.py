"""
Global configuration settings for the Advanced Income Classification project.
Estratto ESATTAMENTE dal MODULO 1 del notebook originale.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import time
from datetime import datetime

# Core ML imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_score, validation_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# Traditional models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Advanced models (checking availability for robustness)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# Imbalanced learning (checking availability)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ imbalanced-learn not available. Install with: pip install imbalanced-learn")
    IMBLEARN_AVAILABLE = False

# Bayesian optimization (checking availability, with RandomizedSearchCV fallback)
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
    print("✅ scikit-optimize (BayesSearchCV) available.")
except ImportError:
    print("⚠️ scikit-optimize not available. Install with: pip install scikit-optimize")
    SKOPT_AVAILABLE = False
    from sklearn.model_selection import RandomizedSearchCV
    print("Using RandomizedSearchCV as fallback for hyperparameter tuning.")

# SHAP for interpretability (checking availability)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

# Optuna for advanced methodological comparison (checking availability)
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
    OPTUNA_AVAILABLE = True
    print("✅ Optuna available for methodological deep dive.")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install with: pip install optuna.")

# Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)

# Advanced sklearn (optional, but good to keep if used later)
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.ensemble import IsolationForest

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(123)
plt.style.use('seaborn-v0_8') # Fallback if 'seaborn-v0_8' is not available
sns.set_theme(style="whitegrid", palette="Set2")

# --- Global Constants ---
RANDOM_STATE = 123
TEST_SIZE = 0.3
CV_FOLDS = 5
CV_REPEATS = 3
DATA_FILEPATH = 'data.csv' # Ensure this matches your file name

print("🚀 ADVANCED INCOME CLASSIFICATION PROJECT")
print("=" * 60)
print(f"⚙️  Configuration:")
print(f"   Random State: {RANDOM_STATE}")
print(f"   Train/Validation Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}%")
print(f"   Cross-Validation: {CV_FOLDS}-fold × {CV_REPEATS} repeats")

print(f"\n📦 Available Libraries:")
print(f"   XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}")
print(f"   LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
print(f"   Imbalanced-learn: {'✅' if IMBLEARN_AVAILABLE else '❌'}")
print(f"   Scikit-optimize: {'✅' if SKOPT_AVAILABLE else '❌'}")
print(f"   SHAP: {'✅' if SHAP_AVAILABLE else '❌'}")
print(f"   Optuna: {'✅' if OPTUNA_AVAILABLE else '❌'}")

print(f"\n🕐 Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
