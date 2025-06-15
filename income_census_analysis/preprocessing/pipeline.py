"""
Advanced preprocessing and data splitting module.
Extracted from Module 4 of the original notebook.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced learning imports (with availability check)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

from ..config.settings import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, CV_REPEATS, 
    FEATURE_SELECTION_K_RATIO, SAMPLING_STRATEGIES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def advanced_preprocessing_and_split(df: pd.DataFrame, target_col: str, 
                                   test_size: float = TEST_SIZE, 
                                   random_state: int = RANDOM_STATE) -> Dict:
    """
    Advanced preprocessing and split with multiple balancing strategies.
    Includes feature selection in the pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        test_size (float): Proportion for validation set
        random_state (int): Random seed for reproducibility
        
    Returns:
        Dict: Comprehensive preprocessing results including balanced datasets
    """
    logger.info("⚙️ ADVANCED PREPROCESSING & DATA SPLITTING")
    logger.info("-" * 50)
    
    # Prepare features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    logger.info(f"📊 Dataset for modeling:")
    logger.info(f"   Features: {X.shape[1]}")
    logger.info(f"   Samples: {X.shape[0]:,}")
    logger.info(f"   Target distribution: {(y==0).sum():,} (Class 0) vs {(y==1).sum():,} (Class 1)")
    
    # Class weights calculation
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    imbalance_ratio = (y==1).sum() / (y==0).sum()
    
    logger.info(f"   Imbalance ratio (Class 1 / Class 0): {imbalance_ratio:.3f}")
    logger.info(f"   Calculated Class weights: {{0: {class_weight_dict[0]:.3f}, 1: {class_weight_dict[1]:.3f}}}")
    
    # Stratified train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"✅ Stratified split completed:")
    logger.info(f"   Training: {X_train.shape[0]:,} samples ({y_train.mean()*100:.1f}% positive)")
    logger.info(f"   Validation: {X_val.shape[0]:,} samples ({y_val.mean()*100:.1f}% positive)")
    
    # Feature type identification
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from feature lists
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    if target_col in categorical_features:
        categorical_features.remove(target_col)
    
    logger.info(f"📋 Feature preprocessing setup:")
    logger.info(f"   Numeric features ({len(numeric_features)}): {numeric_features[:5]}{'...' if len(numeric_features) > 5 else ''}")
    logger.info(f"   Categorical features ({len(categorical_features)}): {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    logger.info("✅ Preprocessing pipeline (Scaler + OneHotEncoder) created.")
    
    # Apply preprocessing
    X_train_preprocessed_temp = preprocessor.fit_transform(X_train)
    X_val_preprocessed_temp = preprocessor.transform(X_val)
    
    # Get feature names after preprocessing
    numeric_feature_names = numeric_features
    categorical_feature_names = []
    if len(categorical_features) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']['onehot']
        categorical_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    
    all_feature_names_before_selection = numeric_feature_names + categorical_feature_names
    
    # Feature Selection using SelectKBest
    logger.info("🔧 Applying Feature Selection (SelectKBest)...")
    num_features_after_preprocessing = X_train_preprocessed_temp.shape[1]
    k_to_select = int(num_features_after_preprocessing * FEATURE_SELECTION_K_RATIO)
    k_to_select = max(1, min(k_to_select, num_features_after_preprocessing))
    
    feature_selector = SelectKBest(score_func=f_classif, k=k_to_select)
    X_train_preprocessed = feature_selector.fit_transform(X_train_preprocessed_temp, y_train)
    X_val_preprocessed = feature_selector.transform(X_val_preprocessed_temp)
    
    # Get selected feature names
    selected_feature_indices = feature_selector.get_support(indices=True)
    all_feature_names_after_selection = [all_feature_names_before_selection[i] for i in selected_feature_indices]
    
    logger.info(f"✅ Selected {len(all_feature_names_after_selection)} features out of {num_features_after_preprocessing} (k={k_to_select}).")
    
    # Create balanced datasets using multiple strategies
    balanced_datasets = {}
    
    # Original data with class weights
    balanced_datasets['original'] = {
        'X_train': X_train_preprocessed,
        'y_train': y_train,
        'description': 'Original data (preprocessed & feature-selected) with class weights'
    }
    logger.info(f"⚖️ Creating balanced datasets...")
    logger.info(f"   ✅ Original: {len(y_train):,} samples (for class weights)")
    
    # Create balanced datasets if imblearn is available
    if IMBLEARN_AVAILABLE:
        # SMOTE oversampling
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)
            balanced_datasets['smote'] = {
                'X_train': X_train_smote,
                'y_train': y_train_smote,
                'description': f'SMOTE oversampling (from {len(y_train):,} to {len(y_train_smote):,})'
            }
            logger.info(f"   ✅ SMOTE: {len(y_train):,} → {len(y_train_smote):,} samples")
        except Exception as e:
            logger.error(f"   ❌ SMOTE failed: {str(e)}")
        
        # ADASYN oversampling
        try:
            adasyn = ADASYN(random_state=random_state, n_neighbors=5)
            X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_preprocessed, y_train)
            balanced_datasets['adasyn'] = {
                'X_train': X_train_adasyn,
                'y_train': y_train_adasyn,
                'description': f'ADASYN oversampling (from {len(y_train):,} to {len(y_train_adasyn):,})'
            }
            logger.info(f"   ✅ ADASYN: {len(y_train):,} → {len(y_train_adasyn):,} samples")
        except Exception as e:
            logger.error(f"   ❌ ADASYN failed: {str(e)}")
        
        # SMOTEENN (Combined sampling)
        try:
            smoteenn = SMOTEENN(random_state=random_state)
            X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train_preprocessed, y_train)
            balanced_datasets['smoteenn'] = {
                'X_train': X_train_smoteenn,
                'y_train': y_train_smoteenn,
                'description': f'SMOTEENN combined sampling (from {len(y_train):,} to {len(y_train_smoteenn):,})'
            }
            logger.info(f"   ✅ SMOTEENN: {len(y_train):,} → {len(y_train_smoteenn):,} samples")
        except Exception as e:
            logger.error(f"   ❌ SMOTEENN failed: {str(e)}")
    else:
        logger.warning("⚠️ imblearn not available, only 'original' data with class weights will be used.")
    
    # Cross-validation strategies
    cv_strategies = {
        'stratified_5fold': StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state),
        'repeated_stratified': RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=random_state)
    }
    
    logger.info(f"🎯 Cross-validation strategies prepared:")
    for name, strategy in cv_strategies.items():
        logger.info(f"   {name}: {strategy}")
    
    # Final summary
    logger.info(f"📊 Preprocessing Summary:")
    logger.info(f"   ✅ Train/Val split: {len(X_train):,}/{len(X_val):,}")
    logger.info(f"   ✅ Features after preprocessing and selection: {len(all_feature_names_after_selection)}")
    logger.info(f"   ✅ Balanced datasets created: {len(balanced_datasets)}")
    logger.info(f"   ✅ CV strategies prepared: {len(cv_strategies)}")
    logger.info(f"✅ Preprocessing module completed!")
    
    return {
        'X_train_raw': X_train,
        'X_val_raw': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'X_val_preprocessed': X_val_preprocessed,
        'preprocessor': preprocessor,
        'feature_selector': feature_selector,
        'balanced_datasets': balanced_datasets,
        'cv_strategies': cv_strategies,
        'class_weight_dict': class_weight_dict,
        'feature_names_after_preprocessing': all_feature_names_after_selection,
        'numeric_features_original': numeric_features,
        'categorical_features_original': categorical_features
    }


def create_preprocessing_visualizations(preprocessing_results: Dict, save_path: Optional[str] = None) -> None:
    """
    Create visualizations for preprocessing results.
    
    Args:
        preprocessing_results (Dict): Results from advanced_preprocessing_and_split
        save_path (Optional[str]): Path to save the visualization
    """
    logger.info("📊 Generating Preprocessing Visualizations...")
    
    plt.figure(figsize=(20, 12))
    
    y_train = preprocessing_results['y_train']
    y_val = preprocessing_results['y_val']
    balanced_datasets = preprocessing_results['balanced_datasets']
    
    # 1. Target Distribution in Training Set
    plt.subplot(2, 4, 1)
    sns.countplot(x=y_train.astype(str), order=['0', '1'], palette='viridis')
    plt.title(f'Target Distribution in Training Set\n({len(y_train):,} samples)')
    plt.xlabel('Income Group (0: <=50K, 1: >50K)')
    plt.ylabel('Count')
    
    # 2. Target Distribution in Validation Set
    plt.subplot(2, 4, 2)
    sns.countplot(x=y_val.astype(str), order=['0', '1'], palette='viridis')
    plt.title(f'Target Distribution in Validation Set\n({len(y_val):,} samples)')
    plt.xlabel('Income Group (0: <=50K, 1: >50K)')
    plt.ylabel('Count')
    
    # 3-5. Target Distribution after Resampling Strategies
    subplot_positions = [3, 4, 5]
    strategy_names = ['smote', 'adasyn', 'smoteenn']
    
    for i, strategy_name in enumerate(strategy_names):
        if strategy_name in balanced_datasets and i < len(subplot_positions):
            plt.subplot(2, 4, subplot_positions[i])
            strategy_y = balanced_datasets[strategy_name]['y_train']
            sns.countplot(x=strategy_y.astype(str), order=['0', '1'], palette='cividis')
            plt.title(f'Target Distribution after {strategy_name.upper()}\n({len(strategy_y):,} samples)')
            plt.xlabel('Income Group (0: <=50K, 1: >50K)')
            plt.ylabel('Count')
    
    # 6. Class Distribution Comparison
    plt.subplot(2, 4, 6)
    strategy_counts = {}
    strategy_counts['Original'] = [
        (y_train == 0).sum(), 
        (y_train == 1).sum()
    ]
    
    for strategy_name in ['smote', 'adasyn', 'smoteenn']:
        if strategy_name in balanced_datasets:
            strategy_y = balanced_datasets[strategy_name]['y_train']
            strategy_counts[strategy_name.upper()] = [
                (strategy_y == 0).sum(),
                (strategy_y == 1).sum()
            ]
    
    strategies = list(strategy_counts.keys())
    class_0_counts = [strategy_counts[s][0] for s in strategies]
    class_1_counts = [strategy_counts[s][1] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    plt.bar(x - width/2, class_0_counts, width, label='Class 0 (<=50K)', alpha=0.8)
    plt.bar(x + width/2, class_1_counts, width, label='Class 1 (>50K)', alpha=0.8)
    
    plt.xlabel('Sampling Strategy')
    plt.ylabel('Count')
    plt.title('Class Distribution Comparison')
    plt.xticks(x, strategies, rotation=45)
    plt.legend()
    plt.yscale('log')  # Log scale due to potentially large differences
    
    # 7. Feature Selection Impact
    plt.subplot(2, 4, 7)
    feature_names = preprocessing_results['feature_names_after_preprocessing']
    original_feature_count = len(preprocessing_results['numeric_features_original']) + len(preprocessing_results['categorical_features_original'])
    
    # Estimate features after one-hot encoding (approximation)
    categorical_expansion = sum([10 for _ in preprocessing_results['categorical_features_original']])  # Rough estimate
    features_after_encoding = len(preprocessing_results['numeric_features_original']) + categorical_expansion
    
    feature_counts = [original_feature_count, features_after_encoding, len(feature_names)]
    feature_stages = ['Original', 'After Encoding', 'After Selection']
    
    bars = plt.bar(feature_stages, feature_counts, color=['lightblue', 'orange', 'darkgreen'])
    plt.title('Feature Count Through Pipeline')
    plt.ylabel('Number of Features')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Missing Data Heatmap (if applicable)
    plt.subplot(2, 4, 8)
    X_train_raw = preprocessing_results['X_train_raw']
    missing_data = X_train_raw.isnull()
    
    if missing_data.sum().sum() > 0:
        # Show only columns with missing data
        missing_cols = missing_data.columns[missing_data.sum() > 0]
        if len(missing_cols) > 0:
            plt.imshow(missing_data[missing_cols].head(100).T, cmap='viridis', aspect='auto')
            plt.title('Missing Data Pattern\n(First 100 samples)')
            plt.xlabel('Sample Index')
            plt.ylabel('Features with Missing Data')
            plt.yticks(range(len(missing_cols)), missing_cols, fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('Missing Data Pattern')
    else:
        plt.text(0.5, 0.5, 'No Missing Data\nDetected', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Missing Data Pattern')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📁 Preprocessing visualizations saved to {save_path}")
    
    plt.show()
    logger.info("✅ Preprocessing Visualizations generated successfully!")


def get_preprocessing_summary(preprocessing_results: Dict) -> Dict:
    """
    Get a summary of the preprocessing results.
    
    Args:
        preprocessing_results (Dict): Results from advanced_preprocessing_and_split
        
    Returns:
        Dict: Preprocessing summary
    """
    summary = {
        'train_samples': len(preprocessing_results['y_train']),
        'val_samples': len(preprocessing_results['y_val']),
        'features_selected': len(preprocessing_results['feature_names_after_preprocessing']),
        'original_numeric_features': len(preprocessing_results['numeric_features_original']),
        'original_categorical_features': len(preprocessing_results['categorical_features_original']),
        'balanced_datasets_created': len(preprocessing_results['balanced_datasets']),
        'class_weights': preprocessing_results['class_weight_dict'],
        'cv_strategies': list(preprocessing_results['cv_strategies'].keys()),
        'feature_selection_ratio': FEATURE_SELECTION_K_RATIO,
        'train_class_distribution': {
            'class_0': (preprocessing_results['y_train'] == 0).sum(),
            'class_1': (preprocessing_results['y_train'] == 1).sum(),
            'imbalance_ratio': (preprocessing_results['y_train'] == 1).sum() / (preprocessing_results['y_train'] == 0).sum()
        }
    }
    
    # Add balanced dataset info
    for strategy, data in preprocessing_results['balanced_datasets'].items():
        strategy_y = data['y_train']
        summary[f'{strategy}_samples'] = len(strategy_y)
        summary[f'{strategy}_class_distribution'] = {
            'class_0': (strategy_y == 0).sum(),
            'class_1': (strategy_y == 1).sum()
        }
    
    return summary
