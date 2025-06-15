"""
Advanced feature engineering module.
Extracted from Module 3 of the original notebook.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import logging
from sklearn.ensemble import IsolationForest

from ..config.settings import (
    COLUMNS_TO_DROP, OCCUPATION_MAPPING, AGE_GROUP_BINS, AGE_GROUP_LABELS,
    WORK_INTENSITY_BINS, WORK_INTENSITY_LABELS, EDUCATION_LEVEL_BINS, 
    EDUCATION_LEVEL_LABELS, RANDOM_STATE, PLOT_STYLE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def advanced_feature_engineering(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Advanced feature engineering with domain knowledge and intelligent feature creation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        
    Returns:
        Tuple containing:
        - Processed dataframe
        - Updated numeric columns list
        - Updated categorical columns list  
        - List of newly created feature names
    """
    logger.info("🚀 Starting Advanced Feature Engineering")
    logger.info("-" * 50)
    
    df_processed = df.copy()
    new_features_created = []
    
    # 1. COLUMN REMOVAL AS PER REQUIREMENTS
    logger.info("🗑️ Removing specified columns...")
    existing_drops = [col for col in COLUMNS_TO_DROP if col in df_processed.columns]
    
    if existing_drops:
        df_processed = df_processed.drop(existing_drops, axis=1)
        logger.info(f"   ✅ Removed: {existing_drops}")
    else:
        logger.info(f"   ⚠️ Columns {COLUMNS_TO_DROP} not found in dataset, skipping removal.")
    
    # Re-identify feature types after dropping columns
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Ensure target is not treated as a feature
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    if target_col in categorical_features:
        categorical_features.remove(target_col)
    
    # 2. INTELLIGENT MISSING VALUE IMPUTATION
    logger.info("🔧 Handling missing values...")
    missing_cols = df_processed.isnull().sum()
    for col in missing_cols[missing_cols > 0].index:
        if df_processed[col].dtype == 'object':
            mode_val = df_processed[col].mode().iloc[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
            df_processed[col] = df_processed[col].fillna(mode_val)
            logger.info(f"   {col}: filled missing values with '{mode_val}'")
        else:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
            logger.info(f"   {col}: filled missing values with median {median_val}")
    
    # 3. SMART OCCUPATION MAPPING (to 5 categories)
    if 'occupation' in df_processed.columns:
        logger.info("👷 Smart Occupation Mapping (to 5 categories)...")
        
        # Calculate income rate for each original occupation to guide mapping
        occupation_income_rate = df_processed.groupby('occupation')[target_col].mean().sort_values(ascending=False)
        
        logger.info("   Original occupation income rates:")
        for occ, rate in occupation_income_rate.head(10).items():
            count = (df_processed['occupation'] == occ).sum()
            logger.info(f"      {occ:<20} {rate:.3f} ({count:,} samples)")
        
        # Apply mapping and handle unmapped values
        df_processed['occupation'] = df_processed['occupation'].map(OCCUPATION_MAPPING).fillna('Unknown_Occupation')
        
        logger.info("   ✅ Mapped to 5 categories:")
        new_dist = df_processed['occupation'].value_counts()
        for occ, count in new_dist.items():
            rate = df_processed[df_processed['occupation'] == occ][target_col].mean()
            logger.info(f"      {occ:<25} {count:,} ({count/len(df_processed)*100:.1f}%) - Income rate: {rate:.3f}")
        
        # Ensure 'occupation' is correctly recognized as categorical
        if 'occupation' in numeric_features:
            numeric_features.remove('occupation')
        if 'occupation' not in categorical_features:
            categorical_features.append('occupation')
    
    # 4. ADVANCED FEATURE CREATION
    logger.info("🚀 Creating advanced features...")
    
    # Age-based features
    if 'age' in df_processed.columns:
        # Age groups using domain knowledge
        df_processed['age_group'] = pd.cut(df_processed['age'], bins=AGE_GROUP_BINS, 
                                         labels=AGE_GROUP_LABELS, include_lowest=True)
        # Age squared to capture non-linear relationships
        df_processed['age_squared'] = df_processed['age'] ** 2
        new_features_created.extend(['age_group', 'age_squared'])
        logger.info("   ✅ Created age-based features")
    
    # Work intensity and patterns
    if 'hours-per-week' in df_processed.columns:
        # Categorize hours per week
        df_processed['work_intensity'] = pd.cut(df_processed['hours-per-week'], bins=WORK_INTENSITY_BINS, 
                                              labels=WORK_INTENSITY_LABELS, include_lowest=True)
        # Indicator for working overtime
        df_processed['is_overtime'] = (df_processed['hours-per-week'] > 40).astype(int)
        # Work intensity score with diminishing returns
        df_processed['work_intensity_score'] = np.where(df_processed['hours-per-week'] <= 40, 
                                                      df_processed['hours-per-week'] / 40,  
                                                      1 + (df_processed['hours-per-week'] - 40) / 60)
        new_features_created.extend(['work_intensity', 'is_overtime', 'work_intensity_score'])
        logger.info("   ✅ Created work pattern features")
    
    # Capital and financial features
    if 'capital-gain' in df_processed.columns and 'capital-loss' in df_processed.columns:
        # Net capital gain/loss
        df_processed['capital_net'] = df_processed['capital-gain'] - df_processed['capital-loss']
        # Indicators for any capital activity
        df_processed['has_capital_gain'] = (df_processed['capital-gain'] > 0).astype(int)
        df_processed['has_capital_loss'] = (df_processed['capital-loss'] > 0).astype(int)
        df_processed['has_any_capital_activity'] = ((df_processed['capital-gain'] > 0) | 
                                                  (df_processed['capital-loss'] > 0)).astype(int)
        # Ratio (avoid division by zero)
        df_processed['capital_gain_to_loss_ratio'] = df_processed['capital-gain'] / (df_processed['capital-loss'] + 1e-6)
        # Log transformations for highly skewed features
        df_processed['capital_gain_log'] = np.log1p(df_processed['capital-gain'])
        df_processed['capital_loss_log'] = np.log1p(df_processed['capital-loss'])
        new_features_created.extend(['capital_net', 'has_capital_gain', 'has_capital_loss', 
                                   'has_any_capital_activity', 'capital_gain_to_loss_ratio',
                                   'capital_gain_log', 'capital_loss_log'])
        logger.info("   ✅ Created capital/financial features")
    
    # Marital status simplification
    if 'marital-status' in df_processed.columns:
        # Simplify marital status into broader categories
        married_mapping = {
            'Married-civ-spouse': 'Married', 'Married-spouse-absent': 'Married', 'Married-AF-spouse': 'Married',
            'Divorced': 'Previously_Married', 'Separated': 'Previously_Married', 'Widowed': 'Previously_Married',
            'Never-married': 'Never_Married'
        }
        df_processed['marital_simple'] = df_processed['marital-status'].map(married_mapping).fillna('Unknown_Marital')
        # Indicator for stable civil marriage
        df_processed['is_stable_marriage'] = (df_processed['marital-status'] == 'Married-civ-spouse').astype(int)
        new_features_created.extend(['marital_simple', 'is_stable_marriage'])
        logger.info("   ✅ Created marital status features")
    
    # Education features (using education-num)
    if 'education-num' in df_processed.columns:
        # Group numerical education levels
        df_processed['education_level'] = pd.cut(df_processed['education-num'], bins=EDUCATION_LEVEL_BINS, 
                                               labels=EDUCATION_LEVEL_LABELS, include_lowest=True)
        # Squared term for non-linear effects
        df_processed['education_num_squared'] = df_processed['education-num'] ** 2
        new_features_created.extend(['education_level', 'education_num_squared'])
        logger.info("   ✅ Created education features")
    
    # 5. INTERACTION FEATURES
    logger.info("🔗 Creating interaction features...")
    interaction_features_count = 0
    
    if 'age' in df_processed.columns and 'education-num' in df_processed.columns:
        df_processed['age_education_interaction'] = df_processed['age'] * df_processed['education-num']
        new_features_created.append('age_education_interaction')
        interaction_features_count += 1
    
    if 'hours-per-week' in df_processed.columns and 'age' in df_processed.columns:
        # Work efficiency proxy
        df_processed['work_efficiency'] = df_processed['hours-per-week'] / (df_processed['age'] + 1)
        new_features_created.append('work_efficiency')
        interaction_features_count += 1
    
    if 'age' in df_processed.columns and 'education-num' in df_processed.columns:
        # Experience proxy
        df_processed['experience_proxy'] = np.maximum(df_processed['age'] - df_processed['education-num'] - 5, 0)
        new_features_created.append('experience_proxy')
        interaction_features_count += 1
    
    logger.info(f"   ✅ Created {interaction_features_count} interaction features")
    
    # 6. OUTLIER DETECTION
    current_numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in current_numeric_features:
        current_numeric_features.remove(target_col)
    
    if len(current_numeric_features) > 3:
        logger.info("🚨 Outlier detection (IsolationForest)...")
        try:
            iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
            outlier_predictions = iso_forest.fit_predict(df_processed[current_numeric_features])
            df_processed['is_outlier'] = (outlier_predictions == -1).astype(int)
            df_processed['outlier_score'] = iso_forest.decision_function(df_processed[current_numeric_features])
            
            outlier_count = df_processed['is_outlier'].sum()
            logger.info(f"   ✅ Detected {outlier_count} outliers ({outlier_count/len(df_processed)*100:.1f}%)")
            new_features_created.extend(['is_outlier', 'outlier_score'])
        except Exception as e:
            logger.error(f"   ❌ Outlier detection failed: {str(e)}. Skipping outlier features.")
    
    # 7. TARGET ENCODING FEATURES
    logger.info("📊 Creating group-based features (Target Encoding-like)...")
    
    # Re-identify categorical features including newly created ones
    current_categorical_features = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in current_categorical_features:
        current_categorical_features.remove(target_col)
    
    # Select specific categorical features for target encoding
    selected_target_encode_cols = []
    for col in ['workclass', 'marital_simple', 'occupation']:
        if col in df_processed.columns:
            selected_target_encode_cols.append(col)
    
    for cat_col in selected_target_encode_cols:
        income_rate_col = f'{cat_col}_income_rate'
        df_processed[income_rate_col] = df_processed.groupby(cat_col)[target_col].transform('mean')
        new_features_created.append(income_rate_col)
        logger.info(f"   ✅ Created {income_rate_col}")
    
    # Update feature lists
    updated_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    updated_categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col in updated_numeric_cols:
        updated_numeric_cols.remove(target_col)
    if target_col in updated_categorical_cols:
        updated_categorical_cols.remove(target_col)
    
    # Summary
    logger.info("📈 Feature Engineering Summary:")
    logger.info(f"   Original features: {df.shape[1]} (excluding target)")
    logger.info(f"   Features removed: {len(COLUMNS_TO_DROP)}")
    logger.info(f"   New features created: {len(new_features_created)}")
    logger.info(f"   Final feature count: {len(updated_numeric_cols)} numeric + {len(updated_categorical_cols)} categorical")
    logger.info(f"   Total features: {len(updated_numeric_cols) + len(updated_categorical_cols)}")
    
    logger.info("📋 List of New Features Created:")
    for i, feature in enumerate(new_features_created, 1):
        logger.info(f"   {i:2d}. {feature}")
    
    logger.info(f"🎯 Dataset ready for preprocessing: {df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
    logger.info("✅ Advanced feature engineering completed!")
    
    return df_processed, updated_numeric_cols, updated_categorical_cols, new_features_created


def create_feature_engineering_visualizations(df_original: pd.DataFrame, df_engineered: pd.DataFrame, 
                                             target_col: str, save_path: Optional[str] = None) -> None:
    """
    Create visualizations to show the impact of feature engineering.
    
    Args:
        df_original (pd.DataFrame): Original dataframe before feature engineering
        df_engineered (pd.DataFrame): Dataframe after feature engineering
        target_col (str): Name of the target column
        save_path (Optional[str]): Path to save the visualization
    """
    logger.info("📊 Generating Feature Engineering Visualizations...")
    
    # Set plot style
    try:
        plt.style.use(PLOT_STYLE)
    except:
        plt.style.use('seaborn')
    
    plt.figure(figsize=(18, 15))
    
    # 1. New Occupation vs Income
    if 'occupation' in df_engineered.columns:
        plt.subplot(3, 3, 1)
        occupation_income = df_engineered.groupby('occupation')[target_col].mean().sort_values(ascending=False)
        occupation_income.plot(kind='bar', color='coolwarm')
        plt.title('Income Rate by New Occupation Category')
        plt.xlabel('Occupation Category')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')
    
    # 2. Age Group vs Income
    if 'age_group' in df_engineered.columns:
        plt.subplot(3, 3, 2)
        age_group_income = df_engineered.groupby('age_group')[target_col].mean()
        age_group_income.plot(kind='bar', color='mako')
        plt.title('Income Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')
    
    # 3. Work Intensity vs Income
    if 'work_intensity' in df_engineered.columns:
        plt.subplot(3, 3, 3)
        work_intensity_income = df_engineered.groupby('work_intensity')[target_col].mean()
        work_intensity_income.plot(kind='bar', color='rocket')
        plt.title('Income Rate by Work Intensity')
        plt.xlabel('Work Intensity')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')
    
    # 4. Distribution of Capital Net
    if 'capital_net' in df_engineered.columns:
        plt.subplot(3, 3, 4)
        plt.hist(df_engineered['capital_net'], bins=50, alpha=0.7, color='purple')
        plt.title('Distribution of Capital Net')
        plt.xlabel('Capital Net')
        plt.ylabel('Count')
        plt.yscale('log')  # Log scale due to skewness
    
    # 5. Distribution of Outlier Score
    if 'outlier_score' in df_engineered.columns:
        plt.subplot(3, 3, 5)
        plt.hist(df_engineered['outlier_score'], bins=50, alpha=0.7, color='darkgreen')
        plt.title('Distribution of Outlier Score')
        plt.xlabel('Outlier Score')
        plt.ylabel('Count')
    
    # 6. Correlation Heatmap with Key New Features
    key_new_features = ['age_squared', 'capital_net', 'work_intensity_score']
    target_encoded_features = [col for col in df_engineered.columns if '_income_rate' in col]
    
    # Include existing important features
    existing_features = ['hours-per-week', 'education-num']
    plot_features = []
    
    for feat_list in [key_new_features, target_encoded_features[:3], existing_features]:
        for feat in feat_list:
            if feat in df_engineered.columns:
                plot_features.append(feat)
    
    if target_col in df_engineered.columns and pd.api.types.is_numeric_dtype(df_engineered[target_col]):
        plot_features.append(target_col)
    
    if len(plot_features) > 1:
        plt.subplot(3, 3, 6)
        corr_matrix = df_engineered[plot_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=.5, annot_kws={"size": 8})
        plt.title('Correlation Matrix with Key New Features')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    
    # 7. Feature Count Comparison
    plt.subplot(3, 3, 7)
    original_features = len([col for col in df_original.columns if col != target_col])
    engineered_features = len([col for col in df_engineered.columns if col != target_col])
    
    plt.bar(['Original', 'After Engineering'], [original_features, engineered_features], 
           color=['lightblue', 'darkblue'])
    plt.title('Feature Count Comparison')
    plt.ylabel('Number of Features')
    for i, v in enumerate([original_features, engineered_features]):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 8. Target Encoding Example
    if target_encoded_features:
        plt.subplot(3, 3, 8)
        example_feature = target_encoded_features[0]
        base_feature = example_feature.replace('_income_rate', '')
        
        if base_feature in df_engineered.columns:
            target_rates = df_engineered.groupby(base_feature)[target_col].mean().sort_values(ascending=False)
            target_rates.plot(kind='bar', color='orange')
            plt.title(f'Target Encoding: {base_feature}')
            plt.xlabel(base_feature.replace('_', ' ').title())
            plt.ylabel('Income Rate')
            plt.xticks(rotation=45, ha='right')
    
    # 9. Interaction Feature Example
    if 'age_education_interaction' in df_engineered.columns:
        plt.subplot(3, 3, 9)
        plt.scatter(df_engineered['age_education_interaction'], df_engineered[target_col], 
                   alpha=0.5, color='red', s=1)
        plt.title('Age × Education Interaction Feature')
        plt.xlabel('Age × Education Interaction')
        plt.ylabel('Income (0/1)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📁 Feature engineering visualizations saved to {save_path}")
    
    plt.show()
    logger.info("✅ Feature Engineering Visualizations generated successfully!")


def get_feature_engineering_summary(new_features_created: List[str], df_original: pd.DataFrame, 
                                   df_engineered: pd.DataFrame) -> dict:
    """
    Get a summary of the feature engineering process.
    
    Args:
        new_features_created (List[str]): List of newly created feature names
        df_original (pd.DataFrame): Original dataframe
        df_engineered (pd.DataFrame): Engineered dataframe
        
    Returns:
        dict: Feature engineering summary
    """
    summary = {
        'original_feature_count': df_original.shape[1] - 1,  # Exclude target
        'engineered_feature_count': df_engineered.shape[1] - 1,  # Exclude target
        'new_features_count': len(new_features_created),
        'new_features_list': new_features_created,
        'features_removed': len(COLUMNS_TO_DROP),
        'removed_features': COLUMNS_TO_DROP,
        'occupation_categories': len(OCCUPATION_MAPPING.values()),
        'feature_types': {
            'age_based': len([f for f in new_features_created if 'age' in f.lower()]),
            'work_based': len([f for f in new_features_created if 'work' in f.lower() or 'hour' in f.lower()]),
            'capital_based': len([f for f in new_features_created if 'capital' in f.lower()]),
            'marital_based': len([f for f in new_features_created if 'marital' in f.lower()]),
            'education_based': len([f for f in new_features_created if 'education' in f.lower()]),
            'interaction_based': len([f for f in new_features_created if 'interaction' in f.lower() or 'efficiency' in f.lower() or 'experience' in f.lower()]),
            'outlier_based': len([f for f in new_features_created if 'outlier' in f.lower()]),
            'target_encoded': len([f for f in new_features_created if 'income_rate' in f.lower()])
        }
    }
    
    return summary
