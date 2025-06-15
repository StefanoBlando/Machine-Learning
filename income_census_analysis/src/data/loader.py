"""
Data loading and exploratory data analysis module.
Extracted from Module 2 of the original notebook.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import logging
from pathlib import Path

from ..config.settings import RANDOM_STATE, PLOT_STYLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_explore_data(filepath: str, target_col: str = 'target') -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load and explore data with advanced analysis, handling '?' as NaN.
    Include visualizations for EDA.
    
    Args:
        filepath (str): Path to the CSV data file
        target_col (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]: 
            - Processed dataframe
            - List of numeric column names
            - List of categorical column names
    """
    logger.info("Starting data loading and exploratory data analysis")
    logger.info("-" * 50)
    
    # Load data with error handling
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Dataset loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        logger.error(f"❌ File {filepath} not found!")
        raise FileNotFoundError(f"Data file not found at {filepath}")
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        raise
    
    # Clean column names and strip whitespace from object columns
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Handle '?' as missing values (common in UCI datasets)
    question_marks_summary = {}
    for col in df.select_dtypes(include=['object']).columns:
        if '?' in df[col].values:
            count = (df[col] == '?').sum()
            question_marks_summary[col] = count
            df[col] = df[col].replace('?', np.nan)
    
    # Basic dataset information
    logger.info(f"📊 Dataset Overview:")
    logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"📈 Feature Types:")
    logger.info(f"   Numeric features ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"   Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Handle target column identification and standardization
    if target_col not in df.columns:
        # Try to identify target column
        potential_targets = ['income', 'target', 'class']
        target_col = None
        for col in potential_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            logger.warning("⚠️ Target column not found. Using last column as target.")
            target_col = df.columns[-1]
    
    # Standardize target column
    if target_col != 'target':
        df['target'] = df[target_col]
    
    # Convert target to 0/1 for easier modeling
    if '>50K' in df['target'].unique() and '<=50K' in df['target'].unique():
        df['target'] = (df['target'] == '>50K').astype(int)
    
    # Analyze target distribution
    target_dist = df['target'].value_counts()
    total_samples = len(df)
    
    logger.info(f"🎯 Target Distribution:")
    for target_val, count in target_dist.items():
        percentage = count/total_samples*100
        logger.info(f"   {target_val}: {count:,} ({percentage:.1f}%)")
    
    # Remove target from feature lists
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    if 'target' in categorical_cols:
        categorical_cols.remove('target')
    if target_col != 'target' and target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Missing values analysis
    missing_analysis = df.isnull().sum()
    missing_percentage = (missing_analysis / len(df) * 100).round(2)
    
    logger.info(f"🔍 Missing Values Analysis:")
    if missing_analysis.sum() == 0:
        logger.info("   ✅ No missing values detected")
    else:
        missing_df = pd.DataFrame({
            'Missing_Count': missing_analysis[missing_analysis > 0],
            'Missing_Percentage': missing_percentage[missing_analysis > 0]
        }).sort_values('Missing_Count', ascending=False)
        logger.info(f"Missing values found:\n{missing_df.to_string()}")
    
    if question_marks_summary:
        logger.info(f"❓ Original '?' values converted to NaN:")
        for col, count in question_marks_summary.items():
            logger.info(f"   {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Statistical summary for numeric features
    if numeric_cols:
        logger.info(f"📊 Statistical Summary for Numeric Features:")
        numeric_summary = df[numeric_cols].describe()
        logger.info(f"\n{numeric_summary.round(2).to_string()}")
        
        # Outliers detection (IQR method)
        logger.info(f"🚨 Potential Outliers (IQR method):")
        outliers_found = False
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                logger.info(f"   {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")
                outliers_found = True
        
        if not outliers_found:
            logger.info("   ✅ No significant outliers detected")
    
    # Categorical features summary
    if categorical_cols:
        logger.info(f"📋 Categorical Features Summary:")
        for col in categorical_cols[:5]:  # Show first 5 to avoid clutter
            unique_count = df[col].nunique()
            mode_value = df[col].mode().iloc[0] if not df[col].isnull().all() else 'N/A'
            mode_freq = (df[col] == mode_value).sum()
            logger.info(f"   {col}: {unique_count} unique values, mode: '{mode_value}' ({mode_freq} times)")
        
        if len(categorical_cols) > 5:
            logger.info(f"   ... and {len(categorical_cols) - 5} more categorical features")
    
    # Correlation analysis for numeric features
    if len(numeric_cols) > 1:
        logger.info(f"🔗 High Correlations (|r| > 0.7):")
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            for feat1, feat2, corr_val in high_corr_pairs:
                logger.info(f"   {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            logger.info("   ✅ No high correlations detected")
    
    # Data quality score
    quality_score = 100
    if df.isnull().sum().sum() > 0:
        quality_score -= (df.isnull().sum().sum() / len(df)) * 10
    if any(df[col].nunique() == 1 for col in df.columns if col != 'target'):
        quality_score -= 5
    
    logger.info(f"⭐ Data Quality Score: {quality_score:.1f}/100")
    logger.info(f"✅ Data exploration completed!")
    logger.info(f"📋 Feature count: {len(numeric_cols)} numeric and {len(categorical_cols)} categorical")
    
    return df, numeric_cols, categorical_cols


def create_eda_visualizations(df: pd.DataFrame, numeric_cols: List[str], 
                            categorical_cols: List[str], target_col: str = 'target', 
                            save_path: Optional[str] = None) -> None:
    """
    Generate EDA visualizations for the dataset.
    
    Args:
        df (pd.DataFrame): The dataframe to visualize
        numeric_cols (List[str]): List of numeric column names
        categorical_cols (List[str]): List of categorical column names
        target_col (str): Name of the target column
        save_path (Optional[str]): Path to save the visualization
    """
    logger.info("📈 Generating EDA Visualizations...")
    
    # Set plot style
    try:
        plt.style.use(PLOT_STYLE)
    except:
        plt.style.use('seaborn')
    
    plt.figure(figsize=(18, 12))
    
    # 1. Target Distribution
    if target_col in df.columns and not df[target_col].isnull().all():
        plt.subplot(2, 3, 1)
        sns.countplot(x=df[target_col].astype(str), data=df, palette='viridis')
        plt.title(f'Distribution of Target Variable ({target_col})')
        plt.xlabel('Income Group (0: <=50K, 1: >50K)')
        plt.ylabel('Count')
    
    # 2. Age Distribution
    if 'age' in df.columns:
        plt.subplot(2, 3, 2)
        sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Age')
        plt.xlabel('Age')
        plt.ylabel('Count')
    
    # 3. Hours per Week Distribution
    if 'hours-per-week' in df.columns:
        plt.subplot(2, 3, 3)
        sns.histplot(df['hours-per-week'], bins=30, kde=True, color='lightcoral')
        plt.title('Distribution of Hours per Week')
        plt.xlabel('Hours per Week')
        plt.ylabel('Count')
    
    # 4. Correlation Heatmap of Numeric Features
    if len(numeric_cols) > 1:
        plt.subplot(2, 3, 4)
        numeric_and_target_cols = numeric_cols + ([target_col] if pd.api.types.is_numeric_dtype(df[target_col]) else [])
        corr_matrix = df[numeric_and_target_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numeric Features')
    
    # 5. Relationship vs Income
    if 'relationship' in df.columns and target_col in df.columns:
        plt.subplot(2, 3, 5)
        if pd.api.types.is_numeric_dtype(df[target_col]):
            sns.barplot(x='relationship', y=target_col, data=df, palette='magma')
            plt.title('Income Rate by Relationship Status')
            plt.ylabel('Income Rate (>50K)')
        else:
            relationship_income = pd.crosstab(df['relationship'], df[target_col], normalize='index')
            relationship_income.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'orange'])
            plt.title('Income Distribution by Relationship Status')
        plt.xticks(rotation=45, ha='right')
    
    # 6. Marital Status vs Income
    if 'marital-status' in df.columns and target_col in df.columns:
        plt.subplot(2, 3, 6)
        if pd.api.types.is_numeric_dtype(df[target_col]):
            sns.barplot(x='marital-status', y=target_col, data=df, palette='cividis')
            plt.title('Income Rate by Marital Status')
            plt.ylabel('Income Rate (>50K)')
        else:
            marital_income = pd.crosstab(df['marital-status'], df[target_col], normalize='index')
            marital_income.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'orange'])
            plt.title('Income Distribution by Marital Status')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📁 EDA visualizations saved to {save_path}")
    
    plt.show()
    logger.info("✅ EDA Visualizations generated successfully!")


def get_dataset_summary(df: pd.DataFrame, numeric_cols: List[str], 
                       categorical_cols: List[str]) -> dict:
    """
    Get a comprehensive summary of the dataset.
    
    Args:
        df (pd.DataFrame): The dataframe to summarize
        numeric_cols (List[str]): List of numeric column names  
        categorical_cols (List[str]): List of categorical column names
        
    Returns:
        dict: Summary statistics dictionary
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / len(df)) * 100,
        'numeric_features': len(numeric_cols),
        'categorical_features': len(categorical_cols),
        'duplicate_rows': df.duplicated().sum(),
        'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else None,
        'numeric_summary': df[numeric_cols].describe().to_dict() if numeric_cols else None,
        'categorical_summary': {
            col: {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].empty else None,
                'missing_count': df[col].isnull().sum()
            } for col in categorical_cols
        } if categorical_cols else None
    }
    
    return summary
