"""
Data loading and exploratory data analysis module.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from pathlib import Path

from ..config.settings import RANDOM_STATE


def load_and_explore_data(filepath: str) -> tuple[pd.DataFrame, list, list]:
    """
    Carica e esplora i dati con analisi avanzata, gestendo '?' come NaN.
    Include visualizzazioni chiave per l'EDA.
    """
    print("\n2. DATA LOADING E EXPLORATORY DATA ANALYSIS")
    print("-" * 50)
    
    # Load data with error handling
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"❌ File {filepath} not found!")
        return None, [], []
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None, [], []
    
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
    
    # Basic info
    print(f"\n📊 Dataset Overview:")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n📈 Feature Types:")
    print(f"   Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"   Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Identify target column (assuming it's 'income' or 'target')
    target_col = None
    if 'income' in df.columns:
        target_col = 'income'
    elif 'target' in df.columns:
        target_col = 'target'
    
    if target_col:
        # Standardize target names for consistency
        df['target'] = df[target_col] 
        # Convert target to 0/1 for easier modeling if it's not already
        if '>50K' in df['target'].unique() and '<=50K' in df['target'].unique():
            df['target'] = (df['target'] == '>50K').astype(int)
        
        target_dist = df['target'].value_counts()
        total_samples = len(df)
        
        print(f"\n🎯 Target Distribution:")
        for target, count in target_dist.items():
            percentage = count/total_samples*100
            print(f"   {target}: {count:,} ({percentage:.1f}%)\n")
        
        # Remove target from categorical_cols for feature processing
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
    else:
        print("⚠️ Warning: Target column not found. Assuming the last column for target analysis if not specified.")
        
    # Missing values analysis (after '?' conversion)
    missing_analysis = df.isnull().sum()
    missing_percentage = (missing_analysis / len(df) * 100).round(2)
    
    print(f"🔍 Missing Values Analysis (after '?' conversion):")
    if missing_analysis.sum() == 0:
        print("   ✅ No missing values detected")
    else:
        missing_df = pd.DataFrame({
            'Missing_Count': missing_analysis[missing_analysis > 0],
            'Missing_Percentage': missing_percentage[missing_analysis > 0]
        }).sort_values('Missing_Count', ascending=False)
        print(missing_df.to_string())
    
    if question_marks_summary:
        print(f"\n❓ Original '?' values converted to NaN:\n")
        for col, count in question_marks_summary.items():
            print(f"   {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Statistical summary for numeric features
    print(f"\n📊 Statistical Summary:")
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe()
        print(f"\n   Numeric Features:\n")
        print(numeric_summary.round(2).to_string())
        
        # Outliers detection (IQR method)
        print(f"\n   🚨 Potential Outliers (IQR method):\n")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"      {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")
    
    # Categorical features summary
    if categorical_cols:
        print(f"\n   Categorical Features:\n")
        for col in categorical_cols[:5]:  # Show first 5 to avoid clutter
            unique_count = df[col].nunique()
            mode_value = df[col].mode().iloc[0] if not df[col].isnull().all() else 'N/A'
            mode_freq = (df[col] == mode_value).sum()
            print(f"      {col}: {unique_count} unique values, mode: '{mode_value}' ({mode_freq} times)")
        if len(categorical_cols) > 5:
            print(f"      ... and {len(categorical_cols) - 5} more categorical features")
    
    # Correlation analysis for numeric features
    if len(numeric_cols) > 1:
        print(f"\n🔗 High Correlations (|r| > 0.7):\n")
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        if high_corr_pairs:
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"      {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            print("      ✅ No high correlations detected")
    
    # Data quality score (simplified)
    quality_score = 100
    if df.isnull().sum().sum() > 0:
        quality_score -= (df.isnull().sum().sum() / len(df)) * 10 
    if any(df[col].nunique() == 1 for col in df.columns if col != target_col): # Check for constant features
        quality_score -= 5
    
    print(f"\n⭐ Data Quality Score: {quality_score:.1f}/100\n")
    
    print(f"✅ Data exploration completed!")
    print(f"📋 Initial feature count: {len(numeric_cols)} numeric and {len(categorical_cols)} categorical (excluding target)")

    # --- EDA Visualizations ---
    print("\n📈 Generating EDA Visualizations...")
    
    plt.figure(figsize=(18, 12))
    
    # 1. Target Distribution
    if target_col and not df[target_col].isnull().all():
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
        # Ensure target is numeric before including in heatmap
        numeric_and_target_cols = numeric_cols + [target_col] if target_col and pd.api.types.is_numeric_dtype(df[target_col]) else numeric_cols
        
        corr_matrix = df[numeric_and_target_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numeric Features')
    
    # 5. Relationship vs Income
    if 'relationship' in df.columns and target_col:
        plt.subplot(2, 3, 5)
        sns.barplot(x='relationship', y=df[target_col], data=df, palette='magma')
        plt.title('Income Rate by Relationship Status')
        plt.xlabel('Relationship')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')

    # 6. Marital Status vs Income
    if 'marital-status' in df.columns and target_col:
        plt.subplot(2, 3, 6)
        sns.barplot(x='marital-status', y=df[target_col], data=df, palette='cividis')
        plt.title('Income Rate by Marital Status')
        plt.xlabel('Marital Status')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    print("✅ EDA Visualizations generated successfully!")

    # Store original info for reference in global variables
    global original_shape, original_numeric_cols, original_categorical_cols, target_feature_name
    original_shape = df.shape
    original_numeric_cols = numeric_cols.copy()
    original_categorical_cols = categorical_cols.copy()
    target_feature_name = target_col # Store the identified target column name

    return df, numeric_cols, categorical_cols
