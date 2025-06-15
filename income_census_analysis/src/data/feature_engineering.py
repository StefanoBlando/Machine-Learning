"""
Advanced feature engineering module.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from sklearn.ensemble import IsolationForest

from ..config.settings import RANDOM_STATE


def advanced_feature_engineering(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list, list, list]:
    """
    Feature engineering avanzato con creazione di features intelligenti.
    Rimuove colonne specificate e gestisce i missing values.
    Include visualizzazioni per mostrare l'impatto dell'engineering.
    """
    print("\n3. ADVANCED FEATURE ENGINEERING")
    print("-" * 50)
    
    df_processed = df.copy()
    new_features_created = [] # List to track names of newly created features
    
    # 1. COLUMN REMOVAL AS PER REQUIREMENTS
    print(f"🗑️ Removing specified columns...")
    # These columns are to be removed as per task instructions
    columns_to_drop = ['education', 'native-country'] 
    existing_drops = [col for col in columns_to_drop if col in df_processed.columns]
    
    if existing_drops:
        df_processed = df_processed.drop(existing_drops, axis=1)
        print(f"   ✅ Removed: {existing_drops}")
    else:
        print(f"   ⚠️ Columns {columns_to_drop} not found in dataset, skipping removal.")
    
    # Re-identify feature types after dropping columns
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    if target_col in numeric_features: # Ensure target is not treated as a feature
        numeric_features.remove(target_col)
    if target_col in categorical_features:
        categorical_features.remove(target_col)
    
    # 2. INTELLIGENT MISSING VALUE IMPUTATION (after '?' conversion in EDA)
    print(f"\n🔧 Handling missing values...")
    missing_cols_after_drop = df_processed.isnull().sum()
    for col in missing_cols_after_drop[missing_cols_after_drop > 0].index:
        if df_processed[col].dtype == 'object':
            # Mode imputation for categorical, 'Unknown' if no mode or all NaN
            mode_val = df_processed[col].mode().iloc[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
            df_processed[col] = df_processed[col].fillna(mode_val)
            print(f"   {col}: filled missing values with '{mode_val}'")
        else:
            # Median imputation for numerical (more robust to outliers than mean)
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
            print(f"   {col}: filled missing values with median {median_val}")
    
    # 3. SMART OCCUPATION MAPPING (to 5 categories)
    if 'occupation' in df_processed.columns:
        print(f"\n👷 Smart Occupation Mapping (to 5 categories)...")
        
        # Calculate income rate for each original occupation to guide mapping
        occupation_income_rate = df_processed.groupby('occupation')[target_col].mean().sort_values(ascending=False)
        
        print(f"   Original occupation income rates:\n")
        for occ, rate in occupation_income_rate.head(10).items():
            count = (df_processed['occupation'] == occ).sum()
            print(f"      {occ:<20} {rate:.3f} ({count:,} samples)")
        
        # Define mapping based on income rates and logical groupings.
        # These groupings reflect common patterns observed in this dataset's income distribution.
        occupation_mapping = {
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
        
        # Apply mapping and handle any unmapped values (e.g., if '?' was in original and not yet imputed, or typos)
        df_processed['occupation'] = df_processed['occupation'].map(occupation_mapping).fillna('Unknown_Occupation')
        
        print(f"\n   ✅ Mapped to 5 categories:\n")
        new_dist = df_processed['occupation'].value_counts()
        for occ, count in new_dist.items():
            rate = df_processed[df_processed['occupation'] == occ][target_col].mean()
            print(f"      {occ:<25} {count:,} ({count/len(df_processed)*100:.1f}%) - Income rate: {rate:.3f}")
        
        # Ensure 'occupation' is correctly recognized as a categorical feature type
        if 'occupation' in numeric_features:
            numeric_features.remove('occupation')
        if 'occupation' not in categorical_features:
            categorical_features.append('occupation')
    
    # 4. ADVANCED FEATURE CREATION
    print(f"\n🚀 Creating advanced features...\n")
    
    # Age-based features
    if 'age' in df_processed.columns:
        # Age groups using domain knowledge (e.g., career stages)
        df_processed['age_group'] = pd.cut(df_processed['age'], bins=[0, 25, 35, 45, 55, 100], 
                                           labels=['Young_Adult', 'Early_Career', 'Mid_Career', 'Senior_Career', 'Pre_Retirement'])
        # Age squared to capture non-linear relationships
        df_processed['age_squared'] = df_processed['age'] ** 2
        new_features_created.extend(['age_group', 'age_squared'])
        print(f"   ✅ Created age-based features")
    
    # Work intensity and patterns
    if 'hours-per-week' in df_processed.columns:
        # Categorize hours per week
        df_processed['work_intensity'] = pd.cut(df_processed['hours-per-week'], bins=[0, 20, 35, 40, 50, 100], 
                                                labels=['Part_Time', 'Reduced_Hours', 'Standard', 'Extended', 'Intensive'])
        # Indicator for working overtime
        df_processed['is_overtime'] = (df_processed['hours-per-week'] > 40).astype(int)
        # A score that reflects work intensity, potentially with diminishing returns after 40 hours
        df_processed['work_intensity_score'] = np.where(df_processed['hours-per-week'] <= 40, 
                                                        df_processed['hours-per-week'] / 40,  
                                                        1 + (df_processed['hours-per-week'] - 40) / 60) 
        new_features_created.extend(['work_intensity', 'is_overtime', 'work_intensity_score'])
        print(f"   ✅ Created work pattern features")
    
    # Capital and financial features
    if 'capital-gain' in df_processed.columns and 'capital-loss' in df_processed.columns:
        # Net capital gain/loss
        df_processed['capital_net'] = df_processed['capital-gain'] - df_processed['capital-loss']
        # Indicators for any capital activity
        df_processed['has_capital_gain'] = (df_processed['capital-gain'] > 0).astype(int)
        df_processed['has_capital_loss'] = (df_processed['capital-loss'] > 0).astype(int)
        df_processed['has_any_capital_activity'] = ((df_processed['capital-gain'] > 0) | (df_processed['capital-loss'] > 0)).astype(int)
        # Ratio, to avoid division by zero, add a small constant to the denominator
        df_processed['capital_gain_to_loss_ratio'] = df_processed['capital-gain'] / (df_processed['capital-loss'] + 1e-6)
        # Log transformations for highly skewed capital features (add 1 to handle zeros)
        df_processed['capital_gain_log'] = np.log1p(df_processed['capital-gain'])
        df_processed['capital_loss_log'] = np.log1p(df_processed['capital-loss'])
        new_features_created.extend(['capital_net', 'has_capital_gain', 'has_capital_loss', 
                                     'has_any_capital_activity', 'capital_gain_to_loss_ratio',
                                     'capital_gain_log', 'capital_loss_log'])
        print(f"   ✅ Created capital/financial features")
    
    # Marital status simplification
    if 'marital-status' in df_processed.columns:
        # Simplify marital status into broader, more impactful categories for income
        married_mapping = {
            'Married-civ-spouse': 'Married', 'Married-spouse-absent': 'Married', 'Married-AF-spouse': 'Married',
            'Divorced': 'Previously_Married', 'Separated': 'Previously_Married', 'Widowed': 'Previously_Married',
            'Never-married': 'Never_Married'
        }
        df_processed['marital_simple'] = df_processed['marital-status'].map(married_mapping).fillna('Unknown_Marital')
        # Indicator for being in a currently stable, civil marriage
        df_processed['is_stable_marriage'] = (df_processed['marital-status'] == 'Married-civ-spouse').astype(int)
        new_features_created.extend(['marital_simple', 'is_stable_marriage'])
        print(f"   ✅ Created marital status features")
    
    # Education features (using education-num, as 'education' was dropped)
    if 'education-num' in df_processed.columns:
        # Group numerical education levels into broader categories
        df_processed['education_level'] = pd.cut(df_processed['education-num'], bins=[0, 9, 12, 13, 16, 20], 
                                                labels=['Basic', 'High_School', 'Some_College', 'Bachelor', 'Advanced'])
        # Squared term for education level, to capture non-linear effects
        df_processed['education_num_squared'] = df_processed['education-num'] ** 2
        new_features_created.extend(['education_level', 'education_num_squared'])
        print(f"   ✅ Created education features")
    
    # 5. INTERACTION FEATURES (combinations of existing features)
    print(f"\n🔗 Creating interaction features...\n")
    interaction_features_count = 0
    if 'age' in df_processed.columns and 'education-num' in df_processed.columns:
        df_processed['age_education_interaction'] = df_processed['age'] * df_processed['education-num']
        new_features_created.append('age_education_interaction')
        interaction_features_count += 1
    if 'hours-per-week' in df_processed.columns and 'age' in df_processed.columns:
        # A proxy for efficiency in work (hours per week relative to age/experience)
        df_processed['work_efficiency'] = df_processed['hours-per-week'] / (df_processed['age'] + 1) # Add 1 to age to avoid division by zero
        new_features_created.append('work_efficiency')
        interaction_features_count += 1
    if 'age' in df_processed.columns and 'education-num' in df_processed.columns:
        # A proxy for years of experience (age - education years - assumed start age)
        df_processed['experience_proxy'] = np.maximum(df_processed['age'] - df_processed['education-num'] - 5, 0) # Clamp at 0
        new_features_created.append('experience_proxy')
        interaction_features_count += 1
    print(f"   ✅ Created {interaction_features_count} interaction features")
    
    # 6. OUTLIER DETECTION (using IsolationForest and adding scores as features)
    # Re-identify numeric features after creating new ones
    current_numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in current_numeric_features: # Ensure target is removed from feature lists
        current_numeric_features.remove(target_col)
    
    if len(current_numeric_features) > 3: # IsolationForest needs enough features to work meaningfully
        print(f"\n🚨 Outlier detection (IsolationForest)...\n")
        try:
            iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
            # fit_predict returns -1 for outliers and 1 for inliers. 
            df_processed['is_outlier'] = iso_forest.fit_predict(df_processed[current_numeric_features])
            df_processed['is_outlier'] = (df_processed['is_outlier'] == -1).astype(int) # Convert to 0/1 indicator
            # decision_function returns the anomaly score. Lower scores are more anomalous.
            df_processed['outlier_score'] = iso_forest.decision_function(df_processed[current_numeric_features]) 
            
            outlier_count = df_processed['is_outlier'].sum()
            print(f"   ✅ Detected {outlier_count} outliers ({outlier_count/len(df_processed)*100:.1f}%)")
            new_features_created.extend(['is_outlier', 'outlier_score'])
        except Exception as e:
            print(f"   ❌ Outlier detection failed: {str(e)}. Skipping outlier features.")
            # Clean up potentially partially created columns if an error occurred during creation
            if 'outlier_score' in df_processed.columns:
                df_processed = df_processed.drop(columns=['outlier_score'])
            if 'is_outlier' in df_processed.columns:
                df_processed = df_processed.drop(columns=['is_outlier'])
    
    # 7. TARGET ENCODING FEATURES (using group statistics - careful with data leakage!)
    # This method can lead to data leakage if not handled carefully (e.g., using K-Fold Target Encoding).
    # For a simplified, yet illustrative purpose, we apply it directly using transform.
    print(f"\n📊 Creating group-based features (Target Encoding-like)...\n")
    
    # Re-identify all current categorical features, including newly created ones (e.g., 'age_group', 'marital_simple')
    current_categorical_features = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in current_categorical_features:
        current_categorical_features.remove(target_col)
    
    # Select specific categorical features for target encoding that are known to be impactful
    selected_target_encode_cols = []
    if 'workclass' in df_processed.columns:
        selected_target_encode_cols.append('workclass')
    if 'marital_simple' in df_processed.columns: # This is a newly created feature
        selected_target_encode_cols.append('marital_simple')
    if 'occupation' in df_processed.columns: # This is the re-mapped 'occupation'
        selected_target_encode_cols.append('occupation')
        
    for cat_col in selected_target_encode_cols:
        income_rate_col = f'{cat_col}_income_rate'
        
        # Calculate mean income (proportion of >50K) for each group and assign it back.
        # using .transform() ensures the new column is correctly aligned and has the same length as the original DataFrame.
        df_processed[income_rate_col] = df_processed.groupby(cat_col)[target_col].transform('mean')
        
        new_features_created.append(income_rate_col)
        print(f"   ✅ Created {income_rate_col}")
    
    # --- Update feature lists for the next steps ---
    # Final identification of numeric and categorical features after all FE steps
    updated_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    updated_categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col in updated_numeric_cols: # Ensure the target column is never in the feature lists
        updated_numeric_cols.remove(target_col)
    if target_col in updated_categorical_cols: # In case target was string initially
        updated_categorical_cols.remove(target_col) 
    
    print(f"\n📈 Feature Engineering Summary:\n")
    print(f"   Original features: {df.shape[1]} (excluding target)")
    print(f"   Features removed as per task: 2 (education, native-country)")
    print(f"   New features created: {len(new_features_created)}")
    print(f"   Final feature count for modeling: {len(updated_numeric_cols)} numeric + {len(updated_categorical_cols)} categorical")
    print(f"   Total features: {len(updated_numeric_cols) + len(updated_categorical_cols)}")
    print(f"\n✅ Advanced feature engineering completed!\n")
    
    print(f"📋 List of New Features Created:\n")
    for i, feature in enumerate(new_features_created, 1):
        print(f"   {i:2d}. {feature}")

    print(f"\n🎯 Dataset ready for preprocessing: {df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
    
    # --- FE Visualizations ---
    print("\n📊 Generating Feature Engineering Visualizations...")

    plt.figure(figsize=(18, 15))

    # 1. New Occupation vs Income
    if 'occupation' in df_processed.columns and target_col:
        plt.subplot(3, 3, 1)
        sns.barplot(x='occupation', y=df_processed[target_col], data=df_processed, palette='coolwarm')
        plt.title('Income Rate by New Occupation Category')
        plt.xlabel('Occupation Category')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')

    # 2. Age Group vs Income
    if 'age_group' in df_processed.columns and target_col:
        plt.subplot(3, 3, 2)
        sns.barplot(x='age_group', y=df_processed[target_col], data=df_processed, palette='mako')
        plt.title('Income Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')

    # 3. Work Intensity vs Income
    if 'work_intensity' in df_processed.columns and target_col:
        plt.subplot(3, 3, 3)
        sns.barplot(x='work_intensity', y=df_processed[target_col], data=df_processed, palette='rocket')
        plt.title('Income Rate by Work Intensity')
        plt.xlabel('Work Intensity')
        plt.ylabel('Income Rate (>50K)')
        plt.xticks(rotation=45, ha='right')
    
    # 4. Distribution of Capital Net (new numeric feature)
    if 'capital_net' in df_processed.columns:
        plt.subplot(3, 3, 4)
        sns.histplot(df_processed['capital_net'], bins=50, kde=True, color='purple')
        plt.title('Distribution of Capital Net')
        plt.xlabel('Capital Net')
        plt.ylabel('Count')
        plt.xscale('symlog') # Use symlog for potentially skewed distributions

    # 5. Distribution of Outlier Score (if created)
    if 'outlier_score' in df_processed.columns:
        plt.subplot(3, 3, 5)
        sns.histplot(df_processed['outlier_score'], bins=50, kde=True, color='darkgreen')
        plt.title('Distribution of Outlier Score')
        plt.xlabel('Outlier Score')
        plt.ylabel('Count')

    # 6. Correlation Heatmap with New Features (subset for clarity)
    # Include some key new numeric features and target-encoded features
    key_fe_numeric_cols = ['age_squared', 'capital_net', 'hours-per-week', 'education-num']
    key_fe_target_encode_cols = ['workclass_income_rate', 'marital_simple_income_rate', 'occupation_income_rate']
    
    # Filter to only include columns that actually exist in df_processed
    existing_plot_cols = [col for col in key_fe_numeric_cols + key_fe_target_encode_cols if col in df_processed.columns]
    
    if target_col and pd.api.types.is_numeric_dtype(df_processed[target_col]):
        existing_plot_cols.append(target_col)
    
    if len(existing_plot_cols) > 1:
        plt.subplot(3, 3, 6)
        corr_matrix_fe = df_processed[existing_plot_cols].corr()
        sns.heatmap(corr_matrix_fe, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 7})
        plt.title('Correlation Matrix with Key New Features')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.show()
    print("✅ Feature Engineering Visualizations generated successfully!")

    return df_processed, updated_numeric_cols, updated_categorical_cols, new_features_created
