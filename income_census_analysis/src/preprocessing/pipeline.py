"""
Advanced preprocessing and data splitting module.
Estratto ESATTAMENTE dal MODULO 4 del notebook originale.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif

# Imbalanced learning imports (with availability check)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

from ..config.settings import RANDOM_STATE, TEST_SIZE, CV_FOLDS, CV_REPEATS


def advanced_preprocessing_and_split(df: pd.DataFrame, target_col: str, 
                                     test_size: float, random_state: int) -> dict:
    """
    Preprocessing avanzato e split intelligente con multiple strategie di bilanciamento.
    Include la selezione delle feature (SelectKBest) nella pipeline.
    Include visualizzazioni per mostrare l'impatto di preprocessing e resampling.
    """
    print("\n4. ADVANCED PREPROCESSING & DATA SPLITTING")
    print("-" * 50)
    
    # Prepare features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col] # Target is already 0/1 from Modulo 2
    
    print(f"📊 Dataset for modeling:\n")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")
    print(f"   Target distribution: {(y==0).sum():,} (Class 0) vs {(y==1).sum():,} (Class 1)")
    
    # Class weights calculation (for models that support it)
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    imbalance_ratio = (y==1).sum() / (y==0).sum()
    
    print(f"   Imbalance ratio (Class 1 / Class 0): {imbalance_ratio:.3f}")
    print(f"   Calculated Class weights: {{0: {class_weight_dict[0]:.3f}, 1: {class_weight_dict[1]:.3f}}}")
    
    # Stratified train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✅ Stratified split completed:\n")
    print(f"   Training: {X_train.shape[0]:,} samples ({y_train.mean()*100:.1f}% positive)")
    print(f"   Validation: {X_val.shape[0]:,} samples ({y_val.mean()*100:.1f}% positive)")
    
    # Feature type identification for preprocessing (using the updated lists from FE)
    # Ensure these lists are derived from the df_engineered passed into this function
    numeric_features_for_pipeline = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features_for_pipeline = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_col in numeric_features_for_pipeline: # Ensure target is removed from feature lists
        numeric_features_for_pipeline.remove(target_col)
    if target_col in categorical_features_for_pipeline:
        categorical_features_for_pipeline.remove(target_col)

    print(f"\n📋 Feature preprocessing setup:\n")
    print(f"   Numeric features ({len(numeric_features_for_pipeline)}): {numeric_features_for_pipeline[:5]}{'...' if len(numeric_features_for_pipeline) > 5 else ''}")
    print(f"   Categorical features ({len(categorical_features_for_pipeline)}): {categorical_features_for_pipeline[:5]}{'...' if len(categorical_features_for_pipeline) > 5 else ''}")
    
    # Advanced preprocessing pipeline: StandardScaler for numeric, OneHotEncoder for categorical
    # Added SelectKBest for feature selection
    
    # Define a default k for SelectKBest. This can be tuned later.
    # A common strategy is to start with a percentage or a fixed number based on domain.
    # Let's say we want to select 80% of the features initially.
    # Number of features after OneHotEncoder is unknown before fitting, so SelectKBest should be applied AFTER OneHotEncoder.
    
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))])
    
    # The ColumnTransformer itself applies to original features.
    # SelectKBest needs to be applied after the features are all numeric.
    # For simplicity and to include it in a "pipeline-like" structure, we'll apply it after the initial transform.
    # In a full sklearn pipeline, this would be a bit more complex, often requiring a 'FunctionTransformer'
    # or ensuring SelectKBest is part of a larger 'Pipeline' that receives preprocessed data.
    # For this setup, we'll indicate it as a step after ColumnTransformer.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features_for_pipeline),
            ('cat', categorical_transformer, categorical_features_for_pipeline)
        ],
        remainder='passthrough' # Keep other columns (e.g., potential new numeric from FE not in original numeric_cols)
    )
    
    print(f"   ✅ Preprocessing pipeline (Scaler + OneHotEncoder) created.")
    
    # --- Apply preprocessing ---
    # Fit on X_train, transform X_train and X_val
    X_train_preprocessed_temp = preprocessor.fit_transform(X_train)
    X_val_preprocessed_temp = preprocessor.transform(X_val)

    # Get feature names BEFORE SelectKBest
    numeric_feature_names_transformed = numeric_features_for_pipeline
    categorical_feature_names_transformed = []
    if len(categorical_features_for_pipeline) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']['onehot']
        categorical_feature_names_transformed = cat_encoder.get_feature_names_out(categorical_features_for_pipeline).tolist()
    
    all_feature_names_before_selection = numeric_feature_names_transformed + categorical_feature_names_transformed
    
    # --- Feature Selection using SelectKBest ---
    print(f"   🔧 Applying Feature Selection (SelectKBest)...")
    # Using f_classif (ANOVA F-value) for classification tasks.
    # 'k' can be a fixed number, 'all', or a percentage. Let's make it dynamic based on the total features.
    # For example, select top 80% of features.
    
    # Ensure X_train_preprocessed_temp is not empty and has enough samples/features for SelectKBest
    if X_train_preprocessed_temp.shape[0] > 0 and X_train_preprocessed_temp.shape[1] > 0:
        num_features_after_preprocessing = X_train_preprocessed_temp.shape[1]
        k_to_select = min(50, num_features_after_preprocessing) # Example: select top 50 features, or all if less than 50

        # Adjust k based on common practices or sensitivity analysis.
        # For a dataset like Adult, usually good number of features, 50-80% is a reasonable starting point.
        # Let's target about 70% of the total preprocessed features as 'k'.
        k_to_select = int(num_features_after_preprocessing * 0.70)
        # Ensure k is at least 1 and not more than total features
        k_to_select = max(1, min(k_to_select, num_features_after_preprocessing))

        feature_selector = SelectKBest(score_func=f_classif, k=k_to_select)
        X_train_preprocessed = feature_selector.fit_transform(X_train_preprocessed_temp, y_train)
        X_val_preprocessed = feature_selector.transform(X_val_preprocessed_temp)
        
        # Get selected feature names
        selected_feature_indices = feature_selector.get_support(indices=True)
        all_feature_names_after_preprocessing = [all_feature_names_before_selection[i] for i in selected_feature_indices]
        
        print(f"   ✅ Selected {len(all_feature_names_after_preprocessing)} features out of {num_features_after_preprocessing} (k={k_to_select}).")
    else:
        print("   ⚠️ Skipping SelectKBest: preprocessed data is empty or invalid.")
        X_train_preprocessed = X_train_preprocessed_temp
        X_val_preprocessed = X_val_preprocessed_temp
        all_feature_names_after_preprocessing = all_feature_names_before_selection

    print(f"   ✅ Feature Selection (SelectKBest) applied.")
    
    # Create balanced datasets using multiple strategies if imblearn is available
    balanced_datasets = {}
    
    # --- Original (class weights only, but with preprocessed data) ---
    balanced_datasets['original'] = {
        'X_train': X_train_preprocessed,
        'y_train': y_train,
        'description': 'Original data (preprocessed & feature-selected) with class weights'
    }
    print(f"\n⚖️ Creating balanced datasets...\n")
    print(f"   ✅ Original: {len(y_train):,} samples (for class weights)")

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
            print(f"   ✅ SMOTE: {len(y_train):,} → {len(y_train_smote):,} samples")
        except Exception as e:
            print(f"   ❌ SMOTE failed: {str(e)}")
        
        # ADASYN oversampling
        try:
            adasyn = ADASYN(random_state=random_state, n_neighbors=5) 
            X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_preprocessed, y_train)
            balanced_datasets['adasyn'] = {
                'X_train': X_train_adasyn,
                'y_train': y_train_adasyn,
                'description': f'ADASYN oversampling (from {len(y_train):,} to {len(y_train_adasyn):,})'
            }
            print(f"   ✅ ADASYN: {len(y_train):,} → {len(y_train_adasyn):,} samples")
        except Exception as e:
            print(f"   ❌ ADASYN failed: {str(e)}")
        
        # SMOTEENN (Combined over + under sampling)
        try:
            smoteenn = SMOTEENN(random_state=random_state)
            X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train_preprocessed, y_train)
            balanced_datasets['smoteenn'] = {
                'X_train': X_train_smoteenn,
                'y_train': y_train_smoteenn,
                'description': f'SMOTEENN combined sampling (from {len(y_train):,} to {len(y_train_smoteenn):,})'
            }
            print(f"   ✅ SMOTEENN: {len(y_train):,} → {len(y_train_smoteenn):,} samples")
        except Exception as e:
            print(f"   ❌ SMOTEENN failed: {str(e)}")
    else:
        print(f"\n⚠️ imblearn not available, only 'original' data with class weights will be used.")
    
    # Cross-validation strategies
    cv_strategies = {
        'stratified_5fold': StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state),
        'repeated_stratified': RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=random_state)
    }
    
    print(f"\n🎯 Cross-validation strategies prepared:\n")
    for name, strategy in cv_strategies.items():
        print(f"   {name}: {strategy}")
    
    # --- Final Summary of Preprocessing ---
    print(f"\n📊 Preprocessing Summary:\n")
    print(f"   ✅ Train/Val split: {len(X_train):,}/{len(X_val):,}")
    print(f"   ✅ Features after preprocessing and selection: {len(all_feature_names_after_preprocessing)}")
    print(f"   ✅ Balanced datasets created: {len(balanced_datasets)}")
    print(f"   ✅ CV strategies prepared: {len(cv_strategies)}")
    
    print(f"\n✅ Preprocessing module completed!")
    print(f"🚀 Ready for model training with {len(balanced_datasets)} balanced datasets and {len(cv_strategies)} CV strategies.")

    # --- Preprocessing Visualizations ---
    print("\n📊 Generating Preprocessing Visualizations...")
    
    plt.figure(figsize=(24, 18)) # Increased figure size for more subplots

    # 1. Target Distribution after Train/Validation Split (CORRECTED ORDER)
    plt.subplot(3, 3, 1)
    # Ensure labels match actual values (0 or 1) and plot in specified order
    sns.countplot(x=y_train.astype(str), order=['0', '1'], palette='viridis') 
    plt.title(f'Target Distribution in Training Set ({len(y_train):,} samples)')
    plt.xlabel('Income Group (0: <=50K, 1: >50K)')
    plt.ylabel('Count')

    plt.subplot(3, 3, 2)
    sns.countplot(x=y_val.astype(str), order=['0', '1'], palette='viridis') # Explicit order
    plt.title(f'Target Distribution in Validation Set ({len(y_val):,} samples)')
    plt.xlabel('Income Group (0: <=50K, 1: >50K)')
    plt.ylabel('Count')
    
    # 3. Target Distribution after Resampling Strategies (if available)
    # Create a dynamic subplot grid for resampling results
    resampling_plots_count = sum(1 for s in ['smote', 'adasyn', 'smoteenn'] if s in balanced_datasets)
    current_subplot_offset = 3 # Start from subplot 3, row 1
    
    if resampling_plots_count > 0:
        for i, strategy_name in enumerate(['smote', 'adasyn', 'smoteenn']):
            if strategy_name in balanced_datasets:
                # Calculate subplot position (row 2, columns 1, 2, 3)
                plt.subplot(3, 3, current_subplot_offset + i) 
                sns.countplot(x=balanced_datasets[strategy_name]['y_train'].astype(str), order=['0', '1'], palette='cividis')
                plt.title(f'Target Distribution after {strategy_name.upper()} ({len(balanced_datasets[strategy_name]["y_train"]):,} samples)')
                plt.xlabel('Income Group (0: <=50K, 1: >50K)')
                plt.ylabel('Count')
    else:
        plt.subplot(3, 3, 3) # Placeholder if no resampling plots
        plt.text(0.5, 0.5, 'No Resampling Plots Available\n(Imblearn not installed or failed)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Resampling Strategies')


    # 4. Distribution of a scaled numeric feature (e.g., 'age' after scaling)
    # Get the index of 'age' in the preprocessed data if it exists
    age_col_idx = -1
    try:
        # Check if 'age' was selected by SelectKBest
        if 'age' in all_feature_names_after_preprocessing and X_train_preprocessed.shape[1] > 0:
            age_col_idx = all_feature_names_after_preprocessing.index('age')
    except ValueError:
        pass # 'age' might not be in the list if some other error occurred, or simply not found

    if age_col_idx != -1 and X_train_preprocessed.shape[1] > age_col_idx: # Ensure index is valid for X_train_preprocessed
        plt.subplot(3, 3, 7) # Adjusted subplot position to avoid overlap with dynamic resampling plots
        sns.histplot(X_train_preprocessed[:, age_col_idx], bins=30, kde=True, color='darkorange')
        plt.title('Distribution of Scaled Age (Training Set)')
        plt.xlabel('Scaled Age')
        plt.ylabel('Count')
        plt.axvline(x=0, color='gray', linestyle='--', label='Mean (approx. 0)')
        plt.axvline(x=1, color='red', linestyle=':', label='Std Dev (approx. 1)')
        plt.axvline(x=-1, color='red', linestyle=':')
        plt.legend()
    else:
        plt.subplot(3, 3, 7) # Placeholder if 'age' or data is not valid
        plt.text(0.5, 0.5, 'Scaled Numeric Feature\nDistribution Not Available', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Scaled Features Example')

    # 5. Boxplot of a scaled numeric feature vs target
    if age_col_idx != -1 and X_train_preprocessed.shape[1] > age_col_idx:
        scaled_age = X_train_preprocessed[:, age_col_idx]
        
        plt.subplot(3, 3, 8) # Adjusted subplot position
        # Create a temporary DataFrame for seaborn if y is not indexed
        temp_df_scaled_age = pd.DataFrame({'Target': y_train, 'Scaled_Age': scaled_age})
        sns.boxplot(x='Target', y='Scaled_Age', data=temp_df_scaled_age, palette='Set2')
        plt.title('Scaled Age vs. Target (Training Set)')
        plt.xlabel('Income Group (0: <=50K, 1: >50K)')
        plt.ylabel('Scaled Age')

    # 6. Distribution of a one-hot encoded categorical feature (example: 'sex')
    # This requires finding the correct column name in the ALL_FEATURE_NAMES_AFTER_PREPROCESSING list
    sex_feature_to_plot = None
    if 'sex_Male' in all_feature_names_after_preprocessing:
        sex_feature_to_plot = 'sex_Male'
    elif 'sex_Female' in all_feature_names_after_preprocessing: # Fallback if 'Male' was dropped (e.g. if 'Female' is minority class and drop='first' on 'Male')
        sex_feature_to_plot = 'sex_Female'
    
    if sex_feature_to_plot and X_train_preprocessed.shape[1] > 0:
        try:
            sex_col_idx = all_feature_names_after_preprocessing.index(sex_feature_to_plot)
            
            plt.subplot(3, 3, 9) # Adjusted subplot position
            # Plot the
