# =============================================================================
# TIM HACKATHON - MODULE 2: BASELINE LEARNING-TO-RANK MODELS (API FIXED)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# PART 1: FEATURE PREPARATION AND EVALUATION METRICS
# =============================================================================

class TIMRankingPipeline:
    """
    Professional Learning-to-Rank pipeline for TIM Hackathon
    Focuses on minimal feature engineering and robust evaluation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        self.cv_results = {}
        
    def prepare_features(self, df, fit_encoders=True):
        """Prepare minimal but effective features for baseline models"""
        if fit_encoders:
            print("PREPARING FEATURES FOR BASELINE MODELS")
            print("="*50)
        
        df_processed = df.copy()
        
        # 1. TEMPORAL FEATURES (already created in Module 1)
        temporal_features = ['month', 'week', 'dayofweek', 'is_weekend']
        if fit_encoders:
            print(f"Temporal features: {temporal_features}")
        
        # 2. ACTION FEATURES (encode categorical)
        categorical_features = ['action', 'action_type', 'action_category', 'action_subcategory']
        
        for col in categorical_features:
            if col in df_processed.columns:
                if fit_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df_processed[f'{col}_encoded'] = df_processed[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df_processed[f'{col}_encoded'] = -1
        
        # 3. PCA FEATURES (use as-is, they're already scaled)
        pca_features = [col for col in df_processed.columns if 'scaledPcaFeatures' in col]
        if fit_encoders:
            print(f"PCA features: {len(pca_features)}")
        
        # 4. INTERACTION FEATURES (minimal set)
        # Action category with temporal
        if 'action_category_encoded' in df_processed.columns:
            df_processed['category_month'] = df_processed['action_category_encoded'] * df_processed['month']
            df_processed['category_weekend'] = df_processed['action_category_encoded'] * df_processed['is_weekend']
        
        # Define feature sets
        encoded_categorical = [f'{col}_encoded' for col in categorical_features if col in df_processed.columns]
        interaction_features = ['category_month', 'category_weekend']
        
        # Remove interaction features that couldn't be created
        interaction_features = [f for f in interaction_features if f in df_processed.columns]
        
        feature_columns = temporal_features + encoded_categorical + pca_features + interaction_features
        
        # Ensure all features exist
        feature_columns = [col for col in feature_columns if col in df_processed.columns]
        
        if fit_encoders:
            print(f"Total features prepared: {len(feature_columns)}")
            print(f"  - Temporal: {len(temporal_features)}")
            print(f"  - Categorical (encoded): {len(encoded_categorical)}")
            print(f"  - PCA: {len(pca_features)}")
            print(f"  - Interactions: {len(interaction_features)}")
        
        self.feature_columns = feature_columns
        return df_processed[['num_telefono', 'data_contatto', 'target', 'was_offered'] + feature_columns]
    
    def calculate_ranking_metrics(self, y_true, y_pred, groups, was_offered=None):
        """Calculate comprehensive ranking metrics"""
        metrics = {}
        
        # Convert to groups for metric calculation
        unique_groups = np.unique(groups)
        
        # NDCG@K calculations
        ndcg_scores = {k: [] for k in [1, 3, 5]}
        map_scores = []
        mrr_scores = []
        hit_rates = {k: [] for k in [1, 3, 5]}
        
        for group in unique_groups:
            group_mask = groups == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            group_offered = was_offered[group_mask] if was_offered is not None else None
            
            # Skip if no positive samples
            if group_true.sum() == 0:
                continue
                
            # Calculate NDCG@K
            for k in [1, 3, 5]:
                ndcg_k = ndcg_score([group_true], [group_pred], k=k)
                ndcg_scores[k].append(ndcg_k)
            
            # Calculate MAP
            map_score = average_precision_score(group_true, group_pred)
            map_scores.append(map_score)
            
            # Calculate MRR
            sorted_indices = np.argsort(group_pred)[::-1]
            sorted_true = group_true[sorted_indices]
            first_relevant = np.where(sorted_true == 1)[0]
            if len(first_relevant) > 0:
                mrr_scores.append(1.0 / (first_relevant[0] + 1))
            else:
                mrr_scores.append(0.0)
            
            # Calculate Hit Rate@K
            for k in [1, 3, 5]:
                top_k_indices = sorted_indices[:k]
                hit_rates[k].append(1.0 if group_true[top_k_indices].sum() > 0 else 0.0)
        
        # Aggregate metrics
        for k in [1, 3, 5]:
            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0.0
            metrics[f'HitRate@{k}'] = np.mean(hit_rates[k]) if hit_rates[k] else 0.0
        
        metrics['MAP'] = np.mean(map_scores) if map_scores else 0.0
        metrics['MRR'] = np.mean(mrr_scores) if mrr_scores else 0.0
        
        return metrics
    
    def prepare_ranking_data(self, df):
        """Prepare data in ranking format for LightGBM and XGBoost"""
        # Create group information (customer-date combinations)
        df['group_id'] = df.groupby(['num_telefono', 'data_contatto']).ngroup()
        
        # Sort by group for ranking algorithms
        df_sorted = df.sort_values(['group_id', 'target'], ascending=[True, False])
        
        # Calculate group sizes
        group_sizes = df_sorted.groupby('group_id').size().values
        
        return df_sorted, group_sizes
    
    def train_lightgbm_ranker(self, train_df, valid_df=None, verbose=True):
        """Train LightGBM ranking model"""
        if verbose:
            print("TRAINING LIGHTGBM RANKER")
            print("="*30)
        
        # Prepare data
        train_sorted, train_group_sizes = self.prepare_ranking_data(train_df)
        
        X_train = train_sorted[self.feature_columns]
        y_train = train_sorted['target']
        
        # LightGBM parameters for ranking
        lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'boosting_type': 'gbdt',
            'num_leaves': 25,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'lambda_l1': 0.1,       
            'lambda_l2': 0.1,       
            'bagging_freq': 5,
            'min_gain_to_split': 0.1,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, group=train_group_sizes)
        
        valid_data = None
        if valid_df is not None:
            valid_sorted, valid_group_sizes = self.prepare_ranking_data(valid_df)
            X_valid = valid_sorted[self.feature_columns]
            y_valid = valid_sorted['target']
            valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_group_sizes, reference=train_data)
        
        # Setup callbacks
        callbacks = []
        if valid_data is not None:
            callbacks.append(lgb.early_stopping(50))
        if verbose:
            callbacks.append(lgb.log_evaluation(100))
        
        valid_sets = [valid_data] if valid_data is not None else None
        
        # Train model
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        
        self.models['lightgbm'] = model
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        self.feature_importance['lightgbm'] = dict(zip(self.feature_columns, importance))
        
        if verbose:
            print(f"LightGBM training completed")
            print(f"Best iteration: {model.best_iteration}")
        
        return model
    
    def train_xgboost_ranker(self, train_df, valid_df=None, verbose=True):
        """Train XGBoost ranking model"""
        if verbose:
            print("TRAINING XGBOOST RANKER")
            print("="*30)
        
        # Prepare data
        train_sorted, train_group_sizes = self.prepare_ranking_data(train_df)
        
        X_train = train_sorted[self.feature_columns]
        y_train = train_sorted['target']
        
        # XGBoost parameters for ranking
        xgb_params = {
            'objective': 'rank:ndcg',
            'eval_metric': ['ndcg@1', 'ndcg@3', 'ndcg@5'],
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbosity': 1 if verbose else 0
        }
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_group_sizes)
        
        evals = []
        if valid_df is not None:
            valid_sorted, valid_group_sizes = self.prepare_ranking_data(valid_df)
            X_valid = valid_sorted[self.feature_columns]
            y_valid = valid_sorted['target']
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            dvalid.set_group(valid_group_sizes)
            evals = [(dtrain, 'train'), (dvalid, 'valid')]
        else:
            evals = [(dtrain, 'train')]
        
        # Train model
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100 if verbose else False
        )
        
        self.models['xgboost'] = model
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        # Ensure all features are included
        feature_importance = {}
        for feature in self.feature_columns:
            feature_importance[feature] = importance.get(feature, 0.0)
        self.feature_importance['xgboost'] = feature_importance
        
        if verbose:
            print(f"XGBoost training completed")
            print(f"Best iteration: {model.best_iteration}")
        
        return model
    
    def predict_ranking(self, model_name, df):
        """Generate ranking predictions"""
        model = self.models[model_name]
        
        if model_name == 'lightgbm':
            predictions = model.predict(df[self.feature_columns])
        elif model_name == 'xgboost':
            dtest = xgb.DMatrix(df[self.feature_columns])
            predictions = model.predict(dtest)
        
        return predictions
    
    def evaluate_model(self, model_name, df, verbose=True):
        """Evaluate model performance"""
        predictions = self.predict_ranking(model_name, df)
        
        # Create group information
        df_eval = df.copy()
        df_eval['group_id'] = df_eval.groupby(['num_telefono', 'data_contatto']).ngroup()
        
        metrics = self.calculate_ranking_metrics(
            df_eval['target'].values,
            predictions,
            df_eval['group_id'].values,
            df_eval['was_offered'].values
        )
        
        if verbose:
            print(f"{model_name.upper()} EVALUATION RESULTS:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return metrics, predictions

# =============================================================================
# PART 2: FIXED CROSS-VALIDATION AND MODEL COMPARISON
# =============================================================================

def perform_cross_validation(pipeline, train_df, n_splits=5, verbose=True):
    """Perform customer-based cross-validation"""
    print("PERFORMING CROSS-VALIDATION")
    print("="*40)
    
    # Get unique customers and create a mapping dataset
    customers = train_df['num_telefono'].unique()
    
    # Create a dummy dataset for GroupKFold (one row per customer)
    customer_df = pd.DataFrame({'customer': customers})
    
    # Group K-Fold on the customer list
    gkf = GroupKFold(n_splits=n_splits)
    
    cv_results = {
        'lightgbm': {'metrics': [], 'predictions': []},
        'xgboost': {'metrics': [], 'predictions': []}
    }
    
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(customer_df, groups=customer_df['customer'])):
        if verbose:
            print(f"\nFold {fold + 1}/{n_splits}")
            print("-" * 20)
        
        # Get customer lists for this fold
        train_customers = customers[train_idx]
        valid_customers = customers[valid_idx]
        
        # Create fold datasets
        fold_train = train_df[train_df['num_telefono'].isin(train_customers)]
        fold_valid = train_df[train_df['num_telefono'].isin(valid_customers)]
        
        print(f"Train customers: {len(train_customers):,}, Valid customers: {len(valid_customers):,}")
        print(f"Train rows: {len(fold_train):,}, Valid rows: {len(fold_valid):,}")
        
        # Prepare features for this fold
        fold_train_processed = pipeline.prepare_features(fold_train, fit_encoders=True)
        fold_valid_processed = pipeline.prepare_features(fold_valid, fit_encoders=False)
        
        # Train and evaluate LightGBM
        lgb_model = pipeline.train_lightgbm_ranker(fold_train_processed, fold_valid_processed, verbose=False)
        lgb_metrics, lgb_preds = pipeline.evaluate_model('lightgbm', fold_valid_processed, verbose=False)
        cv_results['lightgbm']['metrics'].append(lgb_metrics)
        cv_results['lightgbm']['predictions'].append(lgb_preds)
        
        # Train and evaluate XGBoost
        xgb_model = pipeline.train_xgboost_ranker(fold_train_processed, fold_valid_processed, verbose=False)
        xgb_metrics, xgb_preds = pipeline.evaluate_model('xgboost', fold_valid_processed, verbose=False)
        cv_results['xgboost']['metrics'].append(xgb_metrics)
        cv_results['xgboost']['predictions'].append(xgb_preds)
        
        if verbose:
            print(f"  LightGBM NDCG@5: {lgb_metrics['NDCG@5']:.4f}")
            print(f"  XGBoost NDCG@5:  {xgb_metrics['NDCG@5']:.4f}")
    
    # Aggregate CV results
    cv_summary = {}
    for model_name in ['lightgbm', 'xgboost']:
        metrics_list = cv_results[model_name]['metrics']
        cv_summary[model_name] = {}
        
        # Calculate mean and std for each metric
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            cv_summary[model_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
    
    if verbose:
        print(f"\nCROSS-VALIDATION SUMMARY")
        print("="*40)
        for model_name, model_results in cv_summary.items():
            print(f"\n{model_name.upper()}:")
            for metric, stats in model_results.items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return cv_results, cv_summary

def create_model_comparison_visualizations(cv_summary, feature_importance):
    """Create comprehensive visualizations for model comparison"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. NDCG@K Comparison
    ax1 = plt.subplot(3, 3, 1)
    models = list(cv_summary.keys())
    ndcg_metrics = ['NDCG@1', 'NDCG@3', 'NDCG@5']
    
    x = np.arange(len(ndcg_metrics))
    width = 0.35
    
    for i, model in enumerate(models):
        means = [cv_summary[model][metric]['mean'] for metric in ndcg_metrics]
        stds = [cv_summary[model][metric]['std'] for metric in ndcg_metrics]
        ax1.bar(x + i * width, means, width, label=model.upper(), 
                yerr=stds, capsize=5, alpha=0.8)
    
    ax1.set_xlabel('NDCG Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('NDCG@K Comparison', fontweight='bold')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(ndcg_metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. All Metrics Comparison
    ax2 = plt.subplot(3, 3, 2)
    all_metrics = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'MAP', 'MRR', 'HitRate@1', 'HitRate@3', 'HitRate@5']
    
    lgb_means = [cv_summary['lightgbm'][m]['mean'] for m in all_metrics]
    xgb_means = [cv_summary['xgboost'][m]['mean'] for m in all_metrics]
    
    x = np.arange(len(all_metrics))
    ax2.plot(x, lgb_means, 'o-', label='LightGBM', linewidth=2, markersize=6)
    ax2.plot(x, xgb_means, 's-', label='XGBoost', linewidth=2, markersize=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_metrics, rotation=45, ha='right')
    ax2.set_ylabel('Score')
    ax2.set_title('All Metrics Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cross-Validation Stability
    ax3 = plt.subplot(3, 3, 3)
    ndcg5_lgb = cv_summary['lightgbm']['NDCG@5']['values']
    ndcg5_xgb = cv_summary['xgboost']['NDCG@5']['values']
    
    folds = range(1, len(ndcg5_lgb) + 1)
    ax3.plot(folds, ndcg5_lgb, 'o-', label='LightGBM', linewidth=2, markersize=8)
    ax3.plot(folds, ndcg5_xgb, 's-', label='XGBoost', linewidth=2, markersize=8)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('NDCG@5')
    ax3.set_title('Cross-Validation Stability', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Importance Comparison (Top 15)
    ax4 = plt.subplot(3, 3, 4)
    
    # Get top features from LightGBM
    lgb_importance = feature_importance['lightgbm']
    top_features = sorted(lgb_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    feature_names = [f[0] for f in top_features]
    
    lgb_values = [lgb_importance[f] for f in feature_names]
    xgb_values = [feature_importance['xgboost'].get(f, 0) for f in feature_names]
    
    # Normalize for comparison
    lgb_values = np.array(lgb_values) / max(lgb_values) if max(lgb_values) > 0 else lgb_values
    xgb_values = np.array(xgb_values) / max(xgb_values) if max(xgb_values) > 0 else xgb_values
    
    y = np.arange(len(feature_names))
    ax4.barh(y - 0.2, lgb_values, 0.4, label='LightGBM', alpha=0.8)
    ax4.barh(y + 0.2, xgb_values, 0.4, label='XGBoost', alpha=0.8)
    ax4.set_yticks(y)
    ax4.set_yticklabels([f.replace('scaledPcaFeatures_', 'PCA_')[:15] for f in feature_names], fontsize=9)
    ax4.set_xlabel('Normalized Importance')
    ax4.set_title('Top 15 Feature Importance', fontweight='bold')
    ax4.legend()
    
    # 5. Model Performance Distribution
    ax5 = plt.subplot(3, 3, 5)
    
    lgb_ndcg5_values = cv_summary['lightgbm']['NDCG@5']['values']
    xgb_ndcg5_values = cv_summary['xgboost']['NDCG@5']['values']
    
    ax5.boxplot([lgb_ndcg5_values, xgb_ndcg5_values], labels=['LightGBM', 'XGBoost'])
    ax5.set_ylabel('NDCG@5')
    ax5.set_title('Performance Distribution', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Hit Rate Comparison
    ax6 = plt.subplot(3, 3, 6)
    hit_metrics = ['HitRate@1', 'HitRate@3', 'HitRate@5']
    
    lgb_hit = [cv_summary['lightgbm'][m]['mean'] for m in hit_metrics]
    xgb_hit = [cv_summary['xgboost'][m]['mean'] for m in hit_metrics]
    
    x = np.arange(len(hit_metrics))
    ax6.bar(x - 0.2, lgb_hit, 0.4, label='LightGBM', alpha=0.8)
    ax6.bar(x + 0.2, xgb_hit, 0.4, label='XGBoost', alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(hit_metrics)
    ax6.set_ylabel('Hit Rate')
    ax6.set_title('Hit Rate Comparison', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Feature Category Importance
    ax7 = plt.subplot(3, 3, 7)
    
    # Categorize features
    feature_categories = {
        'Temporal': ['month', 'week', 'dayofweek', 'is_weekend'],
        'Action': ['action_encoded', 'action_type_encoded', 'action_category_encoded', 'action_subcategory_encoded'],
        'Interaction': ['category_month', 'category_weekend'],
        'PCA': [f for f in lgb_importance.keys() if 'scaledPcaFeatures' in f]
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        importance_sum = sum(lgb_importance.get(f, 0) for f in features if f in lgb_importance)
        category_importance[category] = importance_sum
    
    categories = list(category_importance.keys())
    importance_values = list(category_importance.values())
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    bars = ax7.bar(categories, importance_values, color=colors, alpha=0.8)
    ax7.set_ylabel('Total Importance')
    ax7.set_title('Feature Category Importance', fontweight='bold')
    ax7.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, importance_values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_values)*0.01,
                 f'{value:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 8. Statistical Significance Test
    ax8 = plt.subplot(3, 3, 8)
    
    # Perform paired t-test for NDCG@5
    lgb_ndcg5 = cv_summary['lightgbm']['NDCG@5']['values']
    xgb_ndcg5 = cv_summary['xgboost']['NDCG@5']['values']
    
    t_stat, p_value = stats.ttest_rel(lgb_ndcg5, xgb_ndcg5)
    
    # Visualize the comparison
    models_names = ['LightGBM', 'XGBoost']
    means = [np.mean(lgb_ndcg5), np.mean(xgb_ndcg5)]
    stds = [np.std(lgb_ndcg5), np.std(xgb_ndcg5)]
    
    bars = ax8.bar(models_names, means, yerr=stds, capsize=10, 
                   color=['#3498db', '#e74c3c'], alpha=0.8)
    ax8.set_ylabel('NDCG@5')
    ax8.set_title(f'Statistical Comparison\n(p-value: {p_value:.4f})', fontweight='bold')
    
    # Add significance indicator
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    ax8.text(0.5, max(means) + max(stds) + 0.01, significance, 
             ha='center', va='bottom', fontweight='bold',
             color='red' if p_value < 0.05 else 'green')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax8.text(bar.get_x() + bar.get_width()/2, mean + std + 0.005,
                 f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 9. Summary Table
    ax9 = plt.subplot(3, 3, 9)
    
    # Create summary statistics
    summary_data = []
    for model in ['lightgbm', 'xgboost']:
        ndcg5_mean = cv_summary[model]['NDCG@5']['mean']
        ndcg5_std = cv_summary[model]['NDCG@5']['std']
        map_mean = cv_summary[model]['MAP']['mean']
        hit1_mean = cv_summary[model]['HitRate@1']['mean']
        
        summary_data.append([
            model.upper(),
            f"{ndcg5_mean:.4f}±{ndcg5_std:.4f}",
            f"{map_mean:.4f}",
            f"{hit1_mean:.4f}"
        ])
    
    # Add statistical test result
    summary_data.append(['Statistical Test', f"p-value: {p_value:.4f}", significance, ''])
    
    table = ax9.table(cellText=summary_data,
                      colLabels=['Model', 'NDCG@5 (±std)', 'MAP', 'HitRate@1'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            elif i == len(summary_data):  # Statistical test row
                cell.set_facecolor('#f39c12')
                cell.set_text_props(weight='bold')
            else:
               cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
   
    ax9.set_title('Performance Summary', fontweight='bold')
    ax9.axis('off')
   
    plt.tight_layout()
    plt.savefig('tim_hackathon_module2_baseline_models.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# PART 3: MAIN EXECUTION
# =============================================================================

def main():
   """Main execution function for Module 2"""
   print("="*80)
   print("TIM HACKATHON - MODULE 2: BASELINE LEARNING-TO-RANK MODELS")
   print("="*80)
   
   # Verify we have the data from Module 1
   try:
       # These should be available from Module 1
       train_df_available = 'train_df' in globals()
       test_df_available = 'test_df' in globals()
       
       if not (train_df_available and test_df_available):
           raise ValueError("Train/test data not found. Please run Module 1 first.")
       
       print(f"Data loaded from Module 1:")
       print(f"  Train dataset: {train_df.shape}")
       print(f"  Test dataset: {test_df.shape}")
       
   except Exception as e:
       print(f"Error: {e}")
       print("Please ensure Module 1 has been executed successfully")
       return None
   
   # Initialize pipeline
   pipeline = TIMRankingPipeline(random_state=RANDOM_STATE)
   
   # Step 1: Prepare features
   print(f"\nSTEP 1: Feature preparation...")
   train_processed = pipeline.prepare_features(train_df, fit_encoders=True)
   test_processed = pipeline.prepare_features(test_df, fit_encoders=False)
   
   print(f"Processed train shape: {train_processed.shape}")
   print(f"Processed test shape: {test_processed.shape}")
   print(f"Features available: {len(pipeline.feature_columns)}")
   
   # Step 2: Cross-validation
   print(f"\nSTEP 2: Cross-validation...")
   cv_results, cv_summary = perform_cross_validation(pipeline, train_processed, n_splits=5, verbose=True)
   
   # Step 3: Train final models on full training data
   print(f"\nSTEP 3: Training final models on full training data...")
   
   # Train LightGBM
   lgb_model = pipeline.train_lightgbm_ranker(train_processed, verbose=True)
   
   # Train XGBoost  
   xgb_model = pipeline.train_xgboost_ranker(train_processed, verbose=True)
   
   # Step 4: Final evaluation on test set
   print(f"\nSTEP 4: Final evaluation on test set...")
   
   lgb_test_metrics, lgb_test_preds = pipeline.evaluate_model('lightgbm', test_processed, verbose=True)
   xgb_test_metrics, xgb_test_preds = pipeline.evaluate_model('xgboost', test_processed, verbose=True)
   
   # Step 5: Model comparison and selection
   print(f"\nSTEP 5: Model comparison and selection...")
   
   # Compare CV performance
   lgb_cv_ndcg5 = cv_summary['lightgbm']['NDCG@5']['mean']
   xgb_cv_ndcg5 = cv_summary['xgboost']['NDCG@5']['mean']
   
   lgb_test_ndcg5 = lgb_test_metrics['NDCG@5']
   xgb_test_ndcg5 = xgb_test_metrics['NDCG@5']
   
   print(f"Cross-validation NDCG@5:")
   print(f"  LightGBM: {lgb_cv_ndcg5:.4f}")
   print(f"  XGBoost:  {xgb_cv_ndcg5:.4f}")
   
   print(f"Test set NDCG@5:")
   print(f"  LightGBM: {lgb_test_ndcg5:.4f}")
   print(f"  XGBoost:  {xgb_test_ndcg5:.4f}")
   
   # Select best model
   best_model = 'lightgbm' if lgb_test_ndcg5 > xgb_test_ndcg5 else 'xgboost'
   print(f"\nBest performing model: {best_model.upper()}")
   
   # Step 6: Create visualizations
   print(f"\nSTEP 6: Creating comprehensive visualizations...")
   create_model_comparison_visualizations(cv_summary, pipeline.feature_importance)
   
   # Step 7: Final summary and recommendations
   print(f"\nSTEP 7: Generating final summary...")
   generate_baseline_summary(cv_summary, lgb_test_metrics, xgb_test_metrics, best_model, pipeline)
   
   print(f"\n{'='*80}")
   print("MODULE 2 COMPLETED SUCCESSFULLY")
   print(f"{'='*80}")
   print("Generated files:")
   print("  - tim_hackathon_module2_baseline_models.png")
   print("Baseline models ready:")
   print(f"  - Best model: {best_model.upper()}")
   print(f"  - Test NDCG@5: {lgb_test_ndcg5 if best_model == 'lightgbm' else xgb_test_ndcg5:.4f}")
   print("Ready for Module 3: Specialized Ranking Models!")
   
   return pipeline, cv_summary, lgb_test_metrics, xgb_test_metrics, best_model

def generate_baseline_summary(cv_summary, lgb_test_metrics, xgb_test_metrics, best_model, pipeline):
   """Generate comprehensive baseline summary"""
   print("BASELINE MODELS SUMMARY REPORT")
   print("="*50)
   
   print(f"\nMODEL PERFORMANCE COMPARISON:")
   print(f"{'Metric':<15} {'LightGBM':<12} {'XGBoost':<12} {'Best':<10}")
   print("-" * 55)
   
   metrics_to_compare = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'MAP', 'MRR', 'HitRate@1']
   
   for metric in metrics_to_compare:
       lgb_score = lgb_test_metrics[metric]
       xgb_score = xgb_test_metrics[metric]
       best_score = max(lgb_score, xgb_score)
       best_model_name = 'LightGBM' if lgb_score > xgb_score else 'XGBoost'
       
       print(f"{metric:<15} {lgb_score:<12.4f} {xgb_score:<12.4f} {best_model_name:<10}")
   
   print(f"\nCROSS-VALIDATION STABILITY:")
   for model_name in ['lightgbm', 'xgboost']:
       ndcg5_mean = cv_summary[model_name]['NDCG@5']['mean']
       ndcg5_std = cv_summary[model_name]['NDCG@5']['std']
       cv_stability = "High" if ndcg5_std < 0.01 else "Medium" if ndcg5_std < 0.02 else "Low"
       print(f"  {model_name.upper()}: {ndcg5_mean:.4f} ± {ndcg5_std:.4f} (Stability: {cv_stability})")
   
   print(f"\nFEATURE ANALYSIS:")
   print(f"  Total features used: {len(pipeline.feature_columns)}")
   
   # Top 5 features from best model
   best_importance = pipeline.feature_importance[best_model]
   top_features = sorted(best_importance.items(), key=lambda x: x[1], reverse=True)[:5]
   
   print(f"  Top 5 features ({best_model.upper()}):")
   for i, (feature, importance) in enumerate(top_features, 1):
       feature_display = feature.replace('scaledPcaFeatures_', 'PCA_')
       print(f"    {i}. {feature_display}: {importance:.1f}")
   
   print(f"\nRECOMMENDATIONS FOR MODULE 3:")
   
   # Performance-based recommendations
   best_ndcg5 = max(lgb_test_metrics['NDCG@5'], xgb_test_metrics['NDCG@5'])
   
   if best_ndcg5 > 0.3:
       print(f"  ✅ Strong baseline performance (NDCG@5: {best_ndcg5:.4f})")
       print(f"  📈 Focus on advanced feature engineering for incremental gains")
       print(f"  🎯 Consider ensemble methods")
   elif best_ndcg5 > 0.2:
       print(f"  ⚠️ Moderate baseline performance (NDCG@5: {best_ndcg5:.4f})")
       print(f"  🔧 Significant improvement potential with feature engineering")
       print(f"  📊 Explore customer segmentation features")
   else:
       print(f"  ❌ Low baseline performance (NDCG@5: {best_ndcg5:.4f})")
       print(f"  🚨 Need substantial feature engineering")
       print(f"  🔍 Investigate data quality and feature selection")
   
   # Feature-based recommendations
   pca_importance = sum(imp for feat, imp in best_importance.items() if 'scaledPcaFeatures' in feat)
   total_importance = sum(best_importance.values())
   pca_ratio = pca_importance / total_importance if total_importance > 0 else 0
   
   print(f"\nFEATURE ENGINEERING PRIORITIES:")
   print(f"  📊 PCA features contribute {pca_ratio:.1%} of importance")
   
   if pca_ratio > 0.7:
       print(f"  🎯 High PCA dominance - explore PCA interactions and clustering")
   elif pca_ratio > 0.4:
       print(f"  ⚖️ Balanced feature importance - enhance all categories")
   else:
       print(f"  📈 Low PCA usage - may need PCA feature engineering")
   
   # Stability recommendations
   lgb_stability = cv_summary['lightgbm']['NDCG@5']['std']
   xgb_stability = cv_summary['xgboost']['NDCG@5']['std']
   avg_stability = (lgb_stability + xgb_stability) / 2
   
   if avg_stability < 0.01:
       print(f"  ✅ High model stability - reliable for production")
   elif avg_stability < 0.02:
       print(f"  ⚠️ Moderate stability - monitor performance variance")
   else:
       print(f"  ❌ Low stability - investigate data consistency")
   
   print(f"\nNEXT STEPS:")
   print(f"  1. Use {best_model.upper()} as baseline reference")
   print(f"  2. Implement advanced feature engineering in Module 3")
   print(f"  3. Target NDCG@5 improvement of +0.05-0.10")
   print(f"  4. Maintain model stability during enhancements")

# Execute Module 2
if __name__ == "__main__":
   pipeline, cv_summary, lgb_test_metrics, xgb_test_metrics, best_model = main()
