# =============================================================================
# TIM HACKATHON - MODULE 7: OPTIMIZED MODELS (USING PRE-TUNED PARAMETERS)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score, average_precision_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class TIMOptimizedModels:
    """
    TIM Optimized Models using pre-tuned hyperparameters
    
    LightGBM Best: CV NDCG@5 = 0.7638 ± 0.0028 (Trial 45)
    XGBoost Best: CV NDCG@5 = 0.7673 (Trial 3) 
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_columns = None
        self.models = {}
        self.training_histories = {}
        
        # Pre-tuned optimal parameters from optimization runs
        self.optimal_params = {
            'lightgbm': {
                # Trial 45: Best CV NDCG@5 = 0.7638 ± 0.0028  
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3, 5],
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': random_state,
                'force_row_wise': True,
                # Optimized hyperparameters
                'num_leaves': 97,
                'learning_rate': 0.029370822768718483,
                'feature_fraction': 0.8598257710495323,
                'bagging_fraction': 0.501871596766175,
                'bagging_freq': 4,
                'lambda_l1': 0.9348865940568215,
                'lambda_l2': 0.1626090480862173,
                'min_gain_to_split': 0.5866986953613079,
                'min_child_samples': 49,
                'max_depth': 14
            },
            'xgboost': {
                # Trial 3: Best CV NDCG@5 = 0.7673
                'objective': 'rank:ndcg',
                'eval_metric': 'ndcg@5',
                'random_state': random_state,
                'verbosity': 0,
                # Optimized hyperparameters
                'eta': 0.05954553793888989,
                'max_depth': 13,
                'subsample': 0.5998368910791798,
                'colsample_bytree': 0.7571172192068059,
                'lambda': 5.924145688620425,
                'alpha': 0.46450412719997725,
                'min_child_weight': 13,
                'gamma': 0.8526206184364576
            }
        }
    
    def prepare_features_consistent(self, df, fit_encoders=True):
        """Consistent feature preparation with Module 6"""
        if fit_encoders:
            print("🔧 PREPARING FEATURES (CONSISTENT WITH MODULE 6)")
            print("="*48)
        
        df_processed = df.copy()
        
        exclude_cols = [
            'num_telefono', 'data_contatto', 'target', 'was_offered', 'action', 'group_id',
            'weekend_key', 'month_key', 'action_category', 'month', 'is_weekend'
        ]
        
        feature_columns = [col for col in df_processed.columns if col not in exclude_cols]
        
        for col in feature_columns.copy():
            if df_processed[col].dtype == 'object':
                feature_columns.remove(col)
        
        self.feature_columns = feature_columns
        
        if fit_encoders:
            print(f"Features prepared: {len(feature_columns)}")
            pca_features = len([f for f in feature_columns if 'scaledPcaFeatures' in f])
            enhanced_features = len([f for f in feature_columns if any(kw in f for kw in ['train_', 'history_', 'weekend_', 'month_'])])
            print(f"  PCA features: {pca_features}")
            print(f"  Enhanced features: {enhanced_features}")
            print(f"  Other features: {len(feature_columns) - pca_features - enhanced_features}")
        
        return df_processed, feature_columns
    
    def prepare_ranking_data(self, df, verbose=False):
        """Prepare ranking data for training"""
        df_processed, feature_columns = self.prepare_features_consistent(df, fit_encoders=verbose)
        
        df_sorted = df_processed.copy()
        df_sorted['group_id'] = df_sorted.groupby(['num_telefono', 'data_contatto']).ngroup()
        df_sorted = df_sorted.sort_values(['group_id', 'target'], ascending=[True, False])
        
        X = df_sorted[feature_columns].values
        y = df_sorted['target'].values
        group_ids = df_sorted['group_id'].values
        
        unique_groups, group_counts = np.unique(group_ids, return_counts=True)
        group_sizes = group_counts
        
        # Customer mapping for proper CV
        customers = df_sorted[['group_id', 'num_telefono']].drop_duplicates()
        group_to_customer = dict(zip(customers['group_id'], customers['num_telefono']))
        
        if verbose:
            print(f"Ranking data prepared:")
            print(f"  Samples: {len(X):,}")
            print(f"  Features: {len(feature_columns)}")
            print(f"  Groups: {len(unique_groups):,}")
            print(f"  Customers: {len(df_sorted['num_telefono'].unique()):,}")
            print(f"  Avg group size: {np.mean(group_counts):.1f}")
        
        return X, y, group_ids, group_sizes, group_to_customer, feature_columns
    
    def perform_cross_validation(self, X, y, group_sizes, group_to_customer, verbose=True):
        """
        Perform robust cross-validation with optimized parameters
        """
        if verbose:
            print("📊 CROSS-VALIDATION WITH OPTIMIZED PARAMETERS")
            print("="*47)
        
        # Get unique customers for proper splitting
        unique_customers = list(set(group_to_customer.values()))
        gkf = GroupKFold(n_splits=5)
        
        customer_array = np.array(unique_customers)
        dummy_X = np.arange(len(unique_customers)).reshape(-1, 1)
        dummy_y = np.zeros(len(unique_customers))
        
        cv_results = {
            'lightgbm': {'scores': [], 'details': []},
            'xgboost': {'scores': [], 'details': []}
        }
        
        for fold, (train_customer_idx, val_customer_idx) in enumerate(gkf.split(dummy_X, dummy_y, customer_array)):
            
            if verbose:
                print(f"\nFold {fold + 1}/5:")
            
            # Get customer lists
            train_customers = set(customer_array[train_customer_idx])
            val_customers = set(customer_array[val_customer_idx])
            
            # Map to groups
            train_groups = [g for g, c in group_to_customer.items() if c in train_customers]
            val_groups = [g for g, c in group_to_customer.items() if c in val_customers]
            
            # Create masks
            train_mask = np.isin(np.repeat(np.arange(len(group_sizes)), group_sizes), train_groups)
            val_mask = np.isin(np.repeat(np.arange(len(group_sizes)), group_sizes), val_groups)
            
            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask]
            X_val_fold = X[val_mask]
            y_val_fold = y[val_mask]
            
            # Group sizes for each fold
            train_group_indices = [i for i, g in enumerate(np.arange(len(group_sizes))) if g in train_groups]
            val_group_indices = [i for i, g in enumerate(np.arange(len(group_sizes))) if g in val_groups]
            
            train_group_sizes = group_sizes[train_group_indices]
            val_group_sizes = group_sizes[val_group_indices]
            
            # Train LightGBM
            try:
                train_data = lgb.Dataset(X_train_fold, label=y_train_fold, group=train_group_sizes)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold, group=val_group_sizes, reference=train_data)
                
                lgb_model = lgb.train(
                    self.optimal_params['lightgbm'],
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                lgb_pred = lgb_model.predict(X_val_fold)
                val_group_mapping = np.repeat(range(len(val_group_sizes)), val_group_sizes)
                lgb_ndcg5 = self.calculate_ndcg(y_val_fold, lgb_pred, val_group_mapping, k=5)
                
                cv_results['lightgbm']['scores'].append(lgb_ndcg5)
                cv_results['lightgbm']['details'].append({
                    'fold': fold,
                    'ndcg5': lgb_ndcg5,
                    'best_iteration': lgb_model.best_iteration
                })
                
                if verbose:
                    print(f"  LightGBM NDCG@5: {lgb_ndcg5:.4f} (iter: {lgb_model.best_iteration})")
                
            except Exception as e:
                print(f"  LightGBM fold {fold} failed: {e}")
                cv_results['lightgbm']['scores'].append(0.50)
            
            # Train XGBoost
            try:
                dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
                dtrain.set_group(train_group_sizes)
                
                dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
                dval.set_group(val_group_sizes)
                
                xgb_model = xgb.train(
                    self.optimal_params['xgboost'],
                    dtrain,
                    num_boost_round=500,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                
                xgb_pred = xgb_model.predict(dval)
                xgb_ndcg5 = self.calculate_ndcg(y_val_fold, xgb_pred, val_group_mapping, k=5)
                
                cv_results['xgboost']['scores'].append(xgb_ndcg5)
                cv_results['xgboost']['details'].append({
                    'fold': fold,
                    'ndcg5': xgb_ndcg5,
                    'best_iteration': xgb_model.best_iteration
                })
                
                if verbose:
                    print(f"  XGBoost NDCG@5:  {xgb_ndcg5:.4f} (iter: {xgb_model.best_iteration})")
                
            except Exception as e:
                print(f"  XGBoost fold {fold} failed: {e}")
                cv_results['xgboost']['scores'].append(0.50)
        
        # Calculate CV statistics
        cv_summary = {}
        for model_name in ['lightgbm', 'xgboost']:
            scores = cv_results[model_name]['scores']
            cv_summary[model_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        if verbose:
            print(f"\nCROSS-VALIDATION SUMMARY:")
            print("="*27)
            for model_name, stats in cv_summary.items():
                print(f"{model_name.upper()}:")
                print(f"  CV NDCG@5: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Stability: {'HIGH' if stats['std'] < 0.02 else 'MEDIUM' if stats['std'] < 0.05 else 'LOW'}")
        
        return cv_results, cv_summary
    
    def train_optimized_models(self, X, y, group_sizes, verbose=True):
        """Train final models with optimized parameters and validation tracking"""
        if verbose:
            print("🚀 TRAINING OPTIMIZED FINAL MODELS")
            print("="*34)
        
        models = {}
        training_histories = {}
        
        # Create train/validation split for learning curves
        n_groups = len(group_sizes)
        val_size = int(n_groups * 0.15)
        val_groups = np.random.choice(n_groups, val_size, replace=False)
        train_groups = np.setdiff1d(np.arange(n_groups), val_groups)
        
        # Create splits
        train_mask = np.isin(np.repeat(np.arange(n_groups), group_sizes), train_groups)
        val_mask = np.isin(np.repeat(np.arange(n_groups), group_sizes), val_groups)
        
        X_train_final = X[train_mask]
        y_train_final = y[train_mask]
        X_val_final = X[val_mask]
        y_val_final = y[val_mask]
        
        train_group_sizes_final = group_sizes[train_groups]
        val_group_sizes_final = group_sizes[val_groups]
        
        # Train LightGBM
        if verbose:
            print("Training LightGBM with optimized parameters...")
        
        train_data = lgb.Dataset(X_train_final, label=y_train_final, group=train_group_sizes_final)
        val_data = lgb.Dataset(X_val_final, label=y_val_final, group=val_group_sizes_final, reference=train_data)
        
        evals_result = {}
        lgb_model = lgb.train(
            self.optimal_params['lightgbm'],
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(100),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        models['lightgbm_optimized'] = lgb_model
        training_histories['lightgbm'] = evals_result
        
        if verbose:
            print(f"✅ LightGBM trained - best iteration: {lgb_model.best_iteration}")
            
            # Analyze overfitting
            if len(evals_result['train']['ndcg@5']) > 10:
                final_train = np.mean(evals_result['train']['ndcg@5'][-10:])
                final_val = np.mean(evals_result['val']['ndcg@5'][-10:])
                overfitting_gap = ((final_train - final_val) / final_train) * 100
                print(f"   Overfitting analysis: Train={final_train:.4f}, Val={final_val:.4f}, Gap={overfitting_gap:.1f}%")
        
        # Train XGBoost
        if verbose:
            print("Training XGBoost with optimized parameters...")
        
        dtrain = xgb.DMatrix(X_train_final, label=y_train_final)
        dtrain.set_group(train_group_sizes_final)
        
        dval = xgb.DMatrix(X_val_final, label=y_val_final)
        dval.set_group(val_group_sizes_final)
        
        evals_result = {}
        xgb_model = xgb.train(
            self.optimal_params['xgboost'],
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=100,
            evals_result=evals_result
        )
        
        models['xgboost_optimized'] = xgb_model
        training_histories['xgboost'] = evals_result
        
        if verbose:
            print(f"✅ XGBoost trained - best iteration: {xgb_model.best_iteration}")
            
            # Analyze overfitting
            if len(evals_result['train']['ndcg@5']) > 10:
                final_train = np.mean(evals_result['train']['ndcg@5'][-10:])
                final_val = np.mean(evals_result['val']['ndcg@5'][-10:])
                overfitting_gap = ((final_train - final_val) / final_train) * 100
                print(f"   Overfitting analysis: Train={final_train:.4f}, Val={final_val:.4f}, Gap={overfitting_gap:.1f}%")
        
        self.models = models
        self.training_histories = training_histories
        
        return models, training_histories
    
    def calculate_ndcg(self, y_true, y_pred, groups, k=5):
        """Calculate NDCG@k properly"""
        ndcg_scores = []
        
        for group in np.unique(groups):
            group_mask = groups == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            if group_true.sum() == 0 or len(group_true) < 2:
                continue
                
            try:
                ndcg_k = ndcg_score([group_true], [group_pred], k=k)
                ndcg_scores.append(ndcg_k)
            except ValueError:
                continue
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def evaluate_models_comprehensive(self, models, test_df, verbose=True):
        """Comprehensive evaluation with confidence intervals"""
        if verbose:
            print("📊 COMPREHENSIVE MODEL EVALUATION")
            print("="*35)
        
        # Prepare test data
        X_test, y_test, groups_test, group_sizes_test, _, _ = self.prepare_ranking_data(test_df, verbose=False)
        
        results = {}
        
        for model_name, model in models.items():
            # Generate predictions
            if 'lightgbm' in model_name:
                predictions = model.predict(X_test)
            elif 'xgboost' in model_name:
                dtest = xgb.DMatrix(X_test)
                predictions = model.predict(dtest)
            
            # Calculate comprehensive metrics
            test_group_mapping = np.repeat(range(len(group_sizes_test)), group_sizes_test)
            metrics = self.calculate_comprehensive_metrics(y_test, predictions, test_group_mapping)
            
            # Calculate confidence intervals for NDCG@5
            ndcg5_with_ci, (ci_lower, ci_upper), n_groups = self.calculate_ndcg_with_confidence(
                y_test, predictions, test_group_mapping, k=5
            )
            
            metrics['NDCG@5_CI_lower'] = ci_lower
            metrics['NDCG@5_CI_upper'] = ci_upper
            metrics['n_evaluated_groups'] = n_groups
            
            results[model_name] = metrics
            
            if verbose:
                model_display = model_name.replace('_optimized', '').upper()
                print(f"\n{model_display} OPTIMIZED Results:")
                print("-" * (len(model_display) + 18))
                for metric, value in metrics.items():
                    if not metric.endswith('_CI_lower') and not metric.endswith('_CI_upper') and metric != 'n_evaluated_groups':
                        print(f"  {metric}: {value:.4f}")
                print(f"  NDCG@5 CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"  Groups evaluated: {n_groups:,}")
        
        return results
    
    def calculate_ndcg_with_confidence(self, y_true, y_pred, groups, k=5, confidence_level=0.95):
        """Calculate NDCG with confidence intervals"""
        ndcg_scores = []
        
        for group in np.unique(groups):
            group_mask = groups == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            if group_true.sum() == 0 or len(group_true) < 2:
                continue
                
            try:
                ndcg_k = ndcg_score([group_true], [group_pred], k=k)
                if 0.0 <= ndcg_k <= 1.0:
                    ndcg_scores.append(ndcg_k)
            except ValueError:
                continue
        
        if not ndcg_scores:
            return 0.0, (0.0, 0.0), 0
        
        mean_ndcg = np.mean(ndcg_scores)
        std_ndcg = np.std(ndcg_scores)
        n_groups = len(ndcg_scores)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n_groups - 1) if n_groups > 1 else 1.96
        margin_error = t_critical * (std_ndcg / np.sqrt(n_groups))
        
        ci_lower = max(0.0, mean_ndcg - margin_error)
        ci_upper = min(1.0, mean_ndcg + margin_error)
        
        return mean_ndcg, (ci_lower, ci_upper), n_groups
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, groups):
        """Calculate comprehensive ranking metrics"""
        metrics = {}
        unique_groups = np.unique(groups)
        
        ndcg_scores = {k: [] for k in [1, 3, 5, 10]}
        hit_rates = {k: [] for k in [1, 3, 5, 10]}
        map_scores = []
        mrr_scores = []
        
        for group in unique_groups:
            group_mask = groups == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            if group_true.sum() == 0:
                continue
            
            # NDCG@K
            for k in [1, 3, 5, 10]:
                if len(group_true) >= k:
                    ndcg_k = ndcg_score([group_true], [group_pred], k=k)
                    ndcg_scores[k].append(ndcg_k)
            
            # Hit Rate@K
            sorted_indices = np.argsort(group_pred)[::-1]
            for k in [1, 3, 5, 10]:
                if k <= len(group_true):
                    top_k_indices = sorted_indices[:k]
                    hit_rates[k].append(1.0 if group_true[top_k_indices].sum() > 0 else 0.0)
            
            # MAP
            try:
                map_score = average_precision_score(group_true, group_pred)
                map_scores.append(map_score)
            except:
                pass
            
            # MRR
            first_relevant = np.where(group_true[sorted_indices] == 1)[0]
            if len(first_relevant) > 0:
                mrr_scores.append(1.0 / (first_relevant[0] + 1))
            else:
                mrr_scores.append(0.0)
        
        # Aggregate metrics
        for k in [1, 3, 5, 10]:
            if ndcg_scores[k]:
                metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[k])
            if hit_rates[k]:
                metrics[f'HitRate@{k}'] = np.mean(hit_rates[k])
        
        if map_scores:
            metrics['MAP'] = np.mean(map_scores)
        if mrr_scores:
            metrics['MRR'] = np.mean(mrr_scores)
        
        return metrics
    
    def perform_statistical_analysis(self, cv_summary, test_results, baseline_ndcg5=0.5030, verbose=True):
        """Perform comprehensive statistical analysis"""
        if verbose:
            print("📊 STATISTICAL ANALYSIS")
            print("="*25)
        
        analysis_results = {}
        
        for model_name in ['lightgbm', 'xgboost']:
            model_key = f'{model_name}_optimized'
            
            if model_name in cv_summary and model_key in test_results:
                cv_mean = cv_summary[model_name]['mean']
                cv_std = cv_summary[model_name]['std']
                test_score = test_results[model_key]['NDCG@5']
                
                # Calculate improvements
                improvement_vs_baseline = ((test_score - baseline_ndcg5) / baseline_ndcg5) * 100
                
                # Calculate CV→Test gap
                cv_test_gap = ((cv_mean - test_score) / cv_mean * 100) if cv_mean > 0 else 0
                
                # Statistical significance test (approximate)
                if cv_std > 0:
                    z_score = (cv_mean - test_score) / cv_std
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    z_score = 0
                    p_value = 1.0
                
                # Confidence intervals
                ci_lower = test_results[model_key].get('NDCG@5_CI_lower', test_score)
                ci_upper = test_results[model_key].get('NDCG@5_CI_upper', test_score)
                ci_width = ci_upper - ci_lower
                
                analysis_results[model_name] = {
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'test_score': test_score,
                    'improvement_vs_baseline': improvement_vs_baseline,
                    'cv_test_gap': cv_test_gap,
                    'z_score': z_score,
                    'p_value': p_value,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_width
                }
                
                if verbose:
                    print(f"\n{model_name.upper()} ANALYSIS:")
                    print(f"  CV Score:       {cv_mean:.4f} ± {cv_std:.4f}")
                    print(f"  Test Score:     {test_score:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
                    print(f"  vs Baseline:    {improvement_vs_baseline:+.2f}%")
                    print(f"  CV→Test Gap:    {abs(cv_test_gap):.1f}%")
                    print(f"  Statistical Sig: {'YES' if p_value < 0.05 else 'NO'} (p={p_value:.3f})")
                    
                    # Quality assessment
                    if abs(cv_test_gap) < 3:
                        quality = "🌟 EXCELLENT"
                    elif abs(cv_test_gap) < 6:
                        quality = "✅ GOOD"
                    elif abs(cv_test_gap) < 10:
                        quality = "⚠️ ACCEPTABLE"
                    else:
                        quality = "❌ POOR"
                    
                    print(f"  Validation Quality: {quality}")
        
        return analysis_results
    
    def generate_final_assessment(self, analysis_results, baseline_ndcg5=0.5030):
        """Generate final assessment and recommendations"""
        print("\n🎯 FINAL ASSESSMENT")
        print("="*20)
        
        best_model = ""
        best_score = 0
        best_improvement = 0
        
        for model_name, analysis in analysis_results.items():
            test_score = analysis['test_score']
            improvement = analysis['improvement_vs_baseline']
            
            if test_score > best_score:
                best_score = test_score
                best_model = model_name
                best_improvement = improvement
        
        print(f"Best Model: {best_model.upper()}")
        print(f"Best Test NDCG@5: {best_score:.4f}")
        print(f"Baseline NDCG@5: {baseline_ndcg5:.4f}")
        print(f"Total Improvement: {best_improvement:+.2f}%")
        
        # Overall quality assessment
        best_analysis = analysis_results[best_model]
        cv_gap = abs(best_analysis['cv_test_gap'])
        cv_stability = best_analysis['cv_std']
        
        print(f"\nModel Quality Assessment:")
        print(f"  CV Stability: {'HIGH' if cv_stability < 0.02 else 'MEDIUM' if cv_stability < 0.05 else 'LOW'} (σ={cv_stability:.4f})")
        print(f"  CV→Test Gap: {cv_gap:.1f}% ({'EXCELLENT' if cv_gap < 3 else 'GOOD' if cv_gap < 6 else 'ACCEPTABLE' if cv_gap < 10 else 'POOR'})")
        
        # Recommendations
        print(f"\nRecommendations:")
        if best_improvement > 10:
            print("🌟 EXCELLENT: Outstanding improvement achieved!")
            print("📈 Proceed to ensemble methods with high confidence")
            print("🎯 Hackathon readiness: VERY HIGH")
        elif best_improvement > 5:
            print("✅ GOOD: Solid improvement over baseline")
            print("📊 Consider ensemble methods for additional gains")
            print("🎯 Hackathon readiness: HIGH")
        elif best_improvement > 2:
            print("📈 MODEST: Reasonable improvement achieved")
            print("⚖️ Ensemble may provide additional benefits")
            print("🎯 Hackathon readiness: MEDIUM")
        else:
            print("⚠️ LIMITED: Minimal improvement over baseline")
            print("🔧 Consider alternative approaches")
            print("🎯 Hackathon readiness: LOW")
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'best_improvement': best_improvement,
            'recommendation': 'PROCEED' if best_improvement > 2 else 'REVIEW'
        }

def create_comprehensive_visualizations(optimizer, cv_summary, test_results, analysis_results, training_histories, baseline_ndcg5=0.5030):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Comparison
    ax1 = plt.subplot(3, 4, 1)
    
    models = ['Baseline']
    scores = [baseline_ndcg5]
    colors = ['lightcoral']
    
    for model_name in ['lightgbm_optimized', 'xgboost_optimized']:
        if model_name in test_results:
            models.append(model_name.replace('_optimized', '').upper())
            scores.append(test_results[model_name]['NDCG@5'])
            colors.append('lightgreen')
    
    bars = ax1.bar(models, scores, color=colors, alpha=0.8)
    ax1.set_ylabel('NDCG@5')
    ax1.set_title('Performance vs Baseline', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. CV vs Test Performance
    ax2 = plt.subplot(3, 4, 2)
    
    cv_scores = []
    test_scores = []
    model_names = []
    
    for model_name in ['lightgbm', 'xgboost']:
        if model_name in cv_summary:
            cv_scores.append(cv_summary[model_name]['mean'])
            test_scores.append(test_results[f'{model_name}_optimized']['NDCG@5'])
            model_names.append(model_name.upper())
    
    scatter = ax2.scatter(cv_scores, test_scores, s=100, alpha=0.8, c=['blue', 'red'])
    
    # Add diagonal line
    min_score = min(min(cv_scores), min(test_scores)) - 0.01
    max_score = max(max(cv_scores), max(test_scores)) + 0.01
    ax2.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5, label='Perfect CV=Test')
    
    ax2.set_xlabel('CV NDCG@5')
    ax2.set_ylabel('Test NDCG@5')
    ax2.set_title('CV vs Test Performance', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for i, name in enumerate(model_names):
        ax2.annotate(name, (cv_scores[i], test_scores[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    # 3. LightGBM Learning Curves
    ax3 = plt.subplot(3, 4, 3)
    
    if 'lightgbm' in training_histories:
        history = training_histories['lightgbm']
        if 'train' in history and 'val' in history:
            train_scores = history['train']['ndcg@5']
            val_scores = history['val']['ndcg@5']
            
            iterations = range(len(train_scores))
            ax3.plot(iterations, train_scores, label='Train', color='blue', alpha=0.8)
            ax3.plot(iterations, val_scores, label='Validation', color='red', alpha=0.8)
            
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('NDCG@5')
            ax3.set_title('LightGBM Learning Curves', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # 4. XGBoost Learning Curves
    ax4 = plt.subplot(3, 4, 4)
    
    if 'xgboost' in training_histories:
        history = training_histories['xgboost']
        if 'train' in history and 'val' in history:
            train_scores = history['train']['ndcg@5']
            val_scores = history['val']['ndcg@5']
            
            iterations = range(len(train_scores))
            ax4.plot(iterations, train_scores, label='Train', color='blue', alpha=0.8)
            ax4.plot(iterations, val_scores, label='Validation', color='red', alpha=0.8)
            
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('NDCG@5')
            ax4.set_title('XGBoost Learning Curves', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    # 5. Comprehensive Metrics
    ax5 = plt.subplot(3, 4, 5)
    
    best_model = max(test_results.keys(), key=lambda k: test_results[k]['NDCG@5'])
    metrics_to_show = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'MAP', 'MRR', 'HitRate@1', 'HitRate@3', 'HitRate@5']
    
    metric_values = []
    metric_labels = []
    
    for metric in metrics_to_show:
        if metric in test_results[best_model]:
            metric_values.append(test_results[best_model][metric])
            metric_labels.append(metric)
    
    bars = ax5.bar(range(len(metric_labels)), metric_values, color='lightblue', alpha=0.8)
    ax5.set_xticks(range(len(metric_labels)))
    ax5.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax5.set_ylabel('Score')
    ax5.set_title(f'Best Model Metrics\n({best_model.replace("_optimized", "").upper()})', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Improvement Breakdown
    ax6 = plt.subplot(3, 4, 6)
    
    improvements = []
    improvement_labels = []
    
    for model_name in ['lightgbm', 'xgboost']:
        if model_name in analysis_results:
            improvement = analysis_results[model_name]['improvement_vs_baseline']
            improvements.append(improvement)
            improvement_labels.append(model_name.upper())
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax6.bar(improvement_labels, improvements, color=colors, alpha=0.8)
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('Improvement vs Baseline', fontweight='bold')
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.grid(True, alpha=0.3)
    
    for bar, improvement in zip(bars, improvements):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.2 if improvement > 0 else -0.5),
                 f'{improvement:+.1f}%', ha='center', va='bottom' if improvement > 0 else 'top',
                 fontweight='bold')
    
    # 7. CV Stability Analysis
    ax7 = plt.subplot(3, 4, 7)
    
    stability_data = []
    stability_labels = []
    
    for model_name in ['lightgbm', 'xgboost']:
        if model_name in cv_summary:
            cv_std = cv_summary[model_name]['std']
            stability_data.append(cv_std)
            stability_labels.append(model_name.upper())
    
    colors = ['green' if std < 0.02 else 'orange' if std < 0.05 else 'red' for std in stability_data]
    bars = ax7.bar(stability_labels, stability_data, color=colors, alpha=0.8)
    
    ax7.set_ylabel('CV Standard Deviation')
    ax7.set_title('Model Stability\n(Lower = Better)', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Add stability thresholds
    ax7.axhline(y=0.02, color='green', linestyle='--', alpha=0.7, label='High Stability')
    ax7.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Medium Stability')
    ax7.legend()
    
    for bar, std in zip(bars, stability_data):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{std:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Confidence Intervals
    ax8 = plt.subplot(3, 4, 8)
    
    ci_data = []
    ci_labels = []
    ci_lowers = []
    ci_uppers = []
    
    for model_name in ['lightgbm_optimized', 'xgboost_optimized']:
        if model_name in test_results:
            score = test_results[model_name]['NDCG@5']
            ci_lower = test_results[model_name].get('NDCG@5_CI_lower', score)
            ci_upper = test_results[model_name].get('NDCG@5_CI_upper', score)
            
            ci_data.append(score)
            ci_lowers.append(score - ci_lower)
            ci_uppers.append(ci_upper - score)
            ci_labels.append(model_name.replace('_optimized', '').upper())
    
    bars = ax8.bar(ci_labels, ci_data, yerr=[ci_lowers, ci_uppers], 
                   capsize=10, color='lightblue', alpha=0.8)
    ax8.set_ylabel('NDCG@5')
    ax8.set_title('Performance with\n95% Confidence Intervals', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, ci_data):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 9. Parameter Comparison
    ax9 = plt.subplot(3, 4, 9)
    
    # Show key parameter differences
    lgb_params = optimizer.optimal_params['lightgbm']
    xgb_params = optimizer.optimal_params['xgboost']
    
    param_comparison = [
        ['Learning Rate', f"{lgb_params['learning_rate']:.4f}", f"{xgb_params['eta']:.4f}"],
        ['Regularization L1', f"{lgb_params['lambda_l1']:.3f}", f"{xgb_params['alpha']:.3f}"],
        ['Regularization L2', f"{lgb_params['lambda_l2']:.3f}", f"{xgb_params['lambda']:.3f}"],
        ['Max Depth', f"{lgb_params['max_depth']}", f"{xgb_params['max_depth']}"],
        ['Subsample', f"{lgb_params['bagging_fraction']:.3f}", f"{xgb_params['subsample']:.3f}"],
        ['Feature Fraction', f"{lgb_params['feature_fraction']:.3f}", f"{xgb_params['colsample_bytree']:.3f}"]
    ]
    
    table = ax9.table(cellText=param_comparison,
                      colLabels=['Parameter', 'LightGBM', 'XGBoost'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(param_comparison) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    ax9.set_title('Optimized Parameters', fontweight='bold')
    ax9.axis('off')
    
    # 10. Statistical Test Results
    ax10 = plt.subplot(3, 4, 10)
    
    stat_data = []
    
    for model_name in ['lightgbm', 'xgboost']:
        if model_name in analysis_results:
            analysis = analysis_results[model_name]
            stat_data.append([
                model_name.upper(),
                f"{analysis['test_score']:.4f}",
                f"{abs(analysis['cv_test_gap']):.1f}%",
                f"{analysis['p_value']:.3f}",
                "✅" if analysis['p_value'] > 0.05 else "⚠️"
            ])
    
    table = ax10.table(cellText=stat_data,
                       colLabels=['Model', 'Test Score', 'CV Gap', 'P-Value', 'Valid'],
                       cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(stat_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    ax10.set_title('Statistical Validation', fontweight='bold')
    ax10.axis('off')
    
    # 11. Cross-Validation Scores Distribution
    ax11 = plt.subplot(3, 4, 11)
    
    for i, model_name in enumerate(['lightgbm', 'xgboost']):
        if model_name in cv_summary:
            scores = cv_summary[model_name]['scores']
            x = np.arange(1, len(scores) + 1) + i * 0.3
            ax11.bar(x, scores, width=0.3, label=model_name.upper(), alpha=0.8)
    
    ax11.set_xlabel('CV Fold')
    ax11.set_ylabel('NDCG@5')
    ax11.set_title('Cross-Validation Scores', fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Final Summary Dashboard
    ax12 = plt.subplot(3, 4, 12)
    
    # Get best model info
    best_model = max(test_results.keys(), key=lambda k: test_results[k]['NDCG@5'])
    best_score = test_results[best_model]['NDCG@5']
    best_improvement = ((best_score - baseline_ndcg5) / baseline_ndcg5) * 100
    
    best_model_name = best_model.replace('_optimized', '').upper()
    
    # Create summary
    ax12.text(0.5, 0.9, 'OPTIMIZED MODELS SUMMARY', ha='center', va='center', 
              fontsize=14, fontweight='bold', transform=ax12.transAxes)
    
    ax12.text(0.5, 0.75, f'Best Model: {best_model_name}', ha='center', va='center', 
              fontsize=12, transform=ax12.transAxes)
    
    ax12.text(0.5, 0.65, f'Test NDCG@5: {best_score:.4f}', ha='center', va='center', 
              fontsize=11, transform=ax12.transAxes)
    
    ax12.text(0.5, 0.55, f'Baseline: {baseline_ndcg5:.4f}', ha='center', va='center', 
              fontsize=11, transform=ax12.transAxes)
    
    ax12.text(0.5, 0.45, f'Improvement: {best_improvement:+.1f}%', ha='center', va='center', 
              fontsize=11, fontweight='bold', transform=ax12.transAxes)
    
    # Status indicator
    if best_improvement > 10:
        status = "🌟 EXCELLENT"
        status_color = 'green'
    elif best_improvement > 5:
        status = "✅ VERY GOOD"
        status_color = 'lightgreen'
    elif best_improvement > 2:
        status = "📈 GOOD"
        status_color = 'orange'
    else:
        status = "⚠️ LIMITED"
        status_color = 'lightcoral'
    
    ax12.text(0.5, 0.3, status, ha='center', va='center', fontsize=16, fontweight='bold',
              transform=ax12.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.3))
    
    # Ready for next step?
    ready = "🚀 READY FOR MODULE 8" if best_improvement > 2 else "🔧 NEEDS REVIEW"
    ax12.text(0.5, 0.15, ready, ha='center', va='center', fontsize=11, fontweight='bold',
              transform=ax12.transAxes)
    
    ax12.set_title('Final Assessment', fontweight='bold')
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig('tim_hackathon_module7_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

def main_module7_optimized():
    """Execute Module 7: Optimized Models with Pre-tuned Parameters"""
    print("="*80)
    print("TIM HACKATHON - MODULE 7: OPTIMIZED MODELS (PRE-TUNED PARAMETERS)")
    print("="*80)
    
    # Load enhanced data from Module 6
    try:
        print(f"Loading enhanced data from Module 6...")
        print(f"  Train enhanced: {train_enhanced.shape}")
        print(f"  Test enhanced: {test_enhanced.shape}")
        print(f"  Module 6 baseline: NDCG@5 = 0.5030")
        print(f"  Using pre-optimized parameters from tuning runs")
    except:
        print("❌ Error: Please run Module 6 first!")
        return
    
    # Initialize optimizer with pre-tuned parameters
    optimizer = TIMOptimizedModels(random_state=RANDOM_STATE)
    
    print(f"\n📋 PRE-TUNED PARAMETERS")
    print("="*24)
    print("LightGBM (Trial 45 - CV NDCG@5: 0.7638 ± 0.0028):")
    for key, value in optimizer.optimal_params['lightgbm'].items():
        if key not in ['objective', 'metric', 'ndcg_eval_at', 'boosting_type', 'verbose', 'random_state', 'force_row_wise']:
            print(f"  {key}: {value}")
    
    print("\nXGBoost (Trial 3 - CV NDCG@5: 0.7673):")
    for key, value in optimizer.optimal_params['xgboost'].items():
        if key not in ['objective', 'eval_metric', 'random_state', 'verbosity']:
            print(f"  {key}: {value}")
    
    # Step 1: Prepare ranking data
    print(f"\n📊 PREPARING RANKING DATA")
    print("="*26)
    X_train, y_train, groups_train, group_sizes_train, group_to_customer, feature_columns = optimizer.prepare_ranking_data(train_enhanced, verbose=True)
    
    # Step 2: Cross-validation with optimized parameters
    print(f"\n📊 CROSS-VALIDATION WITH OPTIMIZED PARAMETERS")
    print("="*47)
    cv_results, cv_summary = optimizer.perform_cross_validation(X_train, y_train, group_sizes_train, group_to_customer)
    
    # Step 3: Train final optimized models
    print(f"\n🚀 TRAINING OPTIMIZED MODELS")
    print("="*29)
    final_models, training_histories = optimizer.train_optimized_models(X_train, y_train, group_sizes_train)
    
    # Step 4: Comprehensive evaluation
    print(f"\n📊 COMPREHENSIVE EVALUATION")
    print("="*29)
    test_results = optimizer.evaluate_models_comprehensive(final_models, test_enhanced)
    
    # Step 5: Statistical analysis
    print(f"\n📈 STATISTICAL ANALYSIS")
    print("="*25)
    analysis_results = optimizer.perform_statistical_analysis(cv_summary, test_results, baseline_ndcg5=0.5030)
    
    # Step 6: Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("="*20)
    final_assessment = optimizer.generate_final_assessment(analysis_results, baseline_ndcg5=0.5030)
    
    # Step 7: Create comprehensive visualizations
    print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*40)
    create_comprehensive_visualizations(optimizer, cv_summary, test_results, analysis_results, training_histories)
    
    print(f"\n✅ MODULE 7 OPTIMIZED COMPLETED")
    print("="*32)
    print("Generated files:")
    print("  - tim_hackathon_module7_optimized.png")
    print("Key results:")
    print(f"  Best model: {final_assessment['best_model'].upper()}")
    print(f"  Best score: {final_assessment['best_score']:.4f}")
    print(f"  Improvement: {final_assessment['best_improvement']:+.2f}%")
    print(f"  Recommendation: {final_assessment['recommendation']}")
    
    if final_assessment['recommendation'] == 'PROCEED':
        print(f"\n🚀 READY FOR MODULE 8: ENSEMBLE METHODS!")
        print(f"   Target ensemble boost: +1-3% additional improvement")
    
    return optimizer, final_models, test_results, analysis_results, cv_summary, training_histories

# Execute Module 7 Optimized
if __name__ == "__main__":
    optimizer, final_models, test_results, analysis_results, cv_summary, training_histories = main_module7_optimized()
