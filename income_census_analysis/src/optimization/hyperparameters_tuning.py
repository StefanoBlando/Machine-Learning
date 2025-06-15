"""
Hyperparameter optimization module.
Extracted from Module 6 of the original notebook.
"""
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional
from sklearn.model_selection import cross_val_score

# Optimization imports (with availability checks)
try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    from sklearn.model_selection import RandomizedSearchCV

from ..config.settings import RANDOM_STATE, SMART_ITERATIONS
from ..evaluation.metrics import calculate_comprehensive_metrics, business_metrics_analysis
from ..models.traditional import get_focused_traditional_params
from ..models.advanced import get_focused_advanced_params

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_smart_iterations_per_model(model_name: str) -> int:
    """
    Get smart iteration count for hyperparameter optimization.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        int: Number of iterations for optimization
    """
    return SMART_ITERATIONS.get(model_name, 10)


def get_focused_param_space(model_name: str, model_config: Dict, use_focused: bool = True) -> Dict:
    """
    Get parameter space for hyperparameter optimization.
    
    Args:
        model_name (str): Name of the model
        model_config (Dict): Model configuration
        use_focused (bool): Whether to use focused parameter space
        
    Returns:
        Dict: Parameter space for optimization
    """
    if not use_focused:
        return model_config['param_space']
    
    # Check model category and get appropriate focused params
    if model_config.get('category') == 'traditional':
        return get_focused_traditional_params(model_name, model_config)
    elif model_config.get('category') == 'advanced':
        return get_focused_advanced_params(model_name, model_config)
    else:
        return model_config['param_space']


def perform_hyperparameter_optimization(models_config: Dict, balanced_datasets: Dict,
                                       cv_strategies: Dict, custom_scorer,
                                       X_val_preprocessed: np.ndarray, y_val: pd.Series,
                                       sampling_strategy_key: str,
                                       use_focused_params: bool = True) -> List[Dict]:
    """
    Perform hyperparameter optimization for all models on a specific sampling strategy.
    
    Args:
        models_config (Dict): Configuration of all models
        balanced_datasets (Dict): Different balanced datasets
        cv_strategies (Dict): Cross-validation strategies
        custom_scorer: Custom scoring function
        X_val_preprocessed (np.ndarray): Preprocessed validation features
        y_val (pd.Series): Validation targets
        sampling_strategy_key (str): Key for the sampling strategy to use
        use_focused_params (bool): Whether to use focused parameter spaces
        
    Returns:
        List[Dict]: Results for all models with this sampling strategy
    """
    logger.info(f"🚀 Starting Hyperparameter Optimization for {sampling_strategy_key.upper()} Sampling Strategy")
    logger.info("-" * 70)
    
    # Get training data for this sampling strategy
    data_for_training = balanced_datasets[sampling_strategy_key]
    X_train_data = data_for_training['X_train']
    y_train_data = data_for_training['y_train']
    cv_strategy = cv_strategies['stratified_5fold']
    
    logger.info(f"📊 Dataset size: {len(y_train_data):,} samples, {y_train_data.mean()*100:.1f}% positive class")
    
    results_for_strategy = []
    SearchCV_Class = BayesSearchCV if SKOPT_AVAILABLE else RandomizedSearchCV
    optimization_method = "Bayesian" if SKOPT_AVAILABLE else "RandomizedSearch"
    
    logger.info(f"🔧 Using {optimization_method} optimization")
    
    for model_name, model_config in models_config.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"🤖 TRAINING: {model_name} with {sampling_strategy_key.upper()} Data")
        logger.info(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            # Get optimization parameters
            n_iter = get_smart_iterations_per_model(model_name)
            param_space = get_focused_param_space(model_name, model_config, use_focused_params)
            
            logger.info(f"🎯 Using {n_iter} iterations with {'focused' if use_focused_params else 'full'} parameter space")
            
            # Create fresh model instance
            base_model_instance = model_config['model'].__class__(**model_config['model'].get_params())
            
            # Apply class weights or balancing based on model type
            base_model_instance = _apply_model_balancing(base_model_instance, model_name, y_train_data)
            
            # Configure search parameters
            search_params = {
                'estimator': base_model_instance,
                'scoring': custom_scorer,
                'cv': cv_strategy,
                'n_jobs': -1,
                'random_state': RANDOM_STATE,
            }
            
            # Add algorithm-specific parameters
            if SKOPT_AVAILABLE:
                search_params['search_spaces'] = param_space
                search_params['n_iter'] = n_iter
            else:
                search_params['param_distributions'] = param_space
                search_params['n_iter'] = n_iter
                search_params['return_train_score'] = True
            
            # Perform optimization
            search = SearchCV_Class(**search_params)
            search.fit(X_train_data, y_train_data)
            
            training_duration = time.time() - start_time
            
            # Get best model and make predictions
            best_model = search.best_estimator_
            y_val_pred_default = best_model.predict(X_val_preprocessed)
            y_val_proba = None
            
            if hasattr(best_model, 'predict_proba'):
                y_val_proba = best_model.predict_proba(X_val_preprocessed)[:, 1]
            
            # Calculate business metrics and find optimal threshold
            business_metrics_result = business_metrics_analysis(y_val, y_val_pred_default, y_val_proba)
            optimal_threshold = business_metrics_result.get('optimal_threshold', 0.5)
            
            # Calculate predictions at optimal threshold
            if y_val_proba is not None:
                y_val_pred_optimal = (y_val_proba >= optimal_threshold).astype(int)
            else:
                y_val_pred_optimal = y_val_pred_default
            
            # Calculate comprehensive metrics
            validation_metrics_default = calculate_comprehensive_metrics(y_val, y_val_pred_default, y_val_proba)
            validation_metrics_optimal = calculate_comprehensive_metrics(y_val, y_val_pred_optimal, y_val_proba)
            
            # Cross-validation score verification
            cv_score_mean = search.best_score_
            cv_scores_recheck = cross_val_score(
                best_model, X_train_data, y_train_data,
                cv=cv_strategy, scoring=custom_scorer, n_jobs=-1
            )
            cv_score_std = cv_scores_recheck.std()
            
            # Compile results
            result = {
                'model_name': model_name,
                'sampling_strategy': sampling_strategy_key,
                'best_params': search.best_params_,
                'cv_score_mean': cv_score_mean,
                'cv_score_std': cv_score_std,
                'validation_metrics_default_threshold': validation_metrics_default,
                'validation_metrics_optimal_threshold': validation_metrics_optimal,
                'business_metrics': business_metrics_result,
                'training_time': training_duration,
                'best_model': best_model,
                'search_object': search,
                'training_samples': len(y_train_data),
                'y_val_pred_default': y_val_pred_default,
                'y_val_pred_optimal': y_val_pred_optimal,
                'y_val_proba': y_val_proba,
                'n_iterations_used': n_iter,
                'optimization_method': optimization_method
            }
            
            results_for_strategy.append(result)
            
            # Log results
            logger.info(f"✅ Training completed in {training_duration:.1f}s")
            logger.info(f"📊 CV Score (Custom Scorer): {cv_score_mean:.4f} ± {cv_score_std:.4f}")
            logger.info(f"🎯 Validation Metrics (Optimal Threshold={optimal_threshold:.3f}):")
            logger.info(f"   F1-Score: {validation_metrics_optimal['f1']:.4f}")
            logger.info(f"   ROC-AUC: {validation_metrics_optimal['roc_auc']:.4f}")
            logger.info(f"   Precision: {validation_metrics_optimal['precision']:.4f}")
            logger.info(f"   Recall: {validation_metrics_optimal['recall']:.4f}")
            
            if not np.isnan(business_metrics_result.get('max_profit', np.nan)):
                logger.info(f"💰 Max profit (at optimal threshold): {business_metrics_result['max_profit']:,}")
            
            # Log best parameters (truncated if too long)
            params_str = str(search.best_params_)
            if len(params_str) > 150:
                params_str = params_str[:150] + '...'
            logger.info(f"⚙️ Best params: {params_str}")
            
        except Exception as e:
            logger.error(f"❌ Training failed for {model_name}: {str(e)}")
            
            # Create failure result
            failure_result = {
                'model_name': model_name,
                'sampling_strategy': sampling_strategy_key,
                'best_params': {},
                'cv_score_mean': np.nan,
                'cv_score_std': np.nan,
                'validation_metrics_default_threshold': {},
                'validation_metrics_optimal_threshold': {},
                'business_metrics': {
                    'current_cost': np.nan,
                    'current_profit': np.nan,
                    'optimal_threshold': np.nan,
                    'max_profit': np.nan
                },
                'training_time': 0,
                'best_model': None,
                'search_object': None,
                'training_samples': 0,
                'y_val_pred_default': None,
                'y_val_pred_optimal': None,
                'y_val_proba': None,
                'n_iterations_used': 0,
                'optimization_method': optimization_method,
                'error': str(e)
            }
            results_for_strategy.append(failure_result)
    
    logger.info(f"\n✅ Hyperparameter optimization completed for {sampling_strategy_key}")
    logger.info(f"📊 Successfully trained: {sum(1 for r in results_for_strategy if r['best_model'] is not None)}/{len(models_config)} models")
    
    return results_for_strategy


def _apply_model_balancing(model_instance, model_name: str, y_train_data: pd.Series):
    """
    Apply appropriate class balancing strategy based on model type.
    
    Args:
        model_instance: Model instance to configure
        model_name (str): Name of the model
        y_train_data (pd.Series): Training target data
        
    Returns:
        Configured model instance
    """
    try:
        if model_name == 'XGBoost':
            # XGBoost uses scale_pos_weight
            scale_pos_weight_current = (y_train_data == 0).sum() / (y_train_data == 1).sum()
            model_instance.set_params(scale_pos_weight=scale_pos_weight_current)
            logger.info(f"   🔧 Applied scale_pos_weight: {scale_pos_weight_current:.3f}")
            
        elif model_name == 'LightGBM':
            # LightGBM uses class_weight='balanced'
            model_instance.set_params(class_weight='balanced')
            logger.info(f"   🔧 Applied class_weight='balanced'")
            
        else:
            # Traditional sklearn models use class_weight dict
            if 'class_weight' in model_instance.get_params():
                from sklearn.utils.class_weight import compute_class_weight
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train_data), y=y_train_data)
                class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
                model_instance.set_params(class_weight=class_weight_dict)
                logger.info(f"   🔧 Applied class_weight: {{0: {class_weight_dict[0]:.3f}, 1: {class_weight_dict[1]:.3f}}}")
            else:
                logger.warning(f"   ⚠️ Model {model_name} does not support class_weight parameter")
                
    except Exception as e:
        logger.warning(f"   ⚠️ Could not apply balancing for {model_name}: {e}")
    
    return model_instance


def run_optimization_for_all_strategies(models_config: Dict, balanced_datasets: Dict,
                                       cv_strategies: Dict, custom_scorer,
                                       X_val_preprocessed: np.ndarray, y_val: pd.Series,
                                       use_focused_params: bool = True) -> List[Dict]:
    """
    Run hyperparameter optimization for all models across all sampling strategies.
    
    Args:
        models_config (Dict): Configuration of all models
        balanced_datasets (Dict): Different balanced datasets
        cv_strategies (Dict): Cross-validation strategies
        custom_scorer: Custom scoring function
        X_val_preprocessed (np.ndarray): Preprocessed validation features
        y_val (pd.Series): Validation targets
        use_focused_params (bool): Whether to use focused parameter spaces
        
    Returns:
        List[Dict]: Results for all models across all strategies
    """
    logger.info("🚀 Starting hyperparameter optimization for ALL models and ALL sampling strategies")
    logger.info("=" * 80)
    
    all_results = []
    
    for strategy_key in balanced_datasets.keys():
        logger.info(f"\n🎯 Processing sampling strategy: {strategy_key.upper()}")
        
        strategy_results = perform_hyperparameter_optimization(
            models_config=models_config,
            balanced_datasets=balanced_datasets,
            cv_strategies=cv_strategies,
            custom_scorer=custom_scorer,
            X_val_preprocessed=X_val_preprocessed,
            y_val=y_val,
            sampling_strategy_key=strategy_key,
            use_focused_params=use_focused_params
        )
        
        all_results.extend(strategy_results)
    
    logger.info(f"\n🎉 OPTIMIZATION COMPLETED!")
    logger.info(f"📊 Total model configurations trained: {len(all_results)}")
    logger.info(f"✅ Successful trainings: {sum(1 for r in all_results if r['best_model'] is not None)}")
    logger.info(f"❌ Failed trainings: {sum(1 for r in all_results if r['best_model'] is None)}")
    
    return all_results


def get_optimization_summary(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Get a summary of the optimization process.
    
    Args:
        all_results (List[Dict]): Results from all model training
        
    Returns:
        Dict[str, Any]: Optimization summary
    """
    successful_results = [r for r in all_results if r['best_model'] is not None]
    failed_results = [r for r in all_results if r['best_model'] is None]
    
    if not successful_results:
        return {
            'total_configurations': len(all_results),
            'successful_trainings': 0,
            'failed_trainings': len(failed_results),
            'error': 'No successful model training'
        }
    
    # Calculate statistics
    f1_scores = [r['validation_metrics_optimal_threshold']['f1'] for r in successful_results 
                 if not np.isnan(r['validation_metrics_optimal_threshold'].get('f1', np.nan))]
    
    training_times = [r['training_time'] for r in successful_results if r['training_time'] > 0]
    
    # Find best model
    best_result = max(successful_results, 
                     key=lambda x: x['validation_metrics_optimal_threshold'].get('f1', 0))
    
    # Group by sampling strategy
    strategy_performance = {}
    for result in successful_results:
        strategy = result['sampling_strategy']
        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        strategy_performance[strategy].append(result['validation_metrics_optimal_threshold'].get('f1', 0))
    
    # Calculate strategy averages
    strategy_averages = {
        strategy: np.mean(scores) for strategy, scores in strategy_performance.items()
    }
    
    return {
        'total_configurations': len(all_results),
        'successful_trainings': len(successful_results),
        'failed_trainings': len(failed_results),
        'optimization_method': successful_results[0]['optimization_method'] if successful_results else 'Unknown',
        'best_model': {
            'name': best_result['model_name'],
            'sampling_strategy': best_result['sampling_strategy'],
            'f1_score': best_result['validation_metrics_optimal_threshold']['f1'],
            'training_time': best_result['training_time']
        },
        'performance_statistics': {
            'f1_scores': {
                'mean': np.mean(f1_scores) if f1_scores else 0,
                'std': np.std(f1_scores) if f1_scores else 0,
                'min': np.min(f1_scores) if f1_scores else 0,
                'max': np.max(f1_scores) if f1_scores else 0
            },
            'training_times': {
                'mean': np.mean(training_times) if training_times else 0,
                'total': np.sum(training_times) if training_times else 0,
                'min': np.min(training_times) if training_times else 0,
                'max': np.max(training_times) if training_times else 0
            }
        },
        'strategy_performance': strategy_averages,
        'best_strategy': max(strategy_averages.items(), key=lambda x: x[1])[0] if strategy_averages else None,
        'models_per_strategy': {
            strategy: len([r for r in successful_results if r['sampling_strategy'] == strategy])
            for strategy in set(r['sampling_strategy'] for r in successful_results)
        }
    }


def save_optimization_results(all_results: List[Dict], output_path: str) -> None:
    """
    Save optimization results to file.
    
    Args:
        all_results (List[Dict]): Results from all model training
        output_path (str): Path to save results
    """
    import json
    import pickle
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary as JSON
    summary = get_optimization_summary(all_results)
    summary_path = output_path / "optimization_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save detailed results (without model objects) as JSON
    results_for_json = []
    for result in all_results:
        json_result = {k: v for k, v in result.items() 
                      if k not in ['best_model', 'search_object', 'y_val_pred_default', 
                                 'y_val_pred_optimal', 'y_val_proba']}
        results_for_json.append(json_result)
    
    results_path = output_path / "optimization_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=2, default=str)
    
    # Save full results (with model objects) as pickle
    full_results_path = output_path / "optimization_results_full.pkl"
    with open(full_results_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    logger.info(f"📁 Optimization results saved to:")
    logger.info(f"   Summary: {summary_path}")
    logger.info(f"   Results (JSON): {results_path}")
    logger.info(f"   Full results (PKL): {full_results_path}")


def load_optimization_results(results_path: str) -> List[Dict]:
    """
    Load optimization results from file.
    
    Args:
        results_path (str): Path to results file
        
    Returns:
        List[Dict]: Loaded results
    """
    import pickle
    from pathlib import Path
    
    results_path = Path(results_path)
    
    if results_path.suffix == '.pkl':
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    elif results_path.suffix == '.json':
        import json
        with open(results_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")


def create_optimization_report(all_results: List[Dict]) -> str:
    """
    Create a comprehensive optimization report.
    
    Args:
        all_results (List[Dict]): Results from all model training
        
    Returns:
        str: Formatted report
    """
    summary = get_optimization_summary(all_results)
    successful_results = [r for r in all_results if r['best_model'] is not None]
    
    report = []
    report.append("🚀 HYPERPARAMETER OPTIMIZATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overview
    report.append("📊 OPTIMIZATION OVERVIEW")
    report.append("-" * 30)
    report.append(f"Total model configurations: {summary['total_configurations']}")
    report.append(f"Successful trainings: {summary['successful_trainings']}")
    report.append(f"Failed trainings: {summary['failed_trainings']}")
    report.append(f"Optimization method: {summary['optimization_method']}")
    report.append("")
    
    # Best model
    if 'best_model' in summary:
        best = summary['best_model']
        report.append("🏆 BEST MODEL")
        report.append("-" * 15)
        report.append(f"Model: {best['name']}")
        report.append(f"Sampling strategy: {best['sampling_strategy']}")
        report.append(f"F1-Score: {best['f1_score']:.4f}")
        report.append(f"Training time: {best['training_time']:.1f}s")
        report.append("")
    
    # Performance statistics
    if 'performance_statistics' in summary:
        perf = summary['performance_statistics']
        report.append("📈 PERFORMANCE STATISTICS")
        report.append("-" * 25)
        report.append(f"F1-Score range: {perf['f1_scores']['min']:.4f} - {perf['f1_scores']['max']:.4f}")
        report.append(f"F1-Score mean: {perf['f1_scores']['mean']:.4f} ± {perf['f1_scores']['std']:.4f}")
        report.append(f"Total training time: {perf['training_times']['total']:.1f}s ({perf['training_times']['total']/60:.1f} min)")
        report.append(f"Average training time: {perf['training_times']['mean']:.1f}s")
        report.append("")
    
    # Strategy comparison
    if 'strategy_performance' in summary and summary['strategy_performance']:
        report.append("⚖️ SAMPLING STRATEGY COMPARISON")
        report.append("-" * 35)
        for strategy, avg_f1 in sorted(summary['strategy_performance'].items(), 
                                     key=lambda x: x[1], reverse=True):
            report.append(f"{strategy:15s}: {avg_f1:.4f} (avg F1)")
        report.append(f"Best strategy: {summary['best_strategy']}")
        report.append("")
    
    # Top 10 model configurations
    if successful_results:
        report.append("🎯 TOP 10 MODEL CONFIGURATIONS")
        report.append("-" * 35)
        sorted_results = sorted(successful_results, 
                              key=lambda x: x['validation_metrics_optimal_threshold'].get('f1', 0), 
                              reverse=True)
        
        for i, result in enumerate(sorted_results[:10], 1):
            f1 = result['validation_metrics_optimal_threshold'].get('f1', 0)
            model_name = result['model_name']
            strategy = result['sampling_strategy']
            time_taken = result['training_time']
            
            report.append(f"{i:2d}. {model_name:<20} ({strategy:<8}) F1: {f1:.4f}, Time: {time_taken:.1f}s")
        report.append("")
    
    # Model-specific insights
    model_performance = {}
    for result in successful_results:
        model_name = result['model_name']
        f1_score = result['validation_metrics_optimal_threshold'].get('f1', 0)
        
        if model_name not in model_performance:
            model_performance[model_name] = []
        model_performance[model_name].append(f1_score)
    
    if model_performance:
        report.append("🤖 MODEL-SPECIFIC PERFORMANCE")
        report.append("-" * 30)
        for model_name, scores in model_performance.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            best_score = np.max(scores)
            report.append(f"{model_name:<20} Best: {best_score:.4f}, Avg: {mean_score:.4f} ± {std_score:.4f}")
        report.append("")
    
    report.append("✅ Report generated successfully!")
    
    return "\n".join(report)
