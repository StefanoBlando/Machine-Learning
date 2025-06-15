#!/usr/bin/env python3
"""
Complete ML pipeline script for Advanced Income Classification.
Follows the exact structure of the original notebook modules.

Usage:
    python scripts/run_complete_pipeline.py --data-path data/raw/data.csv --output-dir results/
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import *
from src.data.loader import load_and_explore_data, create_eda_visualizations
from src.data.feature_engineering import advanced_feature_engineering, create_feature_engineering_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pipeline_run_{get_timestamp()}.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='Advanced Income Classification Pipeline')
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to the CSV data file')
    parser.add_argument('--output-dir', type=str, default='results/experiments',
                       help='Output directory for results')
    parser.add_argument('--target-col', type=str, default='target',
                       help='Name of the target column')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save visualizations to files')
    parser.add_argument('--run-full-training', action='store_true',
                       help='Run complete model training (time-intensive)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / get_timestamp()
    ensure_dir_exists(output_dir)
    
    logger.info("🚀 ADVANCED INCOME CLASSIFICATION - COMPLETE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"⚙️ Configuration:")
    logger.info(f"   Data Path: {args.data_path}")
    logger.info(f"   Output Directory: {output_dir}")
    logger.info(f"   Target Column: {args.target_col}")
    logger.info(f"   Random State: {RANDOM_STATE}")
    logger.info(f"   Train/Val Split: {int(TRAIN_SIZE*100)}/{int(TEST_SIZE*100)}%")
    logger.info(f"   Cross-Validation: {CV_FOLDS}-fold × {CV_REPEATS} repeats")
    logger.info(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # =============================================================================
        # MODULE 2: DATA LOADING & EDA
        # =============================================================================
        logger.info("\n" + "="*80)
        logger.info("📊 MODULE 2: DATA LOADING & EXPLORATORY DATA ANALYSIS")
        logger.info("="*80)
        
        df, numeric_cols, categorical_cols = load_and_explore_data(
            filepath=args.data_path,
            target_col=args.target_col
        )
        
        if args.save_visualizations:
            eda_viz_path = output_dir / "01_eda_visualizations.png"
            create_eda_visualizations(
                df=df, 
                numeric_cols=numeric_cols, 
                categorical_cols=categorical_cols,
                target_col=args.target_col,
                save_path=str(eda_viz_path)
            )
        
        # Save dataset summary
        dataset_summary = {
            'original_shape': df.shape,
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'target_distribution': df[args.target_col].value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = output_dir / "dataset_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(dataset_summary, f, indent=2)
        
        logger.info(f"✅ Module 2 completed. Dataset summary saved to {summary_path}")
        
        # =============================================================================
        # MODULE 3: FEATURE ENGINEERING
        # =============================================================================
        logger.info("\n" + "="*80)
        logger.info("🔧 MODULE 3: ADVANCED FEATURE ENGINEERING")
        logger.info("="*80)
        
        df_engineered, numeric_cols_updated, categorical_cols_updated, created_features = \
            advanced_feature_engineering(df, args.target_col)
        
        if args.save_visualizations:
            fe_viz_path = output_dir / "02_feature_engineering_visualizations.png"
            create_feature_engineering_visualizations(
                df_original=df,
                df_engineered=df_engineered,
                target_col=args.target_col,
                save_path=str(fe_viz_path)
            )
        
        # Save feature engineering summary
        fe_summary = {
            'original_features': len(numeric_cols) + len(categorical_cols),
            'engineered_features': len(numeric_cols_updated) + len(categorical_cols_updated),
            'new_features_created': len(created_features),
            'new_features_list': created_features,
            'removed_features': COLUMNS_TO_DROP,
            'occupation_mapping_applied': True,
            'timestamp': datetime.now().isoformat()
        }
        
        fe_summary_path = output_dir / "feature_engineering_summary.json"
        with open(fe_summary_path, 'w') as f:
            json.dump(fe_summary, f, indent=2)
        
        # Save engineered dataset
        engineered_data_path = output_dir / "engineered_dataset.csv"
        df_engineered.to_csv(engineered_data_path, index=False)
        
        logger.info(f"✅ Module 3 completed. Engineered dataset saved to {engineered_data_path}")
        logger.info(f"📊 Feature engineering summary saved to {fe_summary_path}")
        
        # =============================================================================
        # MODULE 4: PREPROCESSING & DATA SPLITTING
        # =============================================================================
        logger.info("\n" + "="*80)
        logger.info("⚙️ MODULE 4: PREPROCESSING & DATA SPLITTING")
        logger.info("="*80)
        
        # This would normally import from src.preprocessing.pipeline
        # For now, we'll show the structure
        logger.info("🔄 Preprocessing pipeline would include:")
        logger.info("   • Train/validation split (70/30, stratified)")
        logger.info("   • StandardScaler for numeric features")
        logger.info("   • OneHotEncoder for categorical features")
        logger.info("   • SelectKBest feature selection")
        logger.info("   • Multiple class balancing strategies (SMOTE, ADASYN, SMOTEENN)")
        
        preprocessing_summary = {
            'train_test_split': f"{int(TRAIN_SIZE*100)}/{int(TEST_SIZE*100)}%",
            'random_state': RANDOM_STATE,
            'cv_strategy': f"{CV_FOLDS}-fold × {CV_REPEATS} repeats",
            'sampling_strategies': SAMPLING_STRATEGIES,
            'feature_selection': 'SelectKBest with f_classif',
            'scaling_method': 'StandardScaler',
            'encoding_method': 'OneHotEncoder',
            'timestamp': datetime.now().isoformat()
        }
        
        preprocessing_summary_path = output_dir / "preprocessing_summary.json"
        with open(preprocessing_summary_path, 'w') as f:
            json.dump(preprocessing_summary, f, indent=2)
        
        logger.info(f"✅ Module 4 structure defined. Summary saved to {preprocessing_summary_path}")
        
        # =============================================================================
        # MODEL TRAINING (Conditional)
        # =============================================================================
        if args.run_full_training:
            logger.info("\n" + "="*80)
            logger.info("🤖 MODULES 5-6: MODEL CONFIGURATION & TRAINING")
            logger.info("="*80)
            
            logger.info("🏛️ Traditional Models to be trained:")
            for model in TRADITIONAL_MODELS:
                iterations = SMART_ITERATIONS.get(model, 10)
                logger.info(f"   • {model:<20} ({iterations} iterations)")
            
            logger.info("🚀 Advanced Models to be trained:")
            for model in ADVANCED_MODELS:
                iterations = SMART_ITERATIONS.get(model, 15)
                logger.info(f"   • {model:<20} ({iterations} iterations)")
            
            logger.info("🔗 Ensemble Methods to be created:")
            logger.info("   • Voting Classifier (Soft voting)")
            logger.info("   • Stacking Classifier (Meta-learner)")
            logger.info("   • Weighted Average Ensemble")
            
            training_summary = {
                'traditional_models': TRADITIONAL_MODELS,
                'advanced_models': ADVANCED_MODELS,
                'hyperparameter_optimization': 'BayesSearchCV' if SKOPT_AVAILABLE else 'RandomizedSearchCV',
                'sampling_strategies': SAMPLING_STRATEGIES,
                'optimization_method': 'Bayesian' if SKOPT_AVAILABLE else 'Random',
                'ensemble_methods': ['Voting', 'Stacking', 'Weighted'],
                'interpretability_methods': ['SHAP', 'Permutation Importance', 'Feature Importance'],
                'business_optimization': 'Threshold optimization for profit maximization',
                'timestamp': datetime.now().isoformat()
            }
            
            training_summary_path = output_dir / "training_summary.json"
            with open(training_summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            logger.info("⚠️ Full model training not implemented in this script.")
            logger.info("   To run complete training, use the modular approach:")
            logger.info("   1. python scripts/train_traditional_models.py")
            logger.info("   2. python scripts/train_advanced_models.py") 
            logger.info("   3. python scripts/run_ensemble_training.py")
            logger.info("   4. python scripts/evaluate_models.py")
            
            logger.info(f"📊 Training configuration saved to {training_summary_path}")
        
        # =============================================================================
        # FINAL SUMMARY
        # =============================================================================
        logger.info("\n" + "="*80)
        logger.info("📋 PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        
        final_summary = {
            'pipeline_version': '1.0.0',
            'execution_timestamp': datetime.now().isoformat(),
            'input_data': {
                'file_path': args.data_path,
                'original_shape': df.shape,
                'target_column': args.target_col
            },
            'feature_engineering': {
                'original_features': len(numeric_cols) + len(categorical_cols),
                'engineered_features': len(numeric_cols_updated) + len(categorical_cols_updated),
                'new_features_created': len(created_features),
                'removed_features': len(COLUMNS_TO_DROP)
            },
            'configuration': {
                'random_state': RANDOM_STATE,
                'train_test_split': f"{int(TRAIN_SIZE*100)}/{int(TEST_SIZE*100)}%",
                'cv_folds': CV_FOLDS,
                'cv_repeats': CV_REPEATS
            },
            'output_directory': str(output_dir),
            'files_generated': [
                'dataset_summary.json',
                'feature_engineering_summary.json',
                'preprocessing_summary.json',
                'engineered_dataset.csv'
            ],
            'next_steps': [
                'Run preprocessing pipeline',
                'Train traditional models',
                'Train advanced models', 
                'Create ensemble methods',
                'Perform interpretability analysis',
                'Generate final model card'
            ]
        }
        
        if args.save_visualizations:
            final_summary['files_generated'].extend([
                '01_eda_visualizations.png',
                '02_feature_engineering_visualizations.png'
            ])
        
        if args.run_full_training:
            final_summary['files_generated'].append('training_summary.json')
        
        final_summary_path = output_dir / "pipeline_summary.json"
        with open(final_summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        logger.info(f"📊 Dataset Analysis:")
        logger.info(f"   Original features: {len(numeric_cols) + len(categorical_cols)}")
        logger.info(f"   Engineered features: {len(numeric_cols_updated) + len(categorical_cols_updated)}")
        logger.info(f"   New features created: {len(created_features)}")
        logger.info(f"   Target distribution: {df[args.target_col].value_counts().to_dict()}")
        
        logger.info(f"📁 Output Files:")
        for file_name in final_summary['files_generated']:
            file_path = output_dir / file_name
            if file_path.exists():
                logger.info(f"   ✅ {file_name}")
            else:
                logger.info(f"   ❌ {file_name} (not created)")
        
        logger.info(f"🎯 Next Steps:")
        for step in final_summary['next_steps']:
            logger.info(f"   • {step}")
        
        logger.info(f"📁 All results saved to: {output_dir}")
        logger.info(f"📋 Final summary saved to: {final_summary_path}")
        
        # Calculate execution time
        execution_time = datetime.now() - datetime.fromisoformat(final_summary['execution_timestamp'])
        logger.info(f"⏱️ Total execution time: {execution_time.total_seconds():.1f} seconds")
        
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def validate_inputs(args):
    """Validate input arguments."""
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if not args.data_path.endswith('.csv'):
        raise ValueError("Data file must be a CSV file")
    
    # Validate target column will be done during data loading


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
