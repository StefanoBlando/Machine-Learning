#!/usr/bin/env python3
"""
Simple model training script for Advanced Income Classification.
This script demonstrates the basic usage of the repository modules.

Usage:
    python scripts/train_models.py --data-path data/raw/data.csv --output-dir results/
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import *
from src.data.loader import load_and_explore_data
from src.data.feature_engineering import advanced_feature_engineering
from src.preprocessing.pipeline import advanced_preprocessing_and_split
from src.models.traditional import setup_traditional_models
from src.models.advanced import setup_advanced_models
from src.optimization.hyperparameter_tuning import run_optimization_for_all_strategies
from src.evaluation.metrics import create_custom_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Advanced Income Classification Models')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the CSV data file')
    parser.add_argument('--output-dir', type=str, default='results/experiments',
                       help='Output directory for results')
    parser.add_argument('--target-col', type=str, default='target',
                       help='Name of the target column')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['traditional', 'advanced', 'all'],
                       default=['traditional'],
                       help='Which models to train')
    parser.add_argument('--quick-run', action='store_true',
                       help='Use fewer iterations for quick testing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / get_timestamp()
    ensure_dir_exists(output_dir)
    
    logger.info("🚀 Advanced Income Classification - Model Training")
    logger.info("=" * 60)
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Models to train: {args.models}")
    logger.info(f"Quick run: {args.quick_run}")
    
    try:
        # Step 1: Load and explore data
        logger.info("\n📊 Step 1: Loading and exploring data...")
        df, numeric_cols, categorical_cols = load_and_explore_data(
            filepath=args.data_path,
            target_col=args.target_col
        )
        
        # Step 2: Feature engineering
        logger.info("\n🔧 Step 2: Feature engineering...")
        df_engineered, numeric_cols_updated, categorical_cols_updated, created_features = \
            advanced_feature_engineering(df, args.target_col)
        
        # Step 3: Preprocessing and splitting
        logger.info("\n⚙️ Step 3: Preprocessing and data splitting...")
        preprocessing_results = advanced_preprocessing_and_split(
            df_engineered, args.target_col, TEST_SIZE, RANDOM_STATE
        )
        
        # Step 4: Setup models
        logger.info("\n🤖 Step 4: Setting up models...")
        models_config = {}
        
        if 'traditional' in args.models or 'all' in args.models:
            traditional_models = setup_traditional_models(preprocessing_results['class_weight_dict'])
            models_config.update(traditional_models)
            logger.info(f"✅ Traditional models configured: {len(traditional_models)}")
        
        if 'advanced' in args.models or 'all' in args.models:
            advanced_models = setup_advanced_models(preprocessing_results['y_train'])
            models_config.update(advanced_models)
            logger.info(f"✅ Advanced models configured: {len(advanced_models)}")
        
        if not models_config:
            raise ValueError("No models configured!")
        
        # Step 5: Create custom scorer
        logger.info("\n📈 Step 5: Creating custom scorer...")
        custom_scorer = create_custom_scorer()
        
        # Step 6: Hyperparameter optimization
        logger.info("\n🔍 Step 6: Hyperparameter optimization...")
        
        # Reduce iterations for quick run
        if args.quick_run:
            logger.info("⚡ Quick run mode: reducing iterations")
            global SMART_ITERATIONS
            SMART_ITERATIONS = {k: max(2, v // 3) for k, v in SMART_ITERATIONS.items()}
        
        all_results = run_optimization_for_all_strategies(
            models_config=models_config,
            balanced_datasets=preprocessing_results['balanced_datasets'],
            cv_strategies=preprocessing_results['cv_strategies'],
            custom_scorer=custom_scorer,
            X_val_preprocessed=preprocessing_results['X_val_preprocessed'],
            y_val=preprocessing_results['y_val'],
            use_focused_params=True
        )
        
        # Step 7: Save results
        logger.info("\n💾 Step 7: Saving results...")
        from src.optimization.hyperparameter_tuning import save_optimization_results, create_optimization_report
        
        save_optimization_results(all_results, str(output_dir))
        
        # Generate and save report
        report = create_optimization_report(all_results)
        report_path = output_dir / "training_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"📄 Report saved to: {report_path}")
        
        # Print summary
        successful_models = [r for r in all_results if r['best_model'] is not None]
        if successful_models:
            best_model = max(successful_models, 
                           key=lambda x: x['validation_metrics_optimal_threshold'].get('f1', 0))
            
            logger.info("\n🏆 TRAINING SUMMARY")
            logger.info("=" * 30)
            logger.info(f"Total configurations: {len(all_results)}")
            logger.info(f"Successful trainings: {len(successful_models)}")
            logger.info(f"Best model: {best_model['model_name']} ({best_model['sampling_strategy']})")
            logger.info(f"Best F1-Score: {best_model['validation_metrics_optimal_threshold']['f1']:.4f}")
            logger.info(f"Training completed successfully! 🎉")
        else:
            logger.error("❌ No models were successfully trained!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
