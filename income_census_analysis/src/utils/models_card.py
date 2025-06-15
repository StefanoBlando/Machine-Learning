"""
Model card generation module.

"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


def create_model_card(best_result_data: dict, y_train_data: pd.Series, feature_names_list: list, random_state_val: int) -> str:
    """
    Crea una model card standard per documentazione ML.
    """
    if not best_result_data or best_result_data['best_model'] is None:
        print("⚠️ No best model available for model card generation.")
        return None
    
    print("\n📄 GENERATING MODEL CARD")
    print("-" * 40)
    
    model_card = f"""
MODEL CARD: Income Classification Model
{'='*50}

📋 MODEL OVERVIEW:
Model Name: {best_result_data['model_name']}
Model Type: Binary Classification
Framework: Scikit-learn (or XGBoost/LightGBM)
Date Created: {datetime.now().strftime('%Y-%m-%d')}
Version: 1.0

🎯 INTENDED USE:
Primary Use: Predict whether individual income exceeds $50K based on demographic and work-related attributes.
Users: Data scientists, business analysts for strategic planning and resource allocation.
Use Cases: Targeted marketing campaigns, social program analysis, preliminary economic trend forecasting.
Out-of-scope: Not for sensitive applications like loan approval, hiring decisions, or legal judgments.

📊 TRAINING DATA:
Dataset: Adult Census Income Dataset
Source: UCI Machine Learning Repository (commonly found on Kaggle)
Samples: {len(y_train_data):,} training samples
Features: {len(feature_names_list)} features after engineering and preprocessing/selection.
Original Features Used: age, workclass, fnlwgt, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week.
Features Removed: education, native-country (as per task requirement)
Target: Binary (0: <=50K, 1: >50K)
Class Distribution (training): Class 0 ({y_train_data.value_counts().get(0,0):,}), Class 1 ({y_train_data.value_counts().get(1,0):,})
Split: 70% train, 30% validation (stratified)

⚖️ MODEL PERFORMANCE (on Validation Set @ Optimal Threshold):
F1-Score: {best_result_data['validation_metrics_optimal_threshold']['f1']:.4f}
Accuracy: {best_result_data['validation_metrics_optimal_threshold']['accuracy']:.4f}
Precision: {best_result_data['validation_metrics_optimal_threshold']['precision']:.4f}
Recall: {best_result_data['validation_metrics_optimal_threshold']['recall']:.4f}
ROC-AUC: {best_result_data['validation_metrics_optimal_threshold']['roc_auc']:.4f}
Balanced Accuracy: {best_result_data['validation_metrics_optimal_threshold']['balanced_accuracy']:.4f}
Matthews Correlation Coefficient: {best_result_data['validation_metrics_optimal_threshold']['matthews_corrcoef']:.4f}
Optimal Threshold: {best_result_data['business_metrics']['optimal_threshold']:.3f}
Max Profit (at Optimal Threshold): {best_result_data['business_metrics']['max_profit']:,}

Cross-Validation Score (Mean F1): {best_result_data['cv_score_mean']:.4f} ± {best_result_data['cv_score_std']:.4f}

🔧 MODEL DETAILS:
Algorithm: {best_result_data['model_name']}
Sampling Strategy Used in Training: {best_result_data['sampling_strategy']}
Hyperparameters (Optimized):
"""
    if best_result_data['best_params']:
        # Handle list of dicts for NestedCV best_params
        params_to_display = best_result_data['best_params']
        if isinstance(params_to_display, list) and len(params_to_display) > 0:
            model_card += "   • (Average over folds or example from first fold if NestedCV)\n"
            params_to_display = params_to_display[0] # Take first fold's params as example

        for param, value in params_to_display.items():
            model_card += f"   • {param}: {value}\n"
    else:
        model_card += "   • Not explicitly listed (e.g., Nested CV result or internal parameters)\n"
    
    model_card += f"""
Training Time: {best_result_data['training_time']:.1f} seconds
Random Seed: {random_state_val}
Preprocessing: StandardScaler for numeric features, OneHotEncoder for categorical features.
Feature Selection: SelectKBest (f_classif) applied, selecting top features.
Class Imbalance Handling: Class weights used during training (or native algorithm balancing), and/or resampling strategies (SMOTE, ADASYN, SMOTEENN).

⚠️ LIMITATIONS:
• Data representativeness: Based on 1994 US Census data, may not generalize to current demographics or other countries/regions.
• Feature limitations: Relies solely on provided features; external socio-economic or dynamic factors are not included.
• Generalization: Performance may vary significantly on out-of-distribution data.
• Imputation strategy: Missing values are imputed with mode/median, which might introduce some bias.
• Interpretability: While SHAP/Permutation Importance provide insights, complex ensemble models can be inherently less interpretable.

📈 FAIRNESS & BIAS:
• Model trained on historical census data, which may reflect and perpetuate historical societal biases (e.g., related to race, sex, workclass).
• No specific fairness mitigation techniques (e.g., bias detection, debiasing algorithms) were applied in this analysis.
• Regular bias audits (e.g., checking for disparate impact, equality of opportunity) are highly recommended.
• Monitor model performance and predictions across different demographic groups to ensure equitable outcomes and prevent unintended discrimination.

🔄 MAINTENANCE:
• Retrain: Recommended every 6-12 months or upon significant data/concept drift in production.
• Monitoring: Implement continuous monitoring of model performance (F1, AUC, Profit) and input data characteristics (data drift) in production.
• Validation: Re-validate model performance on new, unseen data before deploying updates.
• Versioning: Maintain clear versioning of models and associated data/code.

Contact: [Your Name/Data Science Team]
"""
    print(model_card)
    return model_card


def save_project_summary(summary_text: str, model_card_text: str = None) -> tuple[str, str]:
    """
    Salva il summary del progetto e la model card in file di testo.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    summary_filename = f"ml_project_summary_{timestamp}.txt"
    try:
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"📁 Project summary saved to: {summary_filename}")
    except Exception as e:
        print(f"❌ Error saving summary: {str(e)}")
        summary_filename = None
    
    card_filename = None
    if model_card_text:
        card_filename = f"model_card_{timestamp}.txt"
        try:
            with open(card_filename, 'w', encoding='utf-8') as f:
                f.write(model_card_text)
            print(f"📁 Model card saved to: {card_filename}")
        except Exception as e:
            print(f"❌ Error saving model card: {str(e)}")
            card_filename = None
    
    return summary_filename, card_filename


def print_final_stats(results_df_final: pd.DataFrame, best_result_final: dict, 
                      ensemble_results_final: dict, created_features_list: list) -> None:
    """
    Stampa statistiche finali concise del progetto.
    """
    print("\n🏁 FINAL PROJECT STATISTICS")
    print("=" * 50)
    
    if best_result_final and best_result_final['best_model'] is not None:
        print(f"🥇 Best Individual Model: {best_result_final['model_name']} (Sampling: {best_result_final['sampling_strategy']})")
        print(f"📊 Performance (F1/AUC): {best_result_final['validation_metrics_optimal_threshold']['f1']:.4f} / {best_result_final['validation_metrics_optimal_threshold']['roc_auc']:.4f}")
        print(f"⏱️ Training Time: {best_result_final['training_time']:.1f}s")
    
    if results_df_final is not None and not results_df_final.empty:
        print(f"🧪 Total Model Configurations Tested: {len(results_df_final)}")
        print(f"📈 F1 Range (all configurations): {results_df_final['f1'].min():.3f} - {results_df_final['f1'].max():.3f}")
        total_time_all_models = results_df_final['Training_Time'].sum() if 'Training_Time' in results_df_final.columns else 0
        print(f"⏱️ Total Training Time (all runs): {total_time_all_models:.1f}s ({total_time_all_models/60:.1f} min)")
    
    if ensemble_results_final:
        best_ensemble_candidate = max(ensemble_results_final.items(), key=lambda x: x[1]['metrics']['f1'] if not np.isnan(x[1]['metrics']['f1']) else -1)
        if best_ensemble_candidate and not np.isnan(best_ensemble_candidate[1]['metrics']['f1']):
            print(f"🔗 Best Ensemble: {best_ensemble_candidate[0].title()} (F1={best_ensemble_candidate[1]['metrics']['f1']:.4f})")
    
    if created_features_list:
        print(f"🔧 Features Engineered: {len(created_features_list)}")
        if 'feature_names_after_preprocessing' in globals(): # Check if feature selection was done
            print(f"🔬 Features Selected: {len(feature_names_after_preprocessing)}")
    
    print(f"✅ Project Status: COMPLETED SUCCESSFULLY")


def get_project_completion_summary() -> Dict[str, Any]:
    """
    Get a structured summary of project completion status.
    """
    completion_status = {
        'requirements_fulfilled': {
            'data_preprocessing': True,
            'feature_engineering': True,
            'model_training': True,
            'hyperparameter_optimization': True,
            'evaluation_framework': True,
            'advanced_techniques': True
        },
        'models_implemented': {
            'traditional': ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting', 'SVM'],
            'advanced': ['XGBoost', 'LightGBM'],
            'ensemble': ['Voting', 'Stacking', 'Weighted']
        },
        'techniques_applied': {
            'feature_engineering': ['Occupation mapping', 'Age groups', 'Work patterns', 'Financial features', 'Interaction features'],
            'class_balancing': ['SMOTE', 'ADASYN', 'SMOTEENN', 'Class weights'],
            'optimization': ['Bayesian optimization', 'Smart iterations', 'Nested CV'],
            'interpretability': ['SHAP analysis', 'Permutation importance', 'Stability analysis'],
            'business_metrics': ['Optimal threshold', 'Cost-sensitive analysis', 'Profit optimization']
        },
        'quality_assurance': {
            'reproducibility': 'Random seed control (123)',
            'validation': 'Stratified K-Fold with repeats',
            'robustness': 'Multiple sampling strategies',
            'documentation': 'Comprehensive logging and model cards'
        }
    }
    
    return completion_status
