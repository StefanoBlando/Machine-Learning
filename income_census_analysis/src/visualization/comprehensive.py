"""
Advanced comprehensive visualizations module.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, auc

from ..config.settings import RANDOM_STATE


def create_comprehensive_visualizations(results_df: pd.DataFrame, best_result: dict, 
                                        ensemble_results: dict, interpretability_results: dict, 
                                        all_results: list, y_val: pd.Series, feature_names: list):
    """
    Crea visualizzazioni comprehensive per l'analisi completa, con gestione degli errori.
    Aggiornato per riflettere le diverse strategie di campionamento e metriche ottimizzate.
    """
    print("\n9. ADVANCED VISUALIZATIONS")
    print("-" * 50)
    
    try:
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
    except Exception:
        plt.style.use('default') 
        
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    fig = plt.figure(figsize=(24, 18)) # Increased figure size for more subplots
    gs = fig.add_gridspec(4, 4, hspace=0.7, wspace=0.4) # Increased hspace/wspace for better separation
    
    # 1. Model Performance Comparison (Top subplot - across all sampling strategies)
    ax1 = fig.add_subplot(gs[0, :2])
    try:
        if results_df is not None and not results_df.empty and 'f1' in results_df.columns:
            plot_data = results_df.dropna(subset=['f1']).sort_values('f1', ascending=False).head(10) # Take top 10 overall
            if not plot_data.empty:
                # Combine Model and Sampling Strategy for labels
                models_labels = [f"{str(row['Model']).replace('_', ' ')} ({str(row['Sampling_Strategy'])})" for idx, row in plot_data.iterrows()]
                # Shorten labels if too long
                models_short = [label[:25] + '...' if len(label) > 25 else label for label in models_labels]

                x_pos = np.arange(len(models_short))
                bars = ax1.bar(x_pos, plot_data['f1'], alpha=0.8, color='skyblue', edgecolor='navy')
                ax1.set_xlabel('Model Configuration', fontsize=10)
                ax1.set_ylabel('F1-Score (Optimal Threshold)', fontsize=10)
                ax1.set_title('Top 10 Model Configurations by F1-Score (Optimal Threshold)', fontsize=12, fontweight='bold')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(models_short, rotation=60, ha='right', fontsize=8) # Increased rotation
                ax1.grid(True, axis='y', alpha=0.3)
                for bar, f1_score in zip(bars, plot_data['f1']):
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                            f'{f1_score:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax1.text(0.5, 0.5, 'No Valid Performance Data', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Model Performance Comparison', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Results Data Available', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Model Performance Comparison', fontweight='bold')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        ax1.set_title('Model Performance Comparison', fontweight='bold')
    
    # 2. ROC Curves Comparison
    ax2 = fig.add_subplot(gs[0, 2:])
    try:
        if all_results and y_val is not None:
            # Get top 5 models overall based on optimal threshold F1
            top_models_overall = sorted([r for r in all_results if r['best_model'] is not None and not np.isnan(r['validation_metrics_optimal_threshold'].get('f1', np.nan)) and r['y_val_proba'] is not None and len(np.unique(y_val)) > 1], 
                                key=lambda x: x['validation_metrics_optimal_threshold']['f1'], reverse=True)[:5] # Ensure valid proba and 2 classes
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            roc_plotted = False
            for i, result in enumerate(top_models_overall):
                # Check for valid proba and multiple classes *again* to be safe within the loop
                if 'y_val_proba' in result and result['y_val_proba'] is not None and len(np.unique(y_val)) > 1:
                    try:
                        fpr, tpr, _ = roc_curve(y_val, result['y_val_proba'])
                        auc_score = auc(fpr, tpr)
                        label_name = f"{result['model_name'].replace('_', ' ')} ({result['sampling_strategy']}) (AUC={auc_score:.3f})"
                        ax2.plot(fpr, tpr, color=colors[i], linewidth=2, label=label_name)
                        roc_plotted = True
                    except Exception as e:
                        print(f"   ⚠️ ROC curve plotting issue for {result['model_name']}: {e}")
            
            if ensemble_results:
                ensemble_colors = ['darkred', 'darkblue', 'darkgreen']
                ensemble_styles = ['-', '--', '-.']
                for i, (ensemble_name, ensemble_data) in enumerate(ensemble_results.items()):
                    if 'probabilities' in ensemble_data and ensemble_data['probabilities'] is not None and len(np.unique(y_val)) > 1:
                        try:
                            fpr, tpr, _ = roc_curve(y_val, ensemble_data['probabilities'])
                            auc_score = auc(fpr, tpr)
                            ax2.plot(fpr, tpr, color=ensemble_colors[i % len(ensemble_colors)], linewidth=3, 
                                    linestyle=ensemble_styles[i % len(ensemble_styles)],
                                    label=f"{ensemble_name.title().replace('_', ' ')} Ens. (AUC={auc_score:.3f})")
                            roc_plotted = True
                        except Exception as e:
                            print(f"   ⚠️ ROC curve plotting issue for {ensemble_name}: {e}")
            
            if roc_plotted:
                ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('ROC Curves - Top Models & Ensembles', fontweight='bold')
                ax2.legend(loc='lower right', fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No Prob. Predictions for ROC', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('ROC Curves', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Results for ROC', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('ROC Curves', fontweight='bold')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title('ROC Curves', fontweight='bold')
    
    # 3. Feature Importance Analysis (based on the overall best individual model)
    ax3 = fig.add_subplot(gs[1, :2])
    try:
        if interpretability_results and 'feature_importance' in interpretability_results and not interpretability_results['feature_importance'].empty:
            importance_df = interpretability_results['feature_importance'].dropna(subset=['importance'])
            if not importance_df.empty:
                top_features = importance_df.head(15).sort_values('importance', ascending=True) 
                y_pos = np.arange(len(top_features))
                ax3.barh(y_pos, top_features['importance'], alpha=0.8, color='lightcoral')
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels([str(f).replace('_', ' ')[:25] + '...' if len(str(f)) > 25 else str(f).replace('_', ' ') for f in top_features['feature']], fontsize=8)
                ax3.set_xlabel('Feature Importance')
                ax3.set_title(f'Top 15 Feature Importances\n({best_result["model_name"].replace("_", " ") if best_result else "N/A"})', fontweight='bold')
                ax3.grid(True, axis='x', alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No Valid Feature Importance Data', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Feature Importance', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Feature Importance Not Available', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Feature Importance', fontweight='bold')
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        ax3.set_title('Feature Importance', fontweight='bold')
    
    # 4. SHAP Importance (if available)
    ax4 = fig.add_subplot(gs[1, 2:])
    try:
        if interpretability_results and 'shap_importance' in interpretability_results and not interpretability_results['shap_importance'].empty:
            shap_df = interpretability_results['shap_importance'].dropna(subset=['shap_importance'])
            if not shap_df.empty:
                top_shap = shap_df.head(15).sort_values('shap_importance', ascending=True) 
                y_pos = np.arange(len(top_shap))
                ax4.barh(y_pos, top_shap['shap_importance'], alpha=0.8, color='lightgreen')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels([str(f).replace('_', ' ')[:25] + '...' if len(str(f)) > 25 else str(f).replace('_', ' ') for f in top_shap['feature']], fontsize=8)
                ax4.set_xlabel('SHAP Importance (Mean |SHAP value|)')
                ax4.set_title('Top 15 SHAP Feature Importances', fontweight='bold')
                ax4.grid(True, axis='x', alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Valid SHAP Data', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                ax4.set_title('SHAP Analysis', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'SHAP Analysis Not Available', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('SHAP Analysis', fontweight='bold')
    except Exception as e:
        ax4.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('SHAP Analysis', fontweight='bold')
    
    # 5. Confusion Matrix - Best Model (at optimal threshold)
    ax5 = fig.add_subplot(gs[2, 0])
    try:
        if best_result and 'y_val_pred_optimal' in best_result and best_result['y_val_pred_optimal'] is not None and len(np.unique(y_val)) > 1:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_val, best_result['y_val_pred_optimal'], labels=[0,1])
            # Pad if needed, though confusion_matrix with labels=[0,1] should handle it
            if cm.shape != (2,2):
                temp_cm = np.zeros((2,2), dtype=int)
                unique_pred = np.unique(best_result['y_val_pred_optimal'])
                if 0 in unique_pred: temp_cm[0,0] = ((y_val == 0) & (best_result['y_val_pred_optimal'] == 0)).sum(); temp_cm[1,0] = ((y_val == 1) & (best_result['y_val_pred_optimal'] == 0)).sum()
                if 1 in unique_pred: temp_cm[0,1] = ((y_val == 0) & (best_result['y_val_pred_optimal'] == 1)).sum(); temp_cm[1,1] = ((y_val == 1) & (best_result['y_val_pred_optimal'] == 1)).sum()
                cm = temp_cm

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                       xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
            ax5.set_title(f'Confusion Matrix\n({best_result["model_name"].replace("_", " ")} @ Optimal Thresh)', fontweight='bold')
            ax5.set_ylabel('True Label')
            ax5.set_xlabel('Predicted Label')
        else:
            ax5.text(0.5, 0.5, 'No Predictions Available', ha='center', va='center', transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Confusion Matrix', fontweight='bold')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=ax5.transAxes, fontsize=10)
        ax5.set_title('Confusion Matrix', fontweight='bold')
    
    # 6. Business Metrics Comparison (Max Profit)
    ax6 = fig.add_subplot(gs[2, 1])
    try:
        # Filter out NaN profits and take top 5
        # Ensure plot_data is created from results_df. Max_Profit could be NaN.
        plot_data = results_df.dropna(subset=['Max_Profit']).sort_values('Max_Profit', ascending=False).head(5)
        
        if not plot_data.empty:
            profit_labels = [f"{str(row['Model']).replace('_', ' ')} ({str(row['Sampling_Strategy'])})" for idx, row in plot_data.iterrows()]
            profit_labels_short = [label[:25] + '...' if len(label) > 25 else label for label in profit_labels]

            bars = ax6.bar(np.arange(len(plot_data)), plot_data['Max_Profit'], alpha=0.8, color='cornflowerblue', edgecolor='darkblue')
            ax6.set_xticks(np.arange(len(profit_labels_short)))
            ax6.set_xticklabels(profit_labels_short, rotation=60, ha='right', fontsize=8)
            ax6.set_ylabel('Max Profit')
            ax6.set_title('Top 5 Model Configurations by Max Profit', fontweight='bold')
            ax6.grid(True, axis='y', alpha=0.3)
            for bar, profit_val in zip(bars, plot_data['Max_Profit']):
                height = bar.get_height()
                # Check if height is not NaN and max profit for scaling text
                if not np.isnan(height) and not plot_data['Max_Profit'].empty and plot_data['Max_Profit'].max() > 0:
                    ax6.text(bar.get_x() + bar.get_width()/2., height + plot_data['Max_Profit'].max()*0.01,
                            f'{int(profit_val)}', ha='center', va='bottom', fontsize=10)
                elif not np.isnan(height): # Handle case where max profit is 0 or negative
                    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1, # Small offset
                            f'{int(profit_val)}', ha='center', va='bottom', fontsize=10)
        else:
            ax6.text(0.5, 0.5, 'No Valid Business Profit Data', ha='center', va='center', transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Business Metrics Comparison', fontweight='bold')
    except Exception as e:
        ax6.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=ax6.transAxes, fontsize=10)
        ax6.set_title('Business Metrics Comparison', fontweight='bold')
    
    # 7. Model Performance Radar Chart (Overall Best Individual vs. Ensembles)
    ax7 = fig.add_subplot(gs[2, 2], projection='polar')
    try:
        if best_result and 'validation_metrics_optimal_threshold' in best_result and not np.isnan(best_result['validation_metrics_optimal_threshold'].get('f1', np.nan)):
            metrics_radar = ['f1', 'precision', 'recall', 'roc_auc', 'balanced_accuracy']
            metrics_labels = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'Balanced Acc.']
            
            best_values = [max(0, min(1, best_result['validation_metrics_optimal_threshold'].get(m, 0))) for m in metrics_radar]
            best_values += best_values[:1]
            angles = [n / float(len(metrics_radar)) * 2 * np.pi for n in range(len(metrics_radar))]
            angles += angles[:1]
            
            ax7.plot(angles, best_values, 'o-', linewidth=2, label='Best Individual', color='red')
            ax7.fill(angles, best_values, alpha=0.25, color='red')
            
            if ensemble_results:
                ensemble_colors = ['blue', 'green', 'purple']
                for i, (ens_name, ens_data) in enumerate(ensemble_results.items()):
                    if 'metrics' in ens_data and not np.isnan(ens_data['metrics'].get('f1', np.nan)):
                        ens_values = [max(0, min(1, ens_data['metrics'].get(m, 0))) for m in metrics_radar]
                        ens_values += ens_values[:1]
                        ax7.plot(angles, ens_values, 'o-', linewidth=2, 
                                label=f'{ens_name.title()} Ens.', color=ensemble_colors[i % len(ensemble_colors)])
                        ax7.fill(angles, ens_values, alpha=0.15, color=ensemble_colors[i % len(ensemble_colors)])
            
            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels(metrics_labels, fontsize=8)
            ax7.set_ylim(0, 1)
            ax7.set_title('Performance Radar Chart (Optimal Threshold Metrics)', fontweight='bold', pad=20)
            ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            ax7.grid(True)
        else:
            ax7.text(0.5, 0.5, 'Performance Radar Chart Not Available', ha='center', va='center', transform=ax7.transAxes, fontsize=14)
            ax7.set_title('Performance Radar Chart', fontweight='bold')
    except Exception as e:
        ax7.remove(); fig.add_subplot(gs[2, 2]); ax7 = fig.get_axes()[-1] # Replace polar with normal axes on error
        ax7.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=ax7.transAxes, fontsize=10)
        ax7.set_title('Performance Radar Chart', fontweight='bold')
    
    # 8. Training Time vs Performance
    ax8 = fig.add_subplot(gs[2, 3])
    try:
        if results_df is not None and not results_df.empty and 'Training_Time' in results_df.columns and 'f1' in results_df.columns:
            plot_data = results_df.dropna(subset=['Training_Time', 'f1'])
            if not plot_data.empty:
                scatter = ax8.scatter(plot_data['Training_Time'], plot_data['f1'], 
                                    c=plot_data['roc_auc'] if 'roc_auc' in plot_data.columns else 'blue', 
                                    cmap='viridis', alpha=0.7, s=plot_data['balanced_accuracy']*100 + 50 if 'balanced_accuracy' in plot_data.columns else 60,
