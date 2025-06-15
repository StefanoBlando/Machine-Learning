"""
Detailed performance visualizations module.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from ..config.settings import RANDOM_STATE


def plot_detailed_performance(results_df: pd.DataFrame):
    """
    Crea visualizzazioni dettagliate delle performance di tutti i modelli
    e strategie di campionamento dal Modulo 6.
    """
    print("\n7. DETAILED PERFORMANCE VISUALIZATIONS")
    print("-" * 50)

    if results_df is None or results_df.empty or 'f1' not in results_df.columns:
        print("❌ Nessun DataFrame di risultati valido per le visualizzazioni di performance.")
        print("   Assicurati che il Modulo 6 sia stato eseguito con successo.")
        return

    # Filter out NestedCV results for these plots, as they are a different type of evaluation
    plot_df = results_df[results_df['Sampling_Strategy'] != 'original_nested_cv'].copy()

    if plot_df.empty:
        print("⚠️ Nessun risultato non-NestedCV valido per le visualizzazioni.")
        return

    # Sort by F1 for consistent ordering in plots
    plot_df_sorted = plot_df.sort_values(by='f1', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(20, 15))
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
    sns.set_palette("tab10") # A good palette for distinct categories

    # 1. F1-Score per Modello e Strategia di Campionamento
    plt.subplot(3, 2, 1)
    sns.barplot(x='f1', y='Model', hue='Sampling_Strategy', data=plot_df_sorted, ci=None, palette='viridis')
    plt.title('F1-Score per Modello e Strategia di Campionamento (Optimal Threshold)', fontsize=14, fontweight='bold')
    plt.xlabel('F1-Score', fontsize=12)
    plt.ylabel('Modello', fontsize=12)
    plt.legend(title='Campionamento', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xlim(0.5, plot_df_sorted['f1'].max() * 1.05) # Adjust xlim for better view
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 2. ROC-AUC per Modello e Strategia di Campionamento
    plt.subplot(3, 2, 2)
    sns.barplot(x='roc_auc', y='Model', hue='Sampling_Strategy', data=plot_df_sorted, ci=None, palette='cividis')
    plt.title('ROC-AUC per Modello e Strategia di Campionamento', fontsize=14, fontweight='bold')
    plt.xlabel('ROC-AUC', fontsize=12)
    plt.ylabel('Modello', fontsize=12)
    plt.legend(title='Campionamento', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xlim(0.85, plot_df_sorted['roc_auc'].max() * 1.05) # Adjust xlim for better view
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 3. Max Profit per Modello e Strategia di Campionamento
    plt.subplot(3, 2, 3)
    sns.barplot(x='Max_Profit', y='Model', hue='Sampling_Strategy', data=plot_df_sorted, ci=None, palette='magma')
    plt.title('Max Profit per Modello e Strategia di Campionamento', fontsize=14, fontweight='bold')
    plt.xlabel('Max Profit', fontsize=12)
    plt.ylabel('Modello', fontsize=12)
    plt.legend(title='Campionamento', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 4. Optimal Threshold per Modello e Strategia di Campionamento
    plt.subplot(3, 2, 4)
    sns.barplot(x='Optimal_Threshold', y='Model', hue='Sampling_Strategy', data=plot_df_sorted, ci=None, palette='cubehelix')
    plt.title('Soglia Ottimale per Modello e Strategia di Campionamento', fontsize=14, fontweight='bold')
    plt.xlabel('Soglia Ottimale', fontsize=12)
    plt.ylabel('Modello', fontsize=12)
    plt.legend(title='Campionamento', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xlim(0, 0.5) # Thresholds are typically <= 0.5 for imbalanced
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 5. Training Time per Modello e Strategia di Campionamento
    plt.subplot(3, 2, 5)
    sns.barplot(x='Training_Time', y='Model', hue='Sampling_Strategy', data=plot_df_sorted, ci=None, palette='rocket')
    plt.title('Tempo di Training per Modello e Strategia di Campionamento', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo di Training (secondi)', fontsize=12)
    plt.ylabel('Modello', fontsize=12)
    plt.xscale('log') # Log scale for time if there's a big variance (like SVM)
    plt.legend(title='Campionamento', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout for legends
    plt.show()
    print("✅ Visualizzazioni di performance dettagliate generate!")


# Execute the function if this module is called directly from MODULO 7 context
def execute_modulo_7(results_df: pd.DataFrame):
    """
    Blocco di esecuzione del Modulo 7 - identico al notebook originale.
    """
    if 'results_df' in locals() and results_df is not None and not results_df.empty:
        try:
            plot_detailed_performance(results_df)
        except Exception as e:
            print(f"❌ Errore durante la generazione delle visualizzazioni di performance: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ Il DataFrame dei risultati (results_df) non è disponibile o è vuoto.")
        print("   Assicurati che il Modulo 6 sia stato eseguito correttamente.")

    print("\n" + "="*80)
    print("🎯 NUOVO MODULO 7 - VISUALIZZAZIONI DI PERFORMANCE DETTAGLIATE COMPLETATO")
    print("="*80)
