# =============================================================================
# TIM HACKATHON - MODULE 1: EDA & DATA LOADING (COMPLETE)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# PART 1: DATA LOADER CLASS
# =============================================================================

class TIMDataLoader:
   """
   Professional data loader for TIM Hackathon dataset
   Handles loading, merging, validation and preprocessing
   """
   
   def __init__(self, data_path=".", random_state=42):
       self.data_path = Path(data_path)
       self.random_state = random_state
       self.actions_df = None
       self.features_df = None
       self.merged_df = None
       self.ranking_df = None
       self.all_actions = None
       
   def load_datasets(self):
       """Load actions and features datasets with validation"""
       print("Loading TIM Hackathon datasets...")
       
       # Load actions
       try:
           self.actions_df = pd.read_csv(self.data_path / "actions.csv", delimiter=';')
           print(f"Actions loaded: {self.actions_df.shape}")
       except Exception as e:
           raise FileNotFoundError(f"Error loading actions.csv: {e}")
       
       # Load features
       try:
           self.features_df = pd.read_csv(self.data_path / "features.csv", delimiter=';')
           print(f"Features loaded: {self.features_df.shape}")
       except Exception as e:
           raise FileNotFoundError(f"Error loading features.csv: {e}")
       
       # Validate datasets
       self._validate_datasets()
       return self.actions_df, self.features_df
   
   def _validate_datasets(self):
       """Comprehensive dataset validation"""
       print("Validating datasets...")
       
       # Check actions structure
       required_actions_cols = ['num_telefono', 'data_contatto', 'response', 'action']
       missing_actions = set(required_actions_cols) - set(self.actions_df.columns)
       if missing_actions:
           raise ValueError(f"Actions missing columns: {missing_actions}")
       
       # Check features structure
       required_features_cols = ['num_telefono', 'data_contatto']
       missing_features = set(required_features_cols) - set(self.features_df.columns)
       if missing_features:
           raise ValueError(f"Features missing columns: {missing_features}")
       
       # Check PCA features
       pca_cols = [col for col in self.features_df.columns if 'scaledPcaFeatures' in col]
       if len(pca_cols) != 64:
           print(f"Warning: Expected 64 PCA features, found {len(pca_cols)}")
       
       print("Dataset validation completed successfully")
   
   def diagnose_merge_structure(self):
       """Diagnose merge structure to understand multiple actions per day"""
       print("MERGE STRUCTURE DIAGNOSIS")
       print("="*50)
       
       # Convert dates
       self.actions_df['data_contatto'] = pd.to_datetime(self.actions_df['data_contatto'])
       self.features_df['data_contatto'] = pd.to_datetime(self.features_df['data_contatto'])
       
       # Check for multiple actions per customer-date
       actions_grouped = self.actions_df.groupby(['num_telefono', 'data_contatto']).size()
       actions_multi = (actions_grouped > 1).sum()
       
       print(f"Customer-date pairs with multiple actions: {actions_multi:,}")
       print(f"Max actions per customer-date: {actions_grouped.max()}")
       
       if actions_multi > 0:
           multi_dist = actions_grouped[actions_grouped > 1].value_counts().sort_index()
           print("Distribution of multiple actions per customer-date:")
           for count, freq in multi_dist.items():
               print(f"  {count} actions: {freq:,} pairs")
       
       return actions_grouped
   
   def create_ranking_dataset(self):
       """Create ranking dataset for Learning-to-Rank"""
       print("CREATING RANKING DATASET FOR LEARNING-TO-RANK")
       print("="*50)
       
       # Get all unique actions
       self.all_actions = sorted(self.actions_df['action'].unique())
       print(f"Total unique actions: {len(self.all_actions)}")
       
       # Create customer-date combinations
       customer_dates = self.actions_df[['num_telefono', 'data_contatto']].drop_duplicates()
       print(f"Unique customer-date combinations: {len(customer_dates):,}")
       
       # Deduplicate features
       features_dedupe = self.features_df.drop_duplicates(subset=['num_telefono', 'data_contatto'], keep='first')
       
       # Create ranking dataset
       ranking_data = []
       
       for _, row in customer_dates.iterrows():
           customer = row['num_telefono']
           date = row['data_contatto']
           
           # Get actual interactions
           actual_interactions = self.actions_df[
               (self.actions_df['num_telefono'] == customer) & 
               (self.actions_df['data_contatto'] == date)
           ]
           
           # Create action-response mapping
           action_response_map = dict(zip(actual_interactions['action'], actual_interactions['response']))
           
           # Create row for each possible action
           for action in self.all_actions:
               if action in action_response_map:
                   response = action_response_map[action]
                   target = 1 if response == 'Accettato' else 0
                   was_offered = 1
               else:
                   target = 0
                   was_offered = 0
               
               ranking_data.append({
                   'num_telefono': customer,
                   'data_contatto': date,
                   'action': action,
                   'target': target,
                   'was_offered': was_offered
               })
       
       # Convert to DataFrame and merge with features
       ranking_df = pd.DataFrame(ranking_data)
       self.ranking_df = pd.merge(ranking_df, features_dedupe, on=['num_telefono', 'data_contatto'], how='inner')
       
       # Add basic temporal features
       self.ranking_df['month'] = self.ranking_df['data_contatto'].dt.month
       self.ranking_df['week'] = self.ranking_df['data_contatto'].dt.isocalendar().week
       self.ranking_df['dayofweek'] = self.ranking_df['data_contatto'].dt.dayofweek
       self.ranking_df['is_weekend'] = (self.ranking_df['dayofweek'] >= 5).astype(int)
       
       # Add action components
       self.ranking_df['action_type'] = self.ranking_df['action'].str.split('_').str[0]
       self.ranking_df['action_category'] = self.ranking_df['action'].str.split('_').str[1] 
       self.ranking_df['action_subcategory'] = self.ranking_df['action'].str.split('_').str[2].fillna('Unknown')
       
       # Validation
       print(f"Ranking dataset created: {self.ranking_df.shape}")
       print(f"Actions offered: {self.ranking_df['was_offered'].sum():,}")
       print(f"Actions accepted: {self.ranking_df['target'].sum():,}")
       print(f"Acceptance rate (of offered): {self.ranking_df[self.ranking_df['was_offered']==1]['target'].mean():.4f}")
       
       return self.ranking_df
   
   def create_train_test_split(self, test_size=0.2):
       """Create customer-based train/test split"""
       print(f"Creating customer-based train/test split (test_size={test_size})...")
       
       unique_customers = self.ranking_df['num_telefono'].unique()
       
       train_customers, test_customers = train_test_split(
           unique_customers, test_size=test_size, random_state=self.random_state
       )
       
       train_df = self.ranking_df[self.ranking_df['num_telefono'].isin(train_customers)].copy()
       test_df = self.ranking_df[self.ranking_df['num_telefono'].isin(test_customers)].copy()
       
       print(f"Train customers: {len(train_customers):,}")
       print(f"Test customers: {len(test_customers):,}")
       print(f"Train rows: {len(train_df):,}")
       print(f"Test rows: {len(test_df):,}")
       print(f"Train target rate: {train_df['target'].mean():.4f}")
       print(f"Test target rate: {test_df['target'].mean():.4f}")
       
       return train_df, test_df
   
   def perform_data_quality_analysis(self):
       """Comprehensive data quality analysis"""
       print("PERFORMING DATA QUALITY ANALYSIS")
       print("="*40)
       
       # Use original actions for quality analysis
       df = self.actions_df
       
       quality_report = {
           'total_interactions': len(df),
           'unique_customers': df['num_telefono'].nunique(),
           'unique_actions': df['action'].nunique(),
           'acceptance_rate': (df['response'] == 'Accettato').mean(),
           'date_range': (df['data_contatto'].min(), df['data_contatto'].max())
       }
       
       # Customer interaction patterns
       customer_interactions = df['num_telefono'].value_counts()
       quality_report['interactions_per_customer'] = {
           'mean': customer_interactions.mean(),
           'median': customer_interactions.median(),
           'max': customer_interactions.max()
       }
       
       # Action performance
       action_stats = df.groupby('action').agg({
           'response': ['count', lambda x: (x == 'Accettato').sum()]
       })
       action_stats.columns = ['total_offers', 'total_accepted']
       action_stats['acceptance_rate'] = action_stats['total_accepted'] / action_stats['total_offers']
       quality_report['action_performance'] = action_stats.to_dict('index')
       
       self.quality_report = quality_report
       return quality_report

# =============================================================================
# PART 2: VISUALIZATION FUNCTIONS
# =============================================================================

def create_comprehensive_visualizations(actions_df, ranking_df):
   """Create comprehensive visualizations for EDA"""
   
   fig = plt.figure(figsize=(20, 24))
   
   # 1. Response Distribution
   ax1 = plt.subplot(4, 3, 1)
   response_counts = actions_df['response'].value_counts()
   colors = ['#2ecc71', '#e74c3c']
   ax1.pie(response_counts.values, labels=response_counts.index, autopct='%1.1f%%', 
           colors=colors, startangle=90)
   ax1.set_title('Response Distribution', fontsize=14, fontweight='bold')
   
   # 2. Customer Interaction Frequency
   ax2 = plt.subplot(4, 3, 2)
   customer_interactions = actions_df['num_telefono'].value_counts()
   ax2.hist(customer_interactions.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
   ax2.set_xlabel('Interactions per Customer')
   ax2.set_ylabel('Number of Customers')
   ax2.set_title('Customer Interaction Frequency', fontweight='bold')
   ax2.axvline(customer_interactions.mean(), color='red', linestyle='--', 
               label=f'Mean: {customer_interactions.mean():.1f}')
   ax2.legend()
   
   # 3. Monthly Patterns
   ax3 = plt.subplot(4, 3, 3)
   monthly_data = actions_df.groupby(actions_df['data_contatto'].dt.month).agg({
       'response': ['count', lambda x: (x == 'Accettato').mean()]
   })
   monthly_data.columns = ['volume', 'acceptance_rate']
   
   ax3_twin = ax3.twinx()
   ax3.bar(monthly_data.index, monthly_data['volume'], alpha=0.7, color='lightblue')
   ax3_twin.plot(monthly_data.index, monthly_data['acceptance_rate'], 'ro-', linewidth=2)
   ax3.set_xlabel('Month')
   ax3.set_ylabel('Volume', color='blue')
   ax3_twin.set_ylabel('Acceptance Rate', color='red')
   ax3.set_title('Monthly Volume vs Acceptance Rate', fontweight='bold')
   
   # 4. Action Performance
   ax4 = plt.subplot(4, 3, 4)
   action_stats = actions_df.groupby('action').agg({
       'response': ['count', lambda x: (x == 'Accettato').mean()]
   })
   action_stats.columns = ['volume', 'acceptance_rate']
   action_stats = action_stats.sort_values('acceptance_rate', ascending=True)
   
   bars = ax4.barh(range(len(action_stats)), action_stats['acceptance_rate'], 
                   color=plt.cm.RdYlGn(action_stats['acceptance_rate']))
   ax4.set_yticks(range(len(action_stats)))
   ax4.set_yticklabels([action.replace('Upselling_', '').replace('_', '\n')[:20] 
                        for action in action_stats.index], fontsize=8)
   ax4.set_xlabel('Acceptance Rate')
   ax4.set_title('Action Performance Ranking', fontweight='bold')
   
   # 5. Volume vs Performance
   ax5 = plt.subplot(4, 3, 5)
   scatter = ax5.scatter(action_stats['volume'], action_stats['acceptance_rate'], 
                        s=action_stats['volume']*0.3, alpha=0.6, 
                        c=action_stats['acceptance_rate'], cmap='RdYlGn')
   ax5.set_xlabel('Action Volume')
   ax5.set_ylabel('Acceptance Rate')
   ax5.set_title('Volume vs Performance', fontweight='bold')
   ax5.grid(True, alpha=0.3)
   plt.colorbar(scatter, ax=ax5)
   
   # 6. Day of Week Patterns
   ax6 = plt.subplot(4, 3, 6)
   actions_df['dayofweek'] = actions_df['data_contatto'].dt.day_name()
   day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
   daily_acceptance = actions_df.groupby('dayofweek')['response'].apply(
       lambda x: (x == 'Accettato').mean()
   ).reindex(day_order)
   
   bars = ax6.bar(day_order, daily_acceptance.values, color='orange', alpha=0.8)
   ax6.set_ylabel('Acceptance Rate')
   ax6.set_title('Acceptance Rate by Day', fontweight='bold')
   ax6.tick_params(axis='x', rotation=45)
   
   # 7. Action Categories
   ax7 = plt.subplot(4, 3, 7)
   actions_df['action_category'] = actions_df['action'].str.split('_').str[1]
   category_performance = actions_df.groupby('action_category').agg({
       'response': ['count', lambda x: (x == 'Accettato').mean()]
   })
   category_performance.columns = ['volume', 'acceptance_rate']
   
   bars = ax7.bar(category_performance.index, category_performance['acceptance_rate'], 
                  color=plt.cm.viridis(np.linspace(0, 1, len(category_performance))))
   ax7.set_ylabel('Acceptance Rate')
   ax7.set_title('Performance by Category', fontweight='bold')
   ax7.tick_params(axis='x', rotation=45)
   
   # 8. Multiple Actions Analysis
   ax8 = plt.subplot(4, 3, 8)
   actions_per_day = actions_df.groupby(['num_telefono', 'data_contatto']).size()
   multi_dist = actions_per_day.value_counts().sort_index()
   
   bars = ax8.bar(multi_dist.index, multi_dist.values, color='purple', alpha=0.7)
   ax8.set_xlabel('Actions per Customer-Date')
   ax8.set_ylabel('Frequency')
   ax8.set_title('Multiple Actions Distribution', fontweight='bold')
   
   # 9. Ranking Dataset Structure
   ax9 = plt.subplot(4, 3, 9)
   ranking_summary = [
       ranking_df['target'].sum(),  # Accepted
       ranking_df['was_offered'].sum() - ranking_df['target'].sum(),  # Offered but rejected
       len(ranking_df) - ranking_df['was_offered'].sum()  # Not offered
   ]
   labels = ['Accepted', 'Offered & Rejected', 'Not Offered']
   colors = ['#2ecc71', '#e74c3c', '#95a5a6']
   
   ax9.pie(ranking_summary, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
   ax9.set_title('Ranking Dataset Structure', fontweight='bold')
   
   # 10. PCA Features Sample
   ax10 = plt.subplot(4, 3, 10)
   pca_cols = [col for col in ranking_df.columns if 'scaledPcaFeatures' in col]
   if len(pca_cols) >= 8:
       sample_pca = pca_cols[::8]
       pca_data = ranking_df[sample_pca]
       ax10.boxplot([pca_data[col].dropna() for col in sample_pca], 
                    labels=[f'PCA{i}' for i in range(0, len(pca_cols), 8)])
       ax10.set_ylabel('Feature Value')
       ax10.set_title('PCA Features Distribution', fontweight='bold')
       ax10.tick_params(axis='x', rotation=45)
   
   # 11. Temporal Distribution
   ax11 = plt.subplot(4, 3, 11)
   daily_volume = actions_df.groupby(actions_df['data_contatto'].dt.date).size()
   ax11.plot(daily_volume.index, daily_volume.values, linewidth=1, alpha=0.7, color='purple')
   ax11.set_xlabel('Date')
   ax11.set_ylabel('Daily Interactions')
   ax11.set_title('Daily Interaction Volume', fontweight='bold')
   ax11.tick_params(axis='x', rotation=45)
   ax11.grid(True, alpha=0.3)
   
   # 12. Dataset Summary
   ax12 = plt.subplot(4, 3, 12)
   summary_stats = [
       ['Original Interactions', f"{len(actions_df):,}"],
       ['Ranking Dataset Rows', f"{len(ranking_df):,}"],
       ['Unique Customers', f"{actions_df['num_telefono'].nunique():,}"],
       ['Unique Actions', f"{actions_df['action'].nunique()}"],
       ['Acceptance Rate', f"{(actions_df['response'] == 'Accettato').mean():.3f}"],
       ['Actions Offered', f"{ranking_df['was_offered'].sum():,}"],
       ['Actions Accepted', f"{ranking_df['target'].sum():,}"],
       ['Ready for LTR', "✓"]
   ]
   
   table = ax12.table(cellText=summary_stats, colLabels=['Metric', 'Value'],
                      cellLoc='center', loc='center')
   table.auto_set_font_size(False)
   table.set_fontsize(9)
   table.scale(1.2, 1.5)
   
   for i in range(len(summary_stats) + 1):
       for j in range(2):
           cell = table[(i, j)]
           if i == 0:
               cell.set_facecolor('#34495e')
               cell.set_text_props(weight='bold', color='white')
           else:
               cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
   
   ax12.set_title('Dataset Summary', fontweight='bold')
   ax12.axis('off')
   
   plt.tight_layout()
   plt.savefig('tim_hackathon_module1_complete.png', dpi=300, bbox_inches='tight')
   plt.show()

def perform_statistical_validation(train_df, test_df):
   """Perform statistical validation of train/test split"""
   print("STATISTICAL VALIDATION")
   print("="*30)
   
   # Target distribution comparison
   train_target = train_df['target'].mean()
   test_target = test_df['target'].mean()
   
   print(f"Target rate comparison:")
   print(f"  Train: {train_target:.4f}")
   print(f"  Test:  {test_target:.4f}")
   print(f"  Difference: {abs(train_target - test_target):.4f}")
   
   # Action distribution comparison
   train_actions = train_df['action'].value_counts(normalize=True).sort_index()
   test_actions = test_df['action'].value_counts(normalize=True).sort_index()
   
   # Chi-square test
   try:
       train_counts = train_df['action'].value_counts().sort_index()
       test_counts = test_df['action'].value_counts().reindex(train_counts.index, fill_value=0)
       chi2_stat, chi2_p = chi2_contingency([train_counts, test_counts])[:2]
       print(f"Action distribution test:")
       print(f"  Chi-square p-value: {chi2_p:.4f}")
       print(f"  Distribution similarity: {'Good' if chi2_p > 0.05 else 'Review needed'}")
   except:
       print("  Action distribution test: Could not perform")
   
   # Overall assessment
   split_quality = "PASSED" if abs(train_target - test_target) < 0.01 else "REVIEW NEEDED"
   print(f"Split quality: {split_quality}")

# =============================================================================
# PART 3: MAIN EXECUTION
# =============================================================================

def main():
   """Main execution function for Module 1"""
   print("="*80)
   print("TIM HACKATHON - MODULE 1: EDA & DATA LOADING (COMPLETE)")
   print("="*80)
   
   # Initialize data loader
   loader = TIMDataLoader(random_state=RANDOM_STATE)
   
   # Step 1: Load datasets
   print("\nSTEP 1: Loading datasets...")
   actions_df, features_df = loader.load_datasets()
   
   # Step 2: Diagnose merge structure
   print("\nSTEP 2: Diagnosing data structure...")
   loader.diagnose_merge_structure()
   
   # Step 3: Create ranking dataset
   print("\nSTEP 3: Creating ranking dataset...")
   ranking_df = loader.create_ranking_dataset()
   
   # Step 4: Create train/test split
   print("\nSTEP 4: Creating train/test split...")
   train_df, test_df = loader.create_train_test_split(test_size=0.2)
   
   # Step 5: Data quality analysis
   print("\nSTEP 5: Data quality analysis...")
   quality_report = loader.perform_data_quality_analysis()
   
   # Step 6: Visualizations
   print("\nSTEP 6: Creating visualizations...")
   create_comprehensive_visualizations(actions_df, ranking_df)
   
   # Step 7: Statistical validation
   print("\nSTEP 7: Statistical validation...")
   perform_statistical_validation(train_df, test_df)
   
   print("\n" + "="*80)
   print("MODULE 1 COMPLETED SUCCESSFULLY")
   print("="*80)
   print("Generated files:")
   print("  - tim_hackathon_module1_complete.png")
   print("Datasets ready for Module 2:")
   print(f"  - ranking_df: {ranking_df.shape}")
   print(f"  - train_df: {train_df.shape}")
   print(f"  - test_df: {test_df.shape}")
   print(f"  - all_actions: {len(loader.all_actions)} actions")
   print("Ready for Learning-to-Rank baseline models!")
   
   return loader, ranking_df, train_df, test_df

# Execute Module 1
if __name__ == "__main__":
   loader, ranking_df, train_df, test_df = main()
