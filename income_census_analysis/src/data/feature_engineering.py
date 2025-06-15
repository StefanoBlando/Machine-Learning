"""
Feature engineering utilities for the Income Census Analysis project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional, List
import logging

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for advanced feature engineering and data splitting."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.data_config = self.config.data_config
        self.features_config = self.config.features_config
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group categories for better model interpretation.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with age groups
        """
        df_processed = df.copy()
        
        if 'age' in df_processed.columns:
            # Define age groups
            age_bins = [0, 25, 35, 45, 55, 65, 100]
            age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            
            df_processed['age_group'] = pd.cut(
                df_processed['age'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
            
            logger.info("Age groups created:")
            logger.info(df_processed['age_group'].value_counts().to_dict())
        
        return df_processed
    
    def create_hours_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create working hours categories.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with hours categories
        """
        df_processed = df.copy()
        
        if 'hours-per-week' in df_processed.columns:
            # Define working hours categories
            def categorize_hours(hours):
                if hours < 20:
                    return 'Part-time'
                elif hours <= 40:
                    return 'Full-time'
                elif hours <= 50:
                    return 'Overtime'
                else:
                    return 'Excessive'
            
            df_processed['hours_category'] = df_processed['hours-per-week'].apply(categorize_hours)
            
            logger.info("Hours categories created:")
            logger.info(df_processed['hours_category'].value_counts().to_dict())
        
        return df_processed
    
    def create_capital_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from capital gain and capital loss.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with capital features
        """
        df_processed = df.copy()
        
        if 'capital-gain' in df_processed.columns and 'capital-loss' in df_processed.columns:
            # Net capital (gain - loss)
            df_processed['net_capital'] = df_processed['capital-gain'] - df_processed['capital-loss']
            
            # Has capital gain/loss indicators
            df_processed['has_capital_gain'] = (df_processed['capital-gain'] > 0).astype(int)
            df_processed['has_capital_loss'] = (df_processed['capital-loss'] > 0).astype(int)
            df_processed['has_capital_activity'] = (
                (df_processed['capital-gain'] > 0) | (df_processed['capital-loss'] > 0)
            ).astype(int)
            
            # Capital gain/loss magnitude categories
            def categorize_capital(amount):
                if amount == 0:
                    return 'None'
                elif amount <= 1000:
                    return 'Low'
                elif amount <= 5000:
                    return 'Medium'
                else:
                    return 'High'
            
            df_processed['capital_gain_category'] = df_processed['capital-gain'].apply(categorize_capital)
            df_processed['capital_loss_category'] = df_processed['capital-loss'].apply(categorize_capital)
            
            logger.info("Capital features created")
        
        return df_processed
    
    def create_education_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create education-related features (if education-num is available).
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with education features
        """
        df_processed = df.copy()
        
        if 'education-num' in df_processed.columns:
            # Education level categories
            def categorize_education(edu_num):
                if edu_num <= 8:
                    return 'Elementary'
                elif edu_num <= 12:
                    return 'High School'
                elif edu_num <= 14:
                    return 'Some College'
                elif edu_num <= 16:
                    return 'Bachelor'
                else:
                    return 'Advanced'
            
            df_processed['education_level'] = df_processed['education-num'].apply(categorize_education)
            
            # High education indicator
            df_processed['high_education'] = (df_processed['education-num'] >= 13).astype(int)
            
            logger.info("Education features created:")
            logger.info(df_processed['education_level'].value_counts().to_dict())
        
        return df_processed
    
    def create_work_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create work-related features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with work features
        """
        df_processed = df.copy()
        
        # Government worker indicator
        if 'workclass' in df_processed.columns:
            government_classes = ['Federal-gov', 'Local-gov', 'State-gov']
            df_processed['is_government_worker'] = (
                df_processed['workclass'].isin(government_classes)
            ).astype(int)
            
            # Self-employed indicator
            self_employed_classes = ['Self-emp-not-inc', 'Self-emp-inc']
            df_processed['is_self_employed'] = (
                df_processed['workclass'].isin(self_employed_classes)
            ).astype(int)
        
        # Professional occupation indicator
        if 'occupation' in df_processed.columns:
            professional_occupations = ['Professional', 'Exec-managerial']
            df_processed['is_professional'] = (
                df_processed['occupation'].isin(professional_occupations)
            ).astype(int)
        
        logger.info("Work features created")
        return df_processed
    
    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create family and relationship features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with family features
        """
        df_processed = df.copy()
        
        # Married indicator
        if 'marital-status' in df_processed.columns:
            married_statuses = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
            df_processed['is_married'] = (
                df_processed['marital-status'].isin(married_statuses)
            ).astype(int)
        
        # Head of household indicator
        if 'relationship' in df_processed.columns:
            head_relationships = ['Husband', 'Wife']
            df_processed['is_head_of_household'] = (
                df_processed['relationship'].isin(head_relationships)
            ).astype(int)
        
        logger.info("Family features created")
        return df_processed
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with interaction features
        """
        df_processed = df.copy()
        
        # Age and education interaction
        if 'age' in df_processed.columns and 'education-num' in df_processed.columns:
            df_processed['age_education_interaction'] = (
                df_processed['age'] * df_processed['education-num']
            )
        
        # Hours and age interaction
        if 'hours-per-week' in df_processed.columns and 'age' in df_processed.columns:
            df_processed['hours_age_interaction'] = (
                df_processed['hours-per-week'] * df_processed['age']
            )
        
        # Education and capital gain interaction
        if 'education-num' in df_processed.columns and 'capital-gain' in df_processed.columns:
            df_processed['education_capital_interaction'] = (
                df_processed['education-num'] * np.log1p(df_processed['capital-gain'])
            )
        
        logger.info("Interaction features created")
        return df_processed
    
    def apply_all_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        df_processed = df.copy()
        
        # Apply all feature engineering steps
        df_processed = self.create_age_groups(df_processed)
        df_processed = self.create_hours_categories(df_processed)
        df_processed = self.create_capital_features(df_processed)
        df_processed = self.create_education_features(df_processed)
        df_processed = self.create_work_features(df_processed)
        df_processed = self.create_family_features(df_processed)
        df_processed = self.create_interaction_features(df_processed)
        
        logger.info(f"Feature engineering completed. New shape: {df_processed.shape}")
        return df_processed
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        train_size = self.data_config.get('train_size', 0.7)
        random_state = self.data_config.get('random_state', 123)
        stratify = self.data_config.get('stratify', True)
        
        stratify_param = y if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Training target distribution: {np.bincount(y_train)}")
        logger.info(f"Validation target distribution: {np.bincount(y_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for feature importance analysis.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary with feature information
        """
        feature_info = {
            'original_features': [],
            'engineered_features': [],
            'feature_types': {},
            'feature_descriptions': {}
        }
        
        # Original features (from dataset)
        original_cols = [
            'age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain',
            'capital-loss', 'hours-per-week'
        ]
        
        feature_info['original_features'] = [col for col in original_cols if col in df.columns]
        
        # Engineered features
        engineered_cols = [
            'age_group', 'hours_category', 'net_capital', 'has_capital_gain',
            'has_capital_loss', 'has_capital_activity', 'capital_gain_category',
            'capital_loss_category', 'education_level', 'high_education',
            'is_government_worker', 'is_self_employed', 'is_professional',
            'is_married', 'is_head_of_household', 'age_education_interaction',
            'hours_age_interaction', 'education_capital_interaction'
        ]
        
        feature_info['engineered_features'] = [col for col in engineered_cols if col in df.columns]
        
        # Feature types
        for col in df.columns:
            if col in ['target']:
                continue
            elif df[col].dtype == 'object' or col.endswith('_category') or col.endswith('_group'):
                feature_info['feature_types'][col] = 'categorical'
            elif col.startswith('is_') or col.startswith('has_'):
                feature_info['feature_types'][col] = 'binary'
            else:
                feature_info['feature_types'][col] = 'numerical'
        
        # Feature descriptions
        feature_info['feature_descriptions'] = {
            'age': 'Age of the individual',
            'workclass': 'Type of employment',
            'fnlwgt': 'Final weight (census sampling weight)',
            'education-num': 'Years of education',
            'marital-status': 'Marital status',
            'occupation': 'Occupation category (combined into 5 groups)',
            'relationship': 'Relationship status in family',
            'race': 'Race/ethnicity',
            'sex': 'Gender',
            'capital-gain': 'Capital gains income',
            'capital-loss': 'Capital losses',
            'hours-per-week': 'Hours worked per week',
            'age_group': 'Age categorized into groups',
            'hours_category': 'Working hours categorized',
            'net_capital': 'Net capital (gains - losses)',
            'has_capital_gain': 'Indicator for any capital gains',
            'has_capital_loss': 'Indicator for any capital losses',
            'has_capital_activity': 'Indicator for any capital activity',
            'capital_gain_category': 'Capital gains amount category',
            'capital_loss_category': 'Capital losses amount category',
            'education_level': 'Education level category',
            'high_education': 'Indicator for higher education (≥13 years)',
            'is_government_worker': 'Indicator for government employment',
            'is_self_employed': 'Indicator for self-employment',
            'is_professional': 'Indicator for professional occupation',
            'is_married': 'Indicator for married status',
            'is_head_of_household': 'Indicator for head of household',
            'age_education_interaction': 'Interaction between age and education',
            'hours_age_interaction': 'Interaction between hours worked and age',
            'education_capital_interaction': 'Interaction between education and capital gains'
        }
        
        return feature_info


def engineer_and_split_data(
    df: pd.DataFrame, 
    config_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureEngineer]:
    """
    Convenience function to engineer features and split data.
    
    Args:
        df: Preprocessed DataFrame
        config_path: Path to configuration file
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, feature_engineer)
    """
    engineer = FeatureEngineer(config_path)
    
    # Apply feature engineering
    df_engineered = engineer.apply_all_feature_engineering(df)
    
    # Prepare features and target
    target_col = engineer.features_config.get('target_column', 'target')
    X = df_engineered.drop(columns=[target_col]).values
    y = df_engineered[target_col].values
    
    # Split data
    X_train, X_val, y_train, y_val = engineer.split_data(X, y)
    
    return X_train, X_val, y_train, y_val, engineer


def save_engineered_data(
    X_train: np.ndarray, 
    X_val: np.ndarray, 
    y_train: np.ndarray, 
    y_val: np.ndarray,
    feature_names: List[str],
    config_path: Optional[str] = None
):
    """
    Save engineered and split data to files.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets
        feature_names: List of feature names
        config_path: Path to configuration file
    """
    config = get_config(config_path)
    processed_path = config.get('data.processed_path', 'data/processed/')
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['target'] = y_val
    
    # Save to CSV
    train_path = Path(processed_path) / 'train_data.csv'
    val_path = Path(processed_path) / 'validation_data.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    logger.info(f"Training data saved to {train_path}")
    logger.info(f"Validation data saved to {val_path}")
    
    # Save feature names
    feature_names_path = Path(processed_path) / 'feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    logger.info(f"Feature names saved to {feature_names_path}")


if __name__ == "__main__":
    # Test feature engineering
    from .load_data import load_and_validate_data
    from .preprocess import preprocess_data
    
    # Load and preprocess data
    df = load_and_validate_data()
    X, y, preprocessor = preprocess_data(df)
    
    # Create DataFrame for feature engineering
    feature_names = preprocessor.get_feature_names()
    df_processed = pd.DataFrame(X, columns=feature_names)
    df_processed['target'] = y
    
    # Apply feature engineering and split
    X_train, X_val, y_train, y_val, engineer = engineer_and_split_data(df_processed)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Feature engineering completed successfully!")
