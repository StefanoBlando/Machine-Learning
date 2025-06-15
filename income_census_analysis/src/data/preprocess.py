"""
Data preprocessing utilities for the Income Census Analysis project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any, Optional
import logging
import joblib
from pathlib import Path

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for comprehensive data preprocessing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataPreprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.features_config = self.config.features_config
        self.data_config = self.config.data_config
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.feature_names = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values and outliers.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        logger.info("Starting data cleaning...")
        
        # Remove leading/trailing whitespace from string columns
        string_cols = df_clean.select_dtypes(include=['object']).columns
        for col in string_cols:
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        logger.info(f"Missing values before cleaning: {missing_before}")
        
        # For categorical columns with missing values, fill with mode
        categorical_cols = ['workclass', 'occupation', 'native-country']
        for col in categorical_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        # Remove rows with missing target values
        target_col = self.features_config.get('target_column', 'target')
        if target_col in df_clean.columns:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=[target_col])
            dropped_rows = initial_rows - len(df_clean)
            if dropped_rows > 0:
                logger.info(f"Removed {dropped_rows} rows with missing target values")
        
        # Handle outliers in numerical columns
        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                         'capital-loss', 'hours-per-week']
        
        for col in numerical_cols:
            if col in df_clean.columns:
                # Remove extreme outliers using IQR method
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_count = outliers.sum()
                
                if outliers_count > 0:
                    # Cap outliers instead of removing them
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    logger.info(f"Capped {outliers_count} outliers in {col}")
        
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Missing values after cleaning: {missing_after}")
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        
        return df_clean
    
    def drop_specified_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns as specified in project requirements.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with specified columns removed
        """
        columns_to_drop = self.features_config.get('drop_columns', [])
        
        df_processed = df.copy()
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        
        if existing_cols_to_drop:
            df_processed = df_processed.drop(columns=existing_cols_to_drop)
            logger.info(f"Dropped columns: {existing_cols_to_drop}")
        
        return df_processed
    
    def combine_occupation_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine occupation categories into 5 main groups as specified.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with combined occupation categories
        """
        df_processed = df.copy()
        
        if 'occupation' not in df_processed.columns:
            logger.warning("Occupation column not found, skipping occupation grouping")
            return df_processed
        
        # Get occupation mapping from config
        occupation_mapping = self.features_config.get('occupation_mapping', {})
        
        # Create reverse mapping
        occupation_map = {}
        for category, occupations in occupation_mapping.items():
            for occupation in occupations:
                occupation_map[occupation] = category
        
        # Apply mapping
        df_processed['occupation'] = df_processed['occupation'].map(occupation_map).fillna('Other')
        
        logger.info("Occupation categories combined into 5 groups:")
        logger.info(df_processed['occupation'].value_counts().to_dict())
        
        return df_processed
    
    def encode_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode target variable to binary format.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with encoded target
        """
        df_processed = df.copy()
        target_col = self.features_config.get('target_column', 'target')
        
        if target_col in df_processed.columns:
            # Convert target to binary: '>50K' -> 1, '<=50K' -> 0
            df_processed[target_col] = (df_processed[target_col] == '>50K').astype(int)
            logger.info("Target variable encoded: '>50K' -> 1, '<=50K' -> 0")
        
        return df_processed
    
    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline for numerical and categorical features.
        
        Args:
            df: DataFrame to analyze for feature types
            
        Returns:
            Fitted ColumnTransformer
        """
        # Get feature columns (excluding target)
        target_col = self.features_config.get('target_column', 'target')
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Identify numerical and categorical columns
        numerical_cols = self.features_config.get('numerical_columns', [])
        categorical_cols = self.features_config.get('categorical_columns', [])
        
        # Filter to only include existing columns
        numerical_cols = [col for col in numerical_cols if col in feature_cols]
        categorical_cols = [col for col in categorical_cols if col in feature_cols]
        
        logger.info(f"Numerical columns: {numerical_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        # Create preprocessing steps
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'  # Keep other columns as-is
        )
        
        return preprocessor
    
    def fit_transform_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing pipeline and transform features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Tuple of (X_transformed, y)
        """
        target_col = self.features_config.get('target_column', 'target')
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Create and fit preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline(df)
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Store feature names for later use
        self._create_feature_names(X)
        
        logger.info(f"Features transformed. Shape: {X_transformed.shape}")
        return X_transformed, y
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessing pipeline.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed features array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessing pipeline not fitted. Call fit_transform_features first.")
        
        target_col = self.features_config.get('target_column', 'target')
        X = df.drop(columns=[target_col], errors='ignore')
        
        X_transformed = self.preprocessor.transform(X)
        return X_transformed
    
    def _create_feature_names(self, X: pd.DataFrame):
        """
        Create feature names for the transformed features.
        
        Args:
            X: Original features DataFrame
        """
        feature_names = []
        
        # Get transformers
        transformers = self.preprocessor.transformers_
        
        for name, transformer, columns in transformers:
            if name == 'num':
                # Numerical features keep their original names
                feature_names.extend(columns)
            elif name == 'cat':
                # Categorical features get one-hot encoded names
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_names = transformer.get_feature_names_out(columns)
                    feature_names.extend(cat_names)
                else:
                    # Fallback for older sklearn versions
                    for col in columns:
                        unique_vals = X[col].unique()
                        for val in unique_vals[1:]:  # Skip first category (dropped)
                            feature_names.append(f"{col}_{val}")
            elif name == 'remainder':
                # Passthrough columns
                feature_names.extend(columns)
        
        self.feature_names = feature_names
        logger.info(f"Created {len(feature_names)} feature names")
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if self.feature_names is None:
            logger.warning("Feature names not available. Run fit_transform_features first.")
            return []
        return self.feature_names
    
    def save_preprocessor(self, filepath: Optional[str] = None):
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if filepath is None:
            output_path = Path(self.config.get('output.scalers_path', 'models/model_artifacts/'))
            filepath = output_path / 'preprocessor.pkl'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'config': {
                'numerical_columns': self.features_config.get('numerical_columns', []),
                'categorical_columns': self.features_config.get('categorical_columns', []),
                'target_column': self.features_config.get('target_column', 'target')
            }
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor
        """
        preprocessor_data = joblib.load(filepath)
        self.preprocessor = preprocessor_data['preprocessor']
        self.feature_names = preprocessor_data['feature_names']
        logger.info(f"Preprocessor loaded from {filepath}")


def preprocess_data(df: pd.DataFrame, config_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to preprocess census data.
    
    Args:
        df: Raw DataFrame
        config_path: Path to configuration file
        
    Returns:
        Tuple of (X_transformed, y)
    """
    preprocessor = DataPreprocessor(config_path)
    
    # Apply preprocessing steps
    df_clean = preprocessor.clean_data(df)
    df_processed = preprocessor.drop_specified_columns(df_clean)
    df_processed = preprocessor.combine_occupation_categories(df_processed)
    df_processed = preprocessor.encode_target_variable(df_processed)
    
    # Transform features
    X_transformed, y = preprocessor.fit_transform_features(df_processed)
    
    return X_transformed, y, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    from .load_data import load_and_validate_data
    
    df = load_and_validate_data()
    X, y, preprocessor = preprocess_data(df)
    
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Feature names: {len(preprocessor.get_feature_names())}")
