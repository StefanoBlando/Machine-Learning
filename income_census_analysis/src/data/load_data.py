"""
Data loading utilities for the Income Census Analysis project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading and initial processing of the census data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.data_config = self.config.data_config
        
        # Column names for the Adult dataset
        self.column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
        ]
    
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw census data from CSV file.
        
        Args:
            file_path: Path to data file. If None, uses config path.
            
        Returns:
            Raw DataFrame
        """
        if file_path is None:
            file_path = self.data_config.get('raw_path', 'data/raw/data.csv')
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load data with specified column names
            df = pd.read_csv(
                file_path,
                names=self.column_names,
                skipinitialspace=True,  # Remove leading/trailing whitespace
                na_values=['?', ' ?', '?']  # Treat '?' as missing values
            )
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicates': df.duplicated().sum()
        }
        
        # Get unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        info['unique_values'] = {}
        for col in categorical_cols:
            info['unique_values'][col] = df[col].nunique()
        
        # Get statistics for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        info['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        return info
    
    def print_data_summary(self, df: pd.DataFrame):
        """
        Print a comprehensive summary of the dataset.
        
        Args:
            df: DataFrame to summarize
        """
        info = self.get_data_info(df)
        
        print("="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage'] / 1024**2:.2f} MB")
        print(f"Duplicates: {info['duplicates']}")
        
        print("\n" + "="*50)
        print("COLUMN INFORMATION")
        print("="*50)
        for col in df.columns:
            dtype = info['dtypes'][col]
            missing = info['missing_values'][col]
            missing_pct = info['missing_percentage'][col]
            
            if col in info['unique_values']:
                unique = info['unique_values'][col]
                print(f"{col:15} | {dtype:8} | Missing: {missing:4} ({missing_pct:5.1f}%) | Unique: {unique:4}")
            else:
                print(f"{col:15} | {dtype:8} | Missing: {missing:4} ({missing_pct:5.1f}%)")
        
        print("\n" + "="*50)
        print("TARGET VARIABLE DISTRIBUTION")
        print("="*50)
        target_col = self.config.get('features.target_column', 'target')
        if target_col in df.columns:
            target_counts = df[target_col].value_counts()
            target_pct = df[target_col].value_counts(normalize=True) * 100
            
            for value in target_counts.index:
                count = target_counts[value]
                percentage = target_pct[value]
                print(f"{value:10} | Count: {count:5} | Percentage: {percentage:5.1f}%")
        
        print("\n" + "="*50)
        print("MISSING VALUES SUMMARY")
        print("="*50)
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': [info['missing_values'][col] for col in df.columns],
            'Missing %': [info['missing_percentage'][col] for col in df.columns]
        }).sort_values('Missing Count', ascending=False)
        
        print(missing_summary.to_string(index=False))
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data for expected structure and content.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if all expected columns are present
            expected_cols = set(self.column_names)
            actual_cols = set(df.columns)
            
            if not expected_cols.issubset(actual_cols):
                missing_cols = expected_cols - actual_cols
                logger.error(f"Missing columns: {missing_cols}")
                return False
            
            # Check target variable values
            target_col = self.config.get('features.target_column', 'target')
            expected_targets = {'>50K', '<=50K'}
            actual_targets = set(df[target_col].dropna().unique())
            
            if not expected_targets.issubset(actual_targets):
                logger.error(f"Unexpected target values: {actual_targets}")
                return False
            
            # Check for reasonable data ranges
            if df['age'].min() < 0 or df['age'].max() > 120:
                logger.warning("Age values outside expected range (0-120)")
            
            if df['hours-per-week'].min() < 0 or df['hours-per-week'].max() > 168:
                logger.warning("Hours per week outside expected range (0-168)")
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of the output file
        """
        output_path = Path(self.data_config.get('processed_path', 'data/processed/')) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")


def load_and_validate_data(config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load and validate census data.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated DataFrame
    """
    loader = DataLoader(config_path)
    df = loader.load_raw_data()
    
    if not loader.validate_data(df):
        raise ValueError("Data validation failed")
    
    return df


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    df = loader.load_raw_data()
    loader.print_data_summary(df)
    
    if loader.validate_data(df):
        print("\n✅ Data validation passed!")
    else:
        print("\n❌ Data validation failed!")
