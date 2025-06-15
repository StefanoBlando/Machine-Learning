"""
Configuration utilities for the Income Census Analysis project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class Config:
    """Configuration class for loading and managing project settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create logs directory if it doesn't exist
        log_file = log_config.get('file', 'logs/income_census_analysis.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            "data/raw",
            "data/processed",
            "data/external",
            "models/trained_models",
            "models/model_artifacts",
            "results/figures",
            "results/reports",
            "results/metrics",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.raw_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.raw_path')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    @property
    def features_config(self) -> Dict[str, Any]:
        """Get features configuration."""
        return self.config.get('features', {})
    
    @property
    def models_config(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self.config.get('models', {})
    
    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    @property
    def visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config.get('visualization', {})
    
    @property
    def output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})


# Global configuration instance
_config = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def setup_environment():
    """Setup environment variables and random seeds."""
    import numpy as np
    import random
    
    config = get_config()
    
    # Set random seeds for reproducibility
    seeds = config.get('random_seeds', {})
    
    np.random.seed(seeds.get('numpy', 123))
    random.seed(seeds.get('python', 123))
    
    # Set environment variables for parallel processing
    os.environ['PYTHONHASHSEED'] = str(seeds.get('python', 123))
    
    # Set number of threads for various libraries
    n_jobs = config.get('parallel.n_jobs', -1)
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    
    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    os.environ['MKL_NUM_THREADS'] = str(n_jobs)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_jobs)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Data path: {config.get('data.raw_path')}")
    print(f"Random state: {config.get('data.random_state')}")
    print(f"Models: {list(config.models_config.keys())}")
