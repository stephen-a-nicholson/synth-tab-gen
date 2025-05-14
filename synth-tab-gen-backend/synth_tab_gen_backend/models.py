"""This module contains the configuration classes for the model 
training and data generation process."""

from pydantic import BaseModel

class ModelConfig(BaseModel):
    """
    Configuration for the model training and generation process."""
    model_type: str  # "CTGAN", "TVAE", or "GaussianCopula"
    epochs: int = 100
    batch_size: int = 500
    use_gpu: bool = True

class GenerationConfig(BaseModel):
    """
    Configuration for the data generation process."""
    num_rows: int
    include_quality_metrics: bool = True
    output_format: str = "csv"  # "csv", "json", "parquet"