"""Contains the data models for the SynthTabGen API."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelType(str, Enum):
    """Enumeration of the model types supported by SynthTabGen."""
    CTGAN = "CTGAN"
    TVAE = "TVAE"
    GAUSSIAN_COPULA = "GaussianCopula"


class OutputFormat(str, Enum):
    """Enumeration of the output formats supported by SynthTabGen."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    SQL = "sql"


class JobStatus(str, Enum):
    """Enumeration of the job statuses supported by SynthTabGen."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelConfig(BaseModel):
    """Configuration for the model training."""
    model_type: ModelType
    epochs: int = 100
    batch_size: int = 500
    use_gpu: bool = True


class GenerationConfig(BaseModel):
    """Configuration for the data generation."""
    num_rows: int
    include_quality_metrics: bool = True
    output_format: OutputFormat = OutputFormat.CSV
    missing_values_percent: float = 0
    duplicates_percent: float = 0
    outliers_percent: float = 0


class DatasetInfo(BaseModel):
    """Information about the dataset."""
    dataset_id: str
    filename: str
    columns: List[str]
    rows: int
    created_at: str


class ModelInfo(BaseModel):
    """Information about the model."""
    model_id: str
    model_type: ModelType
    dataset_id: str
    created_at: str
    config: Dict[str, Any]


class Job(BaseModel):
    """Information about a job."""
    job_id: str
    status: JobStatus
    progress: float = 0
    created_at: str
    task_type: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    """Standard API response format."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
