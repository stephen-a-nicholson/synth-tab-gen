"""Contains the data models for the SynthTabGen API."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    CTGAN = "CTGAN"
    TVAE = "TVAE"
    GAUSSIAN_COPULA = "GaussianCopula"


class OutputFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    SQL = "sql"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelConfig(BaseModel):
    model_type: ModelType
    epochs: int = 100
    batch_size: int = 500
    use_gpu: bool = True


class GenerationConfig(BaseModel):
    num_rows: int
    include_quality_metrics: bool = True
    output_format: OutputFormat = OutputFormat.CSV
    missing_values_percent: float = 0
    duplicates_percent: float = 0
    outliers_percent: float = 0


class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    columns: List[str]
    rows: int
    created_at: str


class ModelInfo(BaseModel):
    model_id: str
    model_type: ModelType
    dataset_id: str
    created_at: str
    config: Dict[str, Any]


class Job(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0
    created_at: str
    task_type: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None