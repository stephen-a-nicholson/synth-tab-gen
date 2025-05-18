"""Conain functions to store and retrieve datasets, models, and jobs."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

# In-memory storage (would use a database in production)
datasets = {}
models = {}
jobs = {}


def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().isoformat()


def generate_id():
    """Generate a unique ID"""
    return str(uuid.uuid4())


def store_dataset(filename: str, df: pd.DataFrame) -> str:
    """Store a dataset and return its ID"""
    dataset_id = generate_id()
    datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "filename": filename,
        "data": df,
        "columns": df.columns.tolist(),
        "rows": len(df),
        "created_at": get_timestamp(),
    }
    return dataset_id


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Get a dataset by ID"""
    return datasets.get(dataset_id)


def get_dataset_data(dataset_id: str) -> Optional[pd.DataFrame]:
    """Get the actual dataframe for a dataset"""
    dataset = get_dataset(dataset_id)
    if dataset:
        return dataset["data"]
    return None


# Model operations
def store_model(
    model_type: str, model_obj: Any, dataset_id: str, config: Dict
) -> str:
    """Store a trained model and return its ID"""
    model_id = generate_id()
    models[model_id] = {
        "model_id": model_id,
        "model_type": model_type,
        "model": model_obj,
        "dataset_id": dataset_id,
        "created_at": get_timestamp(),
        "config": config,
    }
    return model_id


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Get a model by ID"""
    return models.get(model_id)


def get_model_object(model_id: str) -> Optional[Any]:
    """Get the actual model object"""
    model = get_model(model_id)
    if model:
        return model["model"]
    return None


def create_job(task_type: str) -> str:
    """Create a new job and return its ID"""
    job_id = generate_id()
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "created_at": get_timestamp(),
        "task_type": task_type,
        "error": None,
        "result": None,
    }
    return job_id


def update_job(job_id: str, updates: Dict[str, Any]) -> bool:
    """Update a job's status and details"""
    if job_id in jobs:
        jobs[job_id].update(updates)
        return True
    return False


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by ID"""
    return jobs.get(job_id)
