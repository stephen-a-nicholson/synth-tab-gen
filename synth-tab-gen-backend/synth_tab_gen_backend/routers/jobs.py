"""Contains functions to handle job-related operations"""

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from synth_tab_gen_backend import storage
from synth_tab_gen_backend.models import (
    APIResponse,
    GenerationConfig,
    OutputFormat,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=APIResponse)
def get_job_status(job_id: str):
    """Get the status of a job"""
    job = storage.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "success": True,
        "message": "Job status retrieved successfully",
        "data": job,
    }


async def generate_data_task(
    job_id: str, model_id: str, config: GenerationConfig
):
    """Background task to generate synthetic data"""
    try:
        # Update job status
        storage.update_job(job_id, {"status": "running"})

        # Get model
        model = storage.get_model_object(model_id)
        if model is None:
            raise Exception("Model not found")

        # Generate synthetic data
        synthetic_data = model.sample(config.num_rows)

        # Apply data quality transformations if specified
        if config.missing_values_percent > 0:
            # Introduce missing values
            pass

        if config.duplicates_percent > 0:
            # Introduce duplicate records
            pass

        if config.outliers_percent > 0:
            # Introduce outliers
            pass

        # Store generated dataset
        dataset_id = storage.store_dataset(
            filename=f"synthetic_data_{model_id}.csv", df=synthetic_data
        )

        # Update job with success
        storage.update_job(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "result": {"dataset_id": dataset_id},
            },
        )

    except Exception as e:
        # Update job with error
        storage.update_job(job_id, {"status": "failed", "error": str(e)})


@router.post("/generate-data/{model_id}", response_model=APIResponse)
async def generate_data(
    model_id: str, config: GenerationConfig, background_tasks: BackgroundTasks
):
    """Generate synthetic data using a trained model"""
    # Check if model exists
    model = storage.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create a new job
    job_id = storage.create_job(task_type="generate_data")

    # Start generation in background
    background_tasks.add_task(
        generate_data_task, job_id=job_id, model_id=model_id, config=config
    )

    return {
        "success": True,
        "message": "Data generation started",
        "data": {"job_id": job_id},
    }


@router.get("/data/{dataset_id}", response_model=APIResponse)
def get_generated_data(
    dataset_id: str, format: OutputFormat = OutputFormat.CSV
):
    """Get generated synthetic data in the specified format"""
    dataset = storage.get_dataset(dataset_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = dataset["data"]
    preview_data = df.head(5).to_dict(orient="records")

    return {
        "success": True,
        "message": "Data preview retrieved successfully",
        "data": {
            "dataset_id": dataset_id,
            "rows": len(df),
            "preview": preview_data,
        },
    }
