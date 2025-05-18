"""Contains endpoints for working with datasets"""

from io import StringIO

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from synth_tab_gen_backend import storage
from synth_tab_gen_backend.models import APIResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=APIResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file (CSV) and store it"""
    if not file.filename.endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Only CSV files are supported",
            },
        )

    try:
        # Read file content
        contents = await file.read()

        # Parse CSV
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Store dataset
        dataset_id = storage.store_dataset(file.filename, df)

        # Get dataset info
        dataset = storage.get_dataset(dataset_id)

        return {
            "success": True,
            "message": "Dataset uploaded successfully",
            "data": {
                "dataset_id": dataset_id,
                "filename": dataset["filename"],
                "columns": dataset["columns"],
                "rows": dataset["rows"],
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error uploading dataset: {str(e)}",
            },
        )


@router.get("/{dataset_id}", response_model=APIResponse)
def get_dataset_info(dataset_id: str):
    """Get information about a specific dataset"""
    dataset = storage.get_dataset(dataset_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "success": True,
        "message": "Dataset retrieved successfully",
        "data": {
            "dataset_id": dataset["dataset_id"],
            "filename": dataset["filename"],
            "columns": dataset["columns"],
            "rows": dataset["rows"],
            "created_at": dataset["created_at"],
        },
    }


@router.get("/", response_model=APIResponse)
def list_datasets():
    """List all available datasets"""
    dataset_list = []

    for _, dataset in storage.datasets.items():
        dataset_list.append(
            {
                "dataset_id": dataset["dataset_id"],
                "filename": dataset["filename"],
                "columns": len(dataset["columns"]),
                "rows": dataset["rows"],
                "created_at": dataset["created_at"],
            }
        )

    return {
        "success": True,
        "message": "Datasets retrieved successfully",
        "data": {"datasets": dataset_list},
    }
