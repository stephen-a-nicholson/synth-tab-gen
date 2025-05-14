""" Contains endpoints for managing models """

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from synth_tab_gen_backend.models import ModelConfig, APIResponse, ModelType
from synth_tab_gen_backend import storage
from sdv.tabular import CTGAN, TVAE, GaussianCopula
import torch

router = APIRouter(prefix="/models", tags=["models"])


async def train_model_task(job_id: str, dataset_id: str, config: ModelConfig):
    """Background task to train a model"""
    try:
        # Update job status
        storage.update_job(job_id, {"status": "running"})
        
        # Get dataset
        df = storage.get_dataset_data(dataset_id)
        if df is None:
            raise Exception("Dataset not found")
        
        # Initialize model based on type
        if config.model_type == ModelType.CTGAN:
            model = CTGAN(epochs=config.epochs, batch_size=config.batch_size, cuda=config.use_gpu)
        elif config.model_type == ModelType.TVAE:
            model = TVAE(epochs=config.epochs, batch_size=config.batch_size, cuda=config.use_gpu)
        else:
            model = GaussianCopula()
        
        # Train model
        model.fit(df)
        
        # Store trained model
        model_id = storage.store_model(
            model_type=config.model_type,
            model_obj=model,
            dataset_id=dataset_id,
            config=config.dict()
        )
        
        # Update job with success
        storage.update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "result": {"model_id": model_id}
        })
        
    except Exception as e:
        # Update job with error
        storage.update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })


@router.post("/train/{dataset_id}", response_model=APIResponse)
async def train_model(
    dataset_id: str, 
    config: ModelConfig, 
    background_tasks: BackgroundTasks
):
    """Train a model using the specified dataset and configuration"""
    # Check if dataset exists
    dataset = storage.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create a new job
    job_id = storage.create_job(task_type="train_model")
    
    # Start training in background
    background_tasks.add_task(
        train_model_task,
        job_id=job_id,
        dataset_id=dataset_id,
        config=config
    )
    
    return {
        "success": True,
        "message": "Model training started",
        "data": {
            "job_id": job_id
        }
    }


@router.get("/{model_id}", response_model=APIResponse)
def get_model_info(model_id: str):
    """Get information about a trained model"""
    model = storage.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    return {
        "success": True,
        "message": "Model retrieved successfully",
        "data": {
            "model_id": model["model_id"],
            "model_type": model["model_type"],
            "dataset_id": model["dataset_id"],
            "created_at": model["created_at"],
            "config": model["config"]
        }
    }


@router.get("/", response_model=APIResponse)
def list_models():
    """List all available models"""
    model_list = []
    
    for model_id, model in storage.models.items():
        model_list.append({
            "model_id": model["model_id"],
            "model_type": model["model_type"],
            "dataset_id": model["dataset_id"],
            "created_at": model["created_at"]
        })
        
    return {
        "success": True,
        "message": "Models retrieved successfully",
        "data": {
            "models": model_list
        }
    }


@router.post("/generate/{model_id}", response_model=APIResponse)
async def generate_data(
    model_id: str,
    config: GenerationConfig,
    background_tasks: BackgroundTasks
):
    """Generate synthetic data using a trained model"""
    # This will be implemented in the jobs router
    pass