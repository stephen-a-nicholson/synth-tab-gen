"""This module defines the FastAPI application and includes the routers"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch

from synth_tab_gen_backend.routers import datasets, models, jobs

app = FastAPI(title="SynthTabGen API")


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    This middleware adds security headers to the response.

    The following headers are added:

    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security: max-age=31536000; includeSubDomains

    These headers are added to help prevent common web attacks such as
    cross-site scripting (XSS) and clickjacking.

    The Strict-Transport-Security header is added to ensure that the API
    is only accessed over HTTPS.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(jobs.router)


@app.get("/")
def read_root():
    """Root endpoint that returns API information and GPU status"""
    return {
        "name": "SynthTabGen API",
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }