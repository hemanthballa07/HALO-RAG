"""
Dependency injection for FastAPI endpoints.
"""

from fastapi import Depends, Request
from typing import Annotated

from ..core.models import ModelLoader, get_model_loader
from ..core.logging import set_request_id


def get_model_loader_dep() -> ModelLoader:
    """
    Dependency to inject model loader.
    
    Returns:
        ModelLoader instance
    """
    return get_model_loader()


def get_request_id(request: Request) -> str:
    """
    Generate and set request ID for tracking.
    
    Args:
        request: FastAPI request object
    
    Returns:
        Request ID
    """
    # Check if request ID already exists in headers
    request_id = request.headers.get("X-Request-ID")
    
    # Generate new ID if not provided
    request_id = set_request_id(request_id)
    
    return request_id


# Type aliases for cleaner endpoint signatures
ModelLoaderDep = Annotated[ModelLoader, Depends(get_model_loader_dep)]
RequestIDDep = Annotated[str, Depends(get_request_id)]
