from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging

# --- Custom Exceptions ---
class SimulationError(Exception):
    """Base exception for simulation-related errors."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ConfigurationError(SimulationError):
    """Raised when there's an error in simulation configuration."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)

class ControlError(SimulationError):
    """Raised when there's an error in control operations."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)

class ResourceError(SimulationError):
    """Raised when there's an error accessing or managing resources."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=404, details=details)

# --- Error Handlers ---
async def simulation_error_handler(request: Request, exc: SimulationError) -> JSONResponse:
    """Handle simulation-related errors."""
    logger = logging.getLogger("error_handler")
    logger.error(f"Simulation error: {exc.message}", exc_info=True, extra=exc.details)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.message,
            "data": exc.details
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    logger = logging.getLogger("error_handler")
    logger.error(f"HTTP error: {exc.detail}", exc_info=True)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail
        }
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger = logging.getLogger("error_handler")
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )

# --- Error Response Models ---
class ErrorResponse:
    """Standard error response format."""
    def __init__(
        self,
        status: str = "error",
        detail: str = "An error occurred",
        data: Optional[Dict[str, Any]] = None
    ):
        self.status = status
        self.detail = detail
        self.data = data or {}

    def dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "data": self.data
        }

# --- Helper Functions ---
def handle_simulation_error(func):
    """Decorator to handle simulation errors in route handlers."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except SimulationError as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=e.message
            )
        except Exception as e:
            logger = logging.getLogger("error_handler")
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {str(e)}"
            )
    return wrapper 