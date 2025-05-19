from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from typing import Dict, Any

def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with additional documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AGARFARM API",
        version="1.0.0",
        description="""
        API for controlling and monitoring the AGARFARM greenhouse simulation.
        
        ## Features
        
        * Real-time simulation control
        * Manual and automated control modes
        * City-specific configurations
        * Batch simulation capabilities
        * WebSocket support for live updates
        
        ## Authentication
        
        This API does not require authentication for local development.
        For production deployment, please implement appropriate security measures.
        
        ## Error Handling
        
        All endpoints return standardized error responses:
        
        ```json
        {
            "status": "error",
            "detail": "Error message",
            "data": {
                "additional": "error details"
            }
        }
        ```
        
        ## WebSocket
        
        The `/ws/snapshot` endpoint provides real-time simulation updates.
        Connect to this endpoint to receive live simulation data.
        """,
        routes=app.routes,
    )

    # Add security schemes if needed
    # openapi_schema["components"]["securitySchemes"] = {
    #     "bearerAuth": {
    #         "type": "http",
    #         "scheme": "bearer",
    #         "bearerFormat": "JWT",
    #     }
    # }

    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "simulation",
            "description": "Simulation control and monitoring endpoints",
            "externalDocs": {
                "description": "AGARFARM Documentation",
                "url": "https://github.com/yourusername/agarfarm",
            },
        }
    ]

    # Add operation metadata
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if "tags" in operation and "simulation" in operation["tags"]:
                operation["responses"].update({
                    "400": {
                        "description": "Bad Request",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "error",
                                    "detail": "Invalid request parameters",
                                    "data": {"field": "error details"}
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Not Found",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "error",
                                    "detail": "Resource not found"
                                }
                            }
                        }
                    },
                    "500": {
                        "description": "Internal Server Error",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "error",
                                    "detail": "An unexpected error occurred"
                                }
                            }
                        }
                    }
                })

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# --- API Documentation Constants ---
API_TITLE = "AGARFARM API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API for controlling and monitoring the AGARFARM greenhouse simulation.

## Features

* Real-time simulation control
* Manual and automated control modes
* City-specific configurations
* Batch simulation capabilities
* WebSocket support for live updates

## Authentication

This API does not require authentication for local development.
For production deployment, please implement appropriate security measures.

## Error Handling

All endpoints return standardized error responses:

```json
{
    "status": "error",
    "detail": "Error message",
    "data": {
        "additional": "error details"
    }
}
```

## WebSocket

The `/ws/snapshot` endpoint provides real-time simulation updates.
Connect to this endpoint to receive live simulation data.
"""

# --- API Tags ---
API_TAGS = [
    {
        "name": "simulation",
        "description": "Simulation control and monitoring endpoints",
        "externalDocs": {
            "description": "AGARFARM Documentation",
            "url": "https://github.com/yourusername/agarfarm",
        },
    }
]

# --- API Examples ---
API_EXAMPLES = {
    "snapshot": {
        "status": "success",
        "data": {
            "temp_c": 25.5,
            "rh": 65.0,
            "sm": 45.0,
            "vpd": 1.2,
            "rain_tank_l": 80.0,
            "current_datetime_iso": "2025-07-01T06:00:00",
            "outside_temp": 22.0,
            "outside_rh": 70.0,
            "actuators": {
                "fan": False,
                "ac": False,
                "vent": False,
                "irrigation": False
            },
            "sim_is_running": True,
            "config": {
                "control_mode": "manual",
                "city": "oslo"
            }
        }
    },
    "error": {
        "status": "error",
        "detail": "Invalid control mode",
        "data": {
            "valid_modes": ["off", "manual", "ml_normal", "ml_eco", "baseline_normal", "baseline_eco"]
        }
    }
} 