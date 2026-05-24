from fastapi import APIRouter, Response

from app.core.metrics import get_metrics, get_content_type

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_content_type(),
    )
