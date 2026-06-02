from fastapi import APIRouter, Response

from app.core.metrics import get_content_type, get_metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_content_type(),
    )
