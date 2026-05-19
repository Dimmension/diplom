from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas import CreateRenderRequest, CreateRenderResponse, RenderResultResponse, RenderStatusResponse
from app.services.renders import (
    create_render_job,
    get_additional_render_urls,
    get_render_job,
    get_render_result,
)

router = APIRouter()


@router.post('/renders', response_model=CreateRenderResponse)
def create_render_endpoint(payload: CreateRenderRequest, db: Session = Depends(get_db)) -> CreateRenderResponse:
    render_job = create_render_job(db, payload.scene_id, payload.scene_config_snapshot)
    return CreateRenderResponse(render_job_id=render_job.id, status=render_job.status)


@router.get('/renders/{render_job_id}', response_model=RenderStatusResponse)
def get_render_status_endpoint(render_job_id: int, db: Session = Depends(get_db)) -> RenderStatusResponse:
    render_job = get_render_job(db, render_job_id)
    return RenderStatusResponse(
        render_job_id=render_job.id,
        status=render_job.status,
        progress=render_job.progress,
        started_at=render_job.started_at,
        updated_at=render_job.updated_at,
        error_code=render_job.error_code,
        error_message=render_job.error_message,
    )


@router.get('/renders/{render_job_id}/result', response_model=RenderResultResponse)
def get_render_result_endpoint(render_job_id: int, db: Session = Depends(get_db)) -> RenderResultResponse:
    render_job, png_url = get_render_result(db, render_job_id)
    mask_url, bbox_url = get_additional_render_urls(render_job_id)
    return RenderResultResponse(
        png_url=png_url,
        mask_url=mask_url,
        bbox_url=bbox_url,
        scene_config_used=render_job.scene_config_used,
        scene_config_suggested=render_job.scene_config_suggested,
    )
