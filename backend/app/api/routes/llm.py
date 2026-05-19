from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas import (
    CreateLlmEnhanceDatasetRequest,
    CreateLlmEnhanceDatasetResponse,
    LlmEnhanceDatasetResultResponse,
    LlmEnhanceDatasetStatusResponse,
    LlmEnhancePreviewItem,
    LlmEnhancePreviewRequest,
    LlmEnhancePreviewResponse,
)
from app.services.llm_enhancer import (
    create_llm_enhance_job,
    enhance_dataset_samples_with_langgraph,
    get_llm_enhance_job,
    get_llm_enhance_job_result,
)

router = APIRouter()


@router.post('/llm/enhance-preview', response_model=LlmEnhancePreviewResponse)
def llm_enhance_preview_endpoint(payload: LlmEnhancePreviewRequest, db: Session = Depends(get_db)) -> LlmEnhancePreviewResponse:
    items_raw = enhance_dataset_samples_with_langgraph(
        db=db,
        dataset_job_id=payload.dataset_job_id,
        sample_count=payload.sample_count,
        llm_model=payload.llm_model,
    )
    items = [LlmEnhancePreviewItem(**item) for item in items_raw]
    return LlmEnhancePreviewResponse(dataset_job_id=payload.dataset_job_id, items=items)


@router.post('/llm/enhance-dataset', response_model=CreateLlmEnhanceDatasetResponse)
def create_llm_enhance_dataset_endpoint(payload: CreateLlmEnhanceDatasetRequest, db: Session = Depends(get_db)) -> CreateLlmEnhanceDatasetResponse:
    job = create_llm_enhance_job(db=db, dataset_job_id=payload.dataset_job_id, llm_model=payload.llm_model)
    return CreateLlmEnhanceDatasetResponse(enhance_job_id=job.id, status=job.status)


@router.get('/llm/enhance-dataset/{enhance_job_id}', response_model=LlmEnhanceDatasetStatusResponse)
def get_llm_enhance_dataset_status_endpoint(enhance_job_id: int, db: Session = Depends(get_db)) -> LlmEnhanceDatasetStatusResponse:
    job = get_llm_enhance_job(db=db, enhance_job_id=enhance_job_id)
    return LlmEnhanceDatasetStatusResponse(
        enhance_job_id=job.id,
        dataset_job_id=job.dataset_job_id,
        status=job.status,
        progress=job.progress,
        started_at=job.started_at,
        updated_at=job.updated_at,
        error_message=job.error_message,
    )


@router.get('/llm/enhance-dataset/{enhance_job_id}/result', response_model=LlmEnhanceDatasetResultResponse)
def get_llm_enhance_dataset_result_endpoint(enhance_job_id: int, db: Session = Depends(get_db)) -> LlmEnhanceDatasetResultResponse:
    payload = get_llm_enhance_job_result(db=db, enhance_job_id=enhance_job_id)
    return LlmEnhanceDatasetResultResponse(
        enhance_job_id=payload['enhance_job_id'],
        dataset_job_id=payload['dataset_job_id'],
        status=payload['status'],
        processed_images=payload['processed_images'],
        total_images=payload['total_images'],
        sample_items=[LlmEnhancePreviewItem(**item) for item in payload['sample_items']],
    )
