from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import DatasetStatus
from app.schemas import (
    CreateYoloDatasetRequest,
    CreateYoloDatasetResponse,
    DatasetStatusSchema,
    YoloDatasetListItem,
    YoloDatasetListResponse,
    YoloDatasetPreviewPair,
    YoloDatasetResultResponse,
    YoloDatasetResultSummary,
    YoloDatasetStatusResponse,
)
from app.services.renders import (
    create_yolo_dataset_job,
    get_yolo_dataset_job,
    get_yolo_dataset_result,
    list_yolo_dataset_jobs,
)
from app.storage import storage

router = APIRouter()


def _build_preview_pairs(summary: dict) -> list[YoloDatasetPreviewPair]:
    preview_pairs_raw = summary.get('preview_pairs') or []
    preview_pairs: list[YoloDatasetPreviewPair] = []
    for pair in preview_pairs_raw:
        if not isinstance(pair, dict):
            continue
        image_key = pair.get('image_key')
        bbox_key = pair.get('bbox_key')
        mask_key = pair.get('mask_key')
        if not (isinstance(image_key, str) and isinstance(bbox_key, str) and isinstance(mask_key, str)):
            continue
        preview_pairs.append(
            YoloDatasetPreviewPair(
                image_url=storage.presign_get(image_key),
                bbox_url=storage.presign_get(bbox_key),
                mask_url=storage.presign_get(mask_key),
            )
        )
    return preview_pairs


@router.post('/datasets/yolo', response_model=CreateYoloDatasetResponse)
def create_yolo_dataset_endpoint(payload: CreateYoloDatasetRequest, db: Session = Depends(get_db)) -> CreateYoloDatasetResponse:
    job = create_yolo_dataset_job(db, payload)
    return CreateYoloDatasetResponse(dataset_job_id=job.id, status=job.status)


@router.get('/datasets/yolo', response_model=YoloDatasetListResponse)
def list_yolo_datasets_endpoint(
    limit: int = Query(50, ge=1, le=200),
    status: DatasetStatusSchema | None = Query(None),
    db: Session = Depends(get_db),
) -> YoloDatasetListResponse:
    jobs = list_yolo_dataset_jobs(db, limit=limit, status_filter=DatasetStatus(status.value) if status else None)
    items: list[YoloDatasetListItem] = []
    for job in jobs:
        config = job.config or {}
        summary = job.summary or {}
        preview_keys = summary.get('preview_image_keys') or []
        preview_image_urls = [storage.presign_get(key) for key in preview_keys if isinstance(key, str)]
        preview_pairs = _build_preview_pairs(summary)
        items.append(
            YoloDatasetListItem(
                dataset_job_id=job.id,
                status=DatasetStatusSchema(job.status.value),
                progress=job.progress,
                count=config.get('count'),
                width=config.get('width'),
                height=config.get('height'),
                split_train_count=config.get('split_train_count'),
                split_val_count=config.get('split_val_count'),
                images_total=summary.get('images_total'),
                empty_labels_count=summary.get('empty_labels_count'),
                class_names=summary.get('class_names') or [],
                preview_image_urls=preview_image_urls,
                preview_pairs=preview_pairs,
                started_at=job.started_at,
                updated_at=job.updated_at,
                error_message=job.error_message,
            )
        )
    return YoloDatasetListResponse(items=items)


@router.get('/datasets/yolo/{dataset_job_id}', response_model=YoloDatasetStatusResponse)
def get_yolo_dataset_status_endpoint(dataset_job_id: int, db: Session = Depends(get_db)) -> YoloDatasetStatusResponse:
    job = get_yolo_dataset_job(db, dataset_job_id)
    return YoloDatasetStatusResponse(
        dataset_job_id=job.id,
        status=job.status,
        progress=job.progress,
        started_at=job.started_at,
        updated_at=job.updated_at,
        error_code=job.error_code,
        error_message=job.error_message,
    )


@router.get('/datasets/yolo/{dataset_job_id}/result', response_model=YoloDatasetResultResponse)
def get_yolo_dataset_result_endpoint(dataset_job_id: int, db: Session = Depends(get_db)) -> YoloDatasetResultResponse:
    zip_url, summary = get_yolo_dataset_result(db, dataset_job_id)
    preview_keys = summary.get('preview_image_keys') or []
    preview_urls = [storage.presign_get(key) for key in preview_keys if isinstance(key, str)]
    summary_payload = {**summary, 'preview_image_urls': preview_urls}
    return YoloDatasetResultResponse(
        zip_url=zip_url,
        summary=YoloDatasetResultSummary(**summary_payload),
    )
