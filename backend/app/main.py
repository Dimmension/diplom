from fastapi import Depends, FastAPI, File, Query, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import Base, engine, get_db
from app.models import AssetKind, DatasetStatus
from app.schemas import (
    AssetListResponse,
    AssetKindSchema,
    AssetUploadResponse,
    CreateRenderRequest,
    CreateRenderResponse,
    CreateYoloDatasetRequest,
    CreateYoloDatasetResponse,
    DatasetStatusSchema,
    CreateSceneRequest,
    RenderResultResponse,
    RenderStatusResponse,
    SceneResponse,
    UpdateSceneConfigRequest,
    LlmEnhancePreviewRequest,
    LlmEnhancePreviewResponse,
    LlmEnhancePreviewItem,
    CreateLlmEnhanceDatasetRequest,
    CreateLlmEnhanceDatasetResponse,
    LlmEnhanceDatasetStatusResponse,
    LlmEnhanceDatasetResultResponse,
    YoloDatasetResultResponse,
    YoloDatasetResultSummary,
    YoloDatasetPreviewPair,
    YoloDatasetListItem,
    YoloDatasetListResponse,
    YoloDatasetStatusResponse,
)
from app.services.assets import list_assets, upload_asset
from app.services.llm_enhancer import (
    create_llm_enhance_job,
    enhance_dataset_samples_with_langgraph,
    get_llm_enhance_job,
    get_llm_enhance_job_result,
)
from app.services.renders import (
    create_render_job,
    create_yolo_dataset_job,
    get_additional_render_urls,
    get_render_job,
    get_render_result,
    get_yolo_dataset_job,
    list_yolo_dataset_jobs,
    get_yolo_dataset_result,
)
from app.services.scenes import create_scene, update_scene_config
from app.storage import storage

settings = get_settings()
app = FastAPI(title=settings.app_name)


def _ensure_asset_kind_enum_values() -> None:
    if engine.dialect.name != 'postgresql':
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_enum e
                        JOIN pg_type t ON e.enumtypid = t.oid
                        WHERE t.typname = 'assetkind' AND e.enumlabel = 'skybox'
                    ) THEN
                        ALTER TYPE assetkind ADD VALUE 'skybox';
                    END IF;
                END $$;
                """
            )
        )


@app.on_event('startup')
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_asset_kind_enum_values()
    try:
        storage.client.head_bucket(Bucket=settings.s3_bucket)
    except Exception:  # noqa: BLE001
        storage.client.create_bucket(Bucket=settings.s3_bucket)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get(f'{settings.api_prefix}/assets', response_model=AssetListResponse)
def list_assets_endpoint(
    kind: AssetKindSchema | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> AssetListResponse:
    assets = list_assets(db=db, kind=AssetKind(kind.value) if kind else None, limit=limit)
    items = []
    for asset in assets:
        preview_glb_url = storage.presign_get(asset.preview_glb_key) if asset.kind != AssetKind.skybox else None
        items.append(
            AssetUploadResponse(
                asset_id=asset.id,
                kind=AssetKindSchema(asset.kind.value),
                filename=asset.original_filename,
                original_url=storage.presign_get(asset.original_key),
                preview_glb_url=preview_glb_url,
                detected_format=asset.detected_format,
                size_bytes=asset.size_bytes,
                created_at=asset.created_at,
            )
        )
    return AssetListResponse(items=items)


@app.post(f'{settings.api_prefix}/assets/upload', response_model=AssetUploadResponse)
async def upload_asset_endpoint(
    kind: AssetKindSchema = Query(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AssetUploadResponse:
    asset = await upload_asset(db=db, kind=AssetKind(kind.value), upload=file)
    preview_glb_url = storage.presign_get(asset.preview_glb_key) if asset.kind != AssetKind.skybox else None
    return AssetUploadResponse(
        asset_id=asset.id,
        kind=AssetKindSchema(asset.kind.value),
        filename=asset.original_filename,
        original_url=storage.presign_get(asset.original_key),
        preview_glb_url=preview_glb_url,
        detected_format=asset.detected_format,
        size_bytes=asset.size_bytes,
        created_at=asset.created_at,
    )


@app.post(f'{settings.api_prefix}/scenes', response_model=SceneResponse)
def create_scene_endpoint(payload: CreateSceneRequest, db: Session = Depends(get_db)) -> SceneResponse:
    scene = create_scene(db, payload.object_asset_id, payload.environment_asset_id, payload.skybox_asset_id)
    return SceneResponse(scene_id=scene.id, scene_config=scene.scene_config)


@app.patch(f'{settings.api_prefix}/scenes/{{scene_id}}/config', response_model=SceneResponse)
def update_scene_config_endpoint(scene_id: int, payload: UpdateSceneConfigRequest, db: Session = Depends(get_db)) -> SceneResponse:
    scene = update_scene_config(db, scene_id, payload.scene_config)
    return SceneResponse(scene_id=scene.id, scene_config=scene.scene_config)


@app.post(f'{settings.api_prefix}/renders', response_model=CreateRenderResponse)
def create_render_endpoint(payload: CreateRenderRequest, db: Session = Depends(get_db)) -> CreateRenderResponse:
    render_job = create_render_job(db, payload.scene_id, payload.scene_config_snapshot)
    return CreateRenderResponse(render_job_id=render_job.id, status=render_job.status)


@app.get(f'{settings.api_prefix}/renders/{{render_job_id}}', response_model=RenderStatusResponse)
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


@app.get(f'{settings.api_prefix}/renders/{{render_job_id}}/result', response_model=RenderResultResponse)
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


@app.post(f'{settings.api_prefix}/datasets/yolo', response_model=CreateYoloDatasetResponse)
def create_yolo_dataset_endpoint(payload: CreateYoloDatasetRequest, db: Session = Depends(get_db)) -> CreateYoloDatasetResponse:
    job = create_yolo_dataset_job(db, payload)
    return CreateYoloDatasetResponse(dataset_job_id=job.id, status=job.status)


@app.get(f'{settings.api_prefix}/datasets/yolo', response_model=YoloDatasetListResponse)
def list_yolo_datasets_endpoint(
    limit: int = Query(50, ge=1, le=200),
    status: DatasetStatusSchema | None = Query(None),
    db: Session = Depends(get_db),
) -> YoloDatasetListResponse:
    jobs = list_yolo_dataset_jobs(db, limit=limit, status_filter=DatasetStatus(status.value) if status else None)
    items = []
    for job in jobs:
        config = job.config or {}
        summary = job.summary or {}
        preview_keys = summary.get('preview_image_keys') or []
        preview_image_urls = [storage.presign_get(key) for key in preview_keys if isinstance(key, str)]
        preview_pairs_raw = summary.get('preview_pairs') or []
        preview_pairs = []
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


@app.get(f'{settings.api_prefix}/datasets/yolo/{{dataset_job_id}}', response_model=YoloDatasetStatusResponse)
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


@app.get(f'{settings.api_prefix}/datasets/yolo/{{dataset_job_id}}/result', response_model=YoloDatasetResultResponse)
def get_yolo_dataset_result_endpoint(dataset_job_id: int, db: Session = Depends(get_db)) -> YoloDatasetResultResponse:
    zip_url, summary = get_yolo_dataset_result(db, dataset_job_id)
    preview_keys = summary.get('preview_image_keys') or []
    preview_urls = [storage.presign_get(key) for key in preview_keys if isinstance(key, str)]
    summary_payload = {**summary, 'preview_image_urls': preview_urls}
    return YoloDatasetResultResponse(
        zip_url=zip_url,
        summary=YoloDatasetResultSummary(**summary_payload),
    )


@app.post(f'{settings.api_prefix}/llm/enhance-preview', response_model=LlmEnhancePreviewResponse)
def llm_enhance_preview_endpoint(payload: LlmEnhancePreviewRequest, db: Session = Depends(get_db)) -> LlmEnhancePreviewResponse:
    items_raw = enhance_dataset_samples_with_langgraph(
        db=db,
        dataset_job_id=payload.dataset_job_id,
        sample_count=payload.sample_count,
        llm_model=payload.llm_model,
    )
    items = [LlmEnhancePreviewItem(**item) for item in items_raw]
    return LlmEnhancePreviewResponse(dataset_job_id=payload.dataset_job_id, items=items)


@app.post(f'{settings.api_prefix}/llm/enhance-dataset', response_model=CreateLlmEnhanceDatasetResponse)
def create_llm_enhance_dataset_endpoint(payload: CreateLlmEnhanceDatasetRequest, db: Session = Depends(get_db)) -> CreateLlmEnhanceDatasetResponse:
    job = create_llm_enhance_job(db=db, dataset_job_id=payload.dataset_job_id, llm_model=payload.llm_model)
    return CreateLlmEnhanceDatasetResponse(enhance_job_id=job.id, status=job.status)


@app.get(f'{settings.api_prefix}/llm/enhance-dataset/{{enhance_job_id}}', response_model=LlmEnhanceDatasetStatusResponse)
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


@app.get(f'{settings.api_prefix}/llm/enhance-dataset/{{enhance_job_id}}/result', response_model=LlmEnhanceDatasetResultResponse)
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
