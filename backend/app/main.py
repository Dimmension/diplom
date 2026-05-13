from fastapi import Depends, FastAPI, File, Query, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import Base, engine, get_db
from app.models import AssetKind
from app.schemas import (
    AssetListResponse,
    AssetKindSchema,
    AssetUploadResponse,
    CreateRenderRequest,
    CreateRenderResponse,
    CreateYoloDatasetRequest,
    CreateYoloDatasetResponse,
    CreateSceneRequest,
    RenderResultResponse,
    RenderStatusResponse,
    SceneResponse,
    UpdateSceneConfigRequest,
    YoloDatasetResultResponse,
    YoloDatasetResultSummary,
    YoloDatasetStatusResponse,
)
from app.services.assets import list_assets, upload_asset
from app.services.renders import (
    create_render_job,
    create_yolo_dataset_job,
    get_additional_render_urls,
    get_render_job,
    get_render_result,
    get_yolo_dataset_job,
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
    return YoloDatasetResultResponse(
        zip_url=zip_url,
        summary=YoloDatasetResultSummary(**summary),
    )
