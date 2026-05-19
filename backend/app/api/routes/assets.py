from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import AssetKind
from app.schemas import AssetKindSchema, AssetListResponse, AssetUploadResponse
from app.services.assets import list_assets, upload_asset
from app.storage import storage

router = APIRouter()


@router.get('/assets', response_model=AssetListResponse)
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


@router.post('/assets/upload', response_model=AssetUploadResponse)
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
