from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Asset, AssetKind
from app.storage import storage

settings = get_settings()

MODEL_FORMATS = {'glb', 'gltf', 'obj', 'fbx'}
MODEL_UPLOAD_FORMATS = MODEL_FORMATS | {'zip'}
SKYBOX_FORMATS = {'hdr', 'exr'}
ALLOWED_FORMATS_BY_KIND = {
    AssetKind.object: MODEL_UPLOAD_FORMATS,
    AssetKind.environment: MODEL_UPLOAD_FORMATS,
    AssetKind.skybox: SKYBOX_FORMATS,
}
ZIP_ENTRYPOINT_MANIFEST = '__entrypoint__.txt'
MAX_ZIP_NESTING_DEPTH = 3


def _detect_extension(filename: str, kind: AssetKind) -> str:
    ext = Path(filename).suffix.lower().lstrip('.')
    allowed_formats = ALLOWED_FORMATS_BY_KIND[kind]
    if ext not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Unsupported format for kind={kind.value}: {ext}. Allowed: {sorted(allowed_formats)}',
        )
    return ext


def _safe_extract_zip(zip_path: str, output_dir: str) -> None:
    max_members = 20000
    max_uncompressed_bytes = 2_000_000_000
    total_uncompressed = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        infos = zf.infolist()
        if len(infos) > max_members:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f'ZIP has too many files: {len(infos)} > {max_members}',
            )

        for info in infos:
            name = info.filename
            if not name or name.endswith('/'):
                continue

            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f'ZIP contains unsafe path: {name}',
                )

            total_uncompressed += info.file_size
            if total_uncompressed > max_uncompressed_bytes:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f'ZIP uncompressed content too large: > {max_uncompressed_bytes} bytes',
                )

            target_path = os.path.abspath(os.path.join(output_dir, name))
            output_abs = os.path.abspath(output_dir)
            if os.path.commonpath([target_path, output_abs]) != output_abs:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f'ZIP path escapes destination: {name}',
                )

        zf.extractall(output_dir)


def _expand_nested_zip_archives(root_dir: str) -> None:
    for _ in range(MAX_ZIP_NESTING_DEPTH):
        nested_zip_rel_paths: list[str] = []
        for current_root, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.lower().endswith('.zip'):
                    abs_zip = os.path.join(current_root, file_name)
                    nested_zip_rel_paths.append(os.path.relpath(abs_zip, root_dir))

        if not nested_zip_rel_paths:
            return

        for rel_zip in nested_zip_rel_paths:
            abs_zip = os.path.join(root_dir, rel_zip)
            if not os.path.exists(abs_zip):
                continue
            extract_dir = abs_zip[:-4]
            os.makedirs(extract_dir, exist_ok=True)
            _safe_extract_zip(abs_zip, extract_dir)
            os.remove(abs_zip)

    remaining_nested = []
    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith('.zip'):
                remaining_nested.append(os.path.relpath(os.path.join(current_root, file_name), root_dir))
    if remaining_nested:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'ZIP nesting too deep. Remaining nested archives: {remaining_nested[:10]}',
        )


def _collect_model_candidates(root_dir: str) -> list[str]:
    candidates: list[str] = []
    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            ext = Path(file_name).suffix.lower().lstrip('.')
            if ext in MODEL_FORMATS:
                abs_path = os.path.join(current_root, file_name)
                rel_path = os.path.relpath(abs_path, root_dir)
                candidates.append(rel_path)
    return sorted(candidates)


def _resolve_entrypoint_from_archive(root_dir: str, archive_name: str) -> tuple[str, str]:
    candidates = _collect_model_candidates(root_dir)
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='ZIP does not contain supported 3D files (.glb/.gltf/.obj/.fbx)',
        )

    if len(candidates) == 1:
        rel_path = candidates[0]
        return os.path.join(root_dir, rel_path), rel_path

    stem = Path(archive_name).stem.lower()
    same_stem = [c for c in candidates if Path(c).stem.lower() == stem]
    if len(same_stem) == 1:
        rel_path = same_stem[0]
        return os.path.join(root_dir, rel_path), rel_path

    root_level = [c for c in candidates if len(Path(c).parts) == 1]
    if len(root_level) == 1:
        rel_path = root_level[0]
        return os.path.join(root_dir, rel_path), rel_path

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f'ZIP contains multiple model files; unable to choose entrypoint: {candidates[:10]}',
    )


def _build_source_package(source_root: str, entrypoint_rel_path: str, output_zip_path: str) -> None:
    with zipfile.ZipFile(output_zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for current_root, _, files in os.walk(source_root):
            for file_name in files:
                abs_path = os.path.join(current_root, file_name)
                rel_path = os.path.relpath(abs_path, source_root).replace('\\', '/')
                zf.write(abs_path, arcname=rel_path)
        zf.writestr(ZIP_ENTRYPOINT_MANIFEST, entrypoint_rel_path.replace('\\', '/'))


async def _save_upload_to_temp(upload: UploadFile) -> tuple[str, int]:
    temp_fd, temp_path = tempfile.mkstemp(prefix='upload_', suffix=Path(upload.filename or '').suffix)
    os.close(temp_fd)

    total_size = 0
    with open(temp_path, 'wb') as out_file:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > settings.max_upload_size_bytes:
                out_file.close()
                os.remove(temp_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f'File too large. Max is {settings.max_upload_size_bytes} bytes',
                )
            out_file.write(chunk)

    await upload.close()
    return temp_path, total_size


def _import_export_with_blender(input_path: str, output_glb_path: str) -> None:
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'convert_to_glb.py'
    cmd = [
        'blender',
        '--background',
        '--factory-startup',
        '--python-exit-code',
        '1',
        '--python',
        str(script_path),
        '--',
        input_path,
        output_glb_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Failed to convert asset to GLB: {result.stderr.strip() or result.stdout.strip()}',
        )


def _ensure_preview_glb(local_original_path: str, ext: str) -> str:
    if ext == 'glb':
        return local_original_path

    temp_preview_path = tempfile.mktemp(prefix='preview_', suffix='.glb')
    _import_export_with_blender(local_original_path, temp_preview_path)
    if not os.path.exists(temp_preview_path) or os.path.getsize(temp_preview_path) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Failed to generate GLB preview from uploaded asset',
        )
    return temp_preview_path


def _skybox_content_type(ext: str) -> str:
    if ext == 'hdr':
        return 'image/vnd.radiance'
    if ext == 'exr':
        return 'image/x-exr'
    return 'application/octet-stream'


def _persist_asset(
    *,
    db: Session,
    kind: AssetKind,
    filename: str,
    ext: str,
    size_bytes: int,
    original_key: str,
    preview_key: str,
) -> Asset:
    asset = Asset(
        kind=kind,
        original_filename=filename,
        detected_format=ext,
        size_bytes=size_bytes,
        original_key=original_key,
        preview_glb_key=preview_key,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def _resolve_zip_entrypoint(local_original_path: str, source_root: str, filename: str) -> tuple[str, str, str]:
    _safe_extract_zip(local_original_path, source_root)
    _expand_nested_zip_archives(source_root)
    entry_abs_path, entry_rel_path = _resolve_entrypoint_from_archive(source_root, filename)
    entry_ext = Path(entry_abs_path).suffix.lower().lstrip('.')
    return entry_abs_path, entry_rel_path, entry_ext


def _resolve_single_file_entrypoint(local_original_path: str, source_root: str, filename: str, ext: str) -> tuple[str, str, str]:
    entry_abs_path = os.path.join(source_root, Path(filename).name)
    shutil.copy2(local_original_path, entry_abs_path)
    entry_rel_path = Path(filename).name
    return entry_abs_path, entry_rel_path, ext


def _resolve_upload_entrypoint(local_original_path: str, source_root: str, filename: str, ext: str) -> tuple[str, str, str]:
    resolver_map = {
        'zip': lambda: _resolve_zip_entrypoint(local_original_path, source_root, filename),
    }
    resolver = resolver_map.get(ext, lambda: _resolve_single_file_entrypoint(local_original_path, source_root, filename, ext))
    return resolver()


def _upload_skybox_asset(
    *,
    db: Session,
    kind: AssetKind,
    filename: str,
    ext: str,
    size_bytes: int,
    local_original_path: str,
    asset_uuid: str,
) -> Asset:
    original_key = f'assets/{kind.value}/{asset_uuid}/source/{Path(filename).name}'
    storage.upload_file(local_original_path, original_key, content_type=_skybox_content_type(ext))
    return _persist_asset(
        db=db,
        kind=kind,
        filename=filename,
        ext=ext,
        size_bytes=size_bytes,
        original_key=original_key,
        preview_key=original_key,
    )


async def upload_asset(db: Session, kind: AssetKind, upload: UploadFile) -> Asset:
    if not upload.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Filename is required')

    ext = _detect_extension(upload.filename, kind)
    local_original_path, size_bytes = await _save_upload_to_temp(upload)
    workspace_dir = tempfile.mkdtemp(prefix='asset_ingest_')
    source_root = os.path.join(workspace_dir, 'source')
    os.makedirs(source_root, exist_ok=True)
    package_path = os.path.join(workspace_dir, 'source_bundle.zip')

    preview_path = local_original_path
    try:
        asset_uuid = uuid.uuid4().hex
        kind_handlers = {
            AssetKind.skybox: lambda: _upload_skybox_asset(
                db=db,
                kind=kind,
                filename=upload.filename,
                ext=ext,
                size_bytes=size_bytes,
                local_original_path=local_original_path,
                asset_uuid=asset_uuid,
            ),
        }
        kind_handler = kind_handlers.get(kind)
        if kind_handler is not None:
            return kind_handler()

        entry_abs_path, entry_rel_path, entry_ext = _resolve_upload_entrypoint(
            local_original_path=local_original_path,
            source_root=source_root,
            filename=upload.filename,
            ext=ext,
        )

        preview_path = _ensure_preview_glb(entry_abs_path, entry_ext)
        _build_source_package(source_root, entry_rel_path, package_path)

        original_key = f'assets/{kind.value}/{asset_uuid}/source/source_bundle.zip'
        preview_key = f'assets/{kind.value}/{asset_uuid}/preview/scene.glb'

        try:
            storage.upload_file(package_path, original_key, content_type='application/zip')
            storage.upload_file(preview_path, preview_key, content_type='model/gltf-binary')
        except FileNotFoundError as err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f'Preview file missing after conversion: {err}',
            ) from err

        asset = _persist_asset(
            db=db,
            kind=kind,
            filename=upload.filename,
            ext=ext,
            size_bytes=size_bytes,
            original_key=original_key,
            preview_key=preview_key,
        )
        return asset
    finally:
        for path in {local_original_path, preview_path}:
            if path and os.path.exists(path):
                os.remove(path)
        shutil.rmtree(workspace_dir, ignore_errors=True)


def list_assets(db: Session, kind: AssetKind | None = None, limit: int = 50) -> list[Asset]:
    query = db.query(Asset)
    if kind is not None:
        query = query.filter(Asset.kind == kind)
    return query.order_by(Asset.created_at.desc()).limit(limit).all()
