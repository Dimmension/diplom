from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import Asset, AssetKind, RenderJob, RenderStatus, Scene
from app.schemas import SceneConfig
from app.services.render_queue import celery_app
from app.storage import storage

settings = get_settings()
ZIP_ENTRYPOINT_MANIFEST = '__entrypoint__.txt'


class RenderTaskError(RuntimeError):
    pass


def create_render_job(db: Session, scene_id: int, scene_config: SceneConfig) -> RenderJob:
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Scene not found')

    render_job = RenderJob(
        scene_id=scene_id,
        status=RenderStatus.queued,
        progress=0,
        scene_config_used=scene_config.model_dump(),
        scene_config_suggested=scene_config.model_dump(),
    )
    db.add(render_job)
    db.commit()
    db.refresh(render_job)

    run_render_job.delay(render_job.id)
    return render_job


def get_render_job(db: Session, render_job_id: int) -> RenderJob:
    render_job = db.query(RenderJob).filter(RenderJob.id == render_job_id).first()
    if render_job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Render job not found')
    return render_job


def get_render_result(db: Session, render_job_id: int) -> tuple[RenderJob, str]:
    render_job = get_render_job(db, render_job_id)
    if render_job.status != RenderStatus.succeeded or not render_job.result_key:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Render is not completed yet')
    return render_job, storage.presign_get(render_job.result_key)


def _set_job_status(job_id: int, *, status_value: RenderStatus, progress: int | None = None, error_code: str | None = None, error_message: str | None = None, result_key: str | None = None) -> None:
    db = SessionLocal()
    try:
        job = db.query(RenderJob).filter(RenderJob.id == job_id).first()
        if job is None:
            return
        job.status = status_value
        if progress is not None:
            job.progress = progress
        if error_code is not None:
            job.error_code = error_code
        if error_message is not None:
            job.error_message = error_message
        if result_key is not None:
            job.result_key = result_key
        if job.started_at is None and status_value == RenderStatus.running:
            job.started_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def _load_scene_assets(db: Session, scene: Scene, scene_config: dict) -> tuple[Asset, Asset, Asset | None]:
    object_asset = db.query(Asset).filter(Asset.id == scene.object_asset_id).first()
    env_asset = db.query(Asset).filter(Asset.id == scene.environment_asset_id).first()
    if object_asset is None or env_asset is None:
        raise RenderTaskError('Scene assets missing')
    skybox_asset_id = scene_config.get('skybox_asset_id')
    if skybox_asset_id is None:
        return object_asset, env_asset, None
    skybox_asset = db.query(Asset).filter(Asset.id == skybox_asset_id).first()
    if skybox_asset is None or skybox_asset.kind != AssetKind.skybox:
        raise RenderTaskError(f'Skybox asset is invalid: {skybox_asset_id}')
    return object_asset, env_asset, skybox_asset


def _run_blenderproc_render(
    *,
    object_key: str,
    environment_key: str,
    skybox_key: str | None,
    scene_config: dict,
    output_path: str,
    on_background_transition: Callable[[], None] | None = None,
) -> None:
    workdir = tempfile.mkdtemp(prefix='render_job_')
    object_local = os.path.join(workdir, f'object_{Path(object_key).name}')
    environment_local = os.path.join(workdir, f'environment_{Path(environment_key).name}')
    skybox_local = os.path.join(workdir, f'skybox_{Path(skybox_key).name}') if skybox_key else None
    config_local = os.path.join(workdir, 'scene_config.json')

    def resolve_asset_entrypoint(local_asset_path: str, extract_dir_name: str) -> str:
        if not local_asset_path.lower().endswith('.zip'):
            return local_asset_path

        extract_root = os.path.join(workdir, extract_dir_name)
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(local_asset_path, 'r') as zf:
            zf.extractall(extract_root)

        manifest_path = os.path.join(extract_root, ZIP_ENTRYPOINT_MANIFEST)
        if not os.path.exists(manifest_path):
            raise RenderTaskError(f'Missing {ZIP_ENTRYPOINT_MANIFEST} in asset archive')

        with open(manifest_path, 'r', encoding='utf-8') as fh:
            rel_entry = fh.read().strip().replace('\\', '/')
        if not rel_entry:
            raise RenderTaskError('Asset archive has empty entrypoint manifest')

        entry_abs = os.path.abspath(os.path.join(extract_root, rel_entry))
        if os.path.commonpath([entry_abs, os.path.abspath(extract_root)]) != os.path.abspath(extract_root):
            raise RenderTaskError('Entrypoint path escapes extracted archive root')
        if not os.path.exists(entry_abs):
            raise RenderTaskError(f'Entrypoint file not found in archive: {rel_entry}')
        return entry_abs

    try:
        storage.download_file(object_key, object_local)
        storage.download_file(environment_key, environment_local)
        if skybox_key and skybox_local:
            storage.download_file(skybox_key, skybox_local)
        object_entrypoint = resolve_asset_entrypoint(object_local, 'object_src')
        environment_entrypoint = resolve_asset_entrypoint(environment_local, 'environment_src')

        with open(config_local, 'w', encoding='utf-8') as fh:
            json.dump(
                {
                    'object_path': object_entrypoint,
                    'environment_path': environment_entrypoint,
                    'skybox_path': skybox_local,
                    'output_path': output_path,
                    'width': settings.render_width,
                    'height': settings.render_height,
                    'require_gpu': settings.render_require_gpu,
                    'scene_config': scene_config,
                },
                fh,
            )

        script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'render_with_blenderproc.py'
        cmd = ['blenderproc', 'run', str(script_path), config_local]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        start = time.time()
        transitioned_background = False

        while True:
            return_code = process.poll()
            elapsed = time.time() - start

            if not transitioned_background and elapsed > settings.render_timeout_seconds:
                transitioned_background = True
                if on_background_transition is not None:
                    on_background_transition()

            if return_code is not None:
                output = process.communicate()[0] if process.stdout else ''
                if return_code != 0:
                    tail = '\n'.join(output.splitlines()[-30:])
                    raise RenderTaskError(tail or 'blenderproc failed')
                return

            time.sleep(0.2)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@celery_app.task(name='app.services.renders.run_render_job')
def run_render_job(render_job_id: int) -> None:
    db = SessionLocal()
    temp_output = tempfile.mktemp(prefix=f'render_{render_job_id}_', suffix='.png')

    try:
        _set_job_status(render_job_id, status_value=RenderStatus.running, progress=10)

        job = db.query(RenderJob).filter(RenderJob.id == render_job_id).first()
        if job is None:
            return

        scene = db.query(Scene).filter(Scene.id == job.scene_id).first()
        if scene is None:
            raise RenderTaskError('Scene not found for render job')

        object_asset, env_asset, skybox_asset = _load_scene_assets(db, scene, job.scene_config_used)
        _set_job_status(render_job_id, status_value=RenderStatus.running, progress=30)

        _run_blenderproc_render(
            object_key=object_asset.original_key,
            environment_key=env_asset.original_key,
            skybox_key=skybox_asset.original_key if skybox_asset else None,
            scene_config=job.scene_config_used,
            output_path=temp_output,
            on_background_transition=lambda: _set_job_status(
                render_job_id, status_value=RenderStatus.running_background, progress=55
            ),
        )

        result_key = f'renders/{render_job_id}/final.png'
        storage.upload_file(temp_output, result_key, content_type='image/png')
        _set_job_status(render_job_id, status_value=RenderStatus.succeeded, progress=100, result_key=result_key)
    except RenderTaskError as err:
        _set_job_status(
            render_job_id,
            status_value=RenderStatus.failed,
            error_code='RENDER_FAILED',
            error_message=str(err),
            progress=100,
        )
    except Exception as err:  # noqa: BLE001
        _set_job_status(
            render_job_id,
            status_value=RenderStatus.failed,
            error_code='UNEXPECTED_ERROR',
            error_message=str(err),
            progress=100,
        )
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        db.close()
