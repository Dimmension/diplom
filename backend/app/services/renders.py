from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import random
import re
import select
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import Asset, AssetKind, DatasetStatus, RenderJob, RenderStatus, Scene, YoloDatasetJob
from app.schemas import CreateYoloDatasetRequest, SceneConfig
from app.services.render_queue import celery_app
from app.storage import storage

settings = get_settings()
ZIP_ENTRYPOINT_MANIFEST = '__entrypoint__.txt'
logger = logging.getLogger(__name__)


class RenderTaskError(RuntimeError):
    pass


class GpuUnavailableError(RenderTaskError):
    pass


GPU_CHECK_CACHE_TTL_SECONDS = 30.0
_gpu_check_cache: tuple[float, bool, str | None] | None = None


def _ensure_render_worker_available() -> None:
    try:
        inspector = celery_app.control.inspect(timeout=1.5)
        reachable_workers = inspector.ping()
    except Exception:  # noqa: BLE001
        reachable_workers = None

    if not reachable_workers:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Render worker is unavailable. GPU render is required.',
        )


def _run_gpu_probe() -> None:
    global _gpu_check_cache

    if not settings.render_require_gpu:
        return

    now = time.time()
    if _gpu_check_cache and now - _gpu_check_cache[0] < GPU_CHECK_CACHE_TTL_SECONDS:
        _, ok, cached_error = _gpu_check_cache
        if ok:
            return
        raise GpuUnavailableError(cached_error or 'GPU is required but unavailable')

    probe_script = """
import bpy
import sys

scene = bpy.context.scene
scene.render.engine = 'CYCLES'

addon = bpy.context.preferences.addons.get('cycles')
if addon is None:
    print('Cycles addon not found')
    sys.exit(19)

prefs = addon.preferences
selected_backend = None

for backend in ('OPTIX', 'CUDA'):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
    except Exception as exc:  # noqa: BLE001
        print(f'Backend {backend} init failed: {exc}')
        continue

    has_gpu = any(dev.type in {'CUDA', 'OPTIX'} and dev.name for dev in prefs.devices)
    if has_gpu:
        selected_backend = backend
        break

devices_dump = [(dev.name, dev.type, dev.use) for dev in prefs.devices]
if selected_backend is None:
    print(f'No GPU backend available. Devices: {devices_dump!r}')
    sys.exit(17)

print(f'Selected backend: {selected_backend}')
print(f'Devices: {devices_dump!r}')
sys.exit(0)
"""

    probe_cmd = ['blender', '-b', '--python-expr', probe_script]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as err:
        message = 'GPU probe timed out before render start'
        _gpu_check_cache = (now, False, message)
        raise GpuUnavailableError(message) from err
    except FileNotFoundError as err:
        message = 'Blender executable is unavailable in worker runtime'
        _gpu_check_cache = (now, False, message)
        raise GpuUnavailableError(message) from err

    if result.returncode == 0:
        _gpu_check_cache = (now, True, None)
        return

    stderr_tail = '\n'.join(result.stderr.splitlines()[-5:]).strip()
    stdout_tail = '\n'.join(result.stdout.splitlines()[-5:]).strip()
    details = stderr_tail or stdout_tail
    message = 'GPU is required but no GPU device found'
    if details:
        message = f'{message}. {details}'

    _gpu_check_cache = (now, False, message)
    raise GpuUnavailableError(message)


def create_render_job(db: Session, scene_id: int, scene_config: SceneConfig) -> RenderJob:
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Scene not found')
    _ensure_render_worker_available()

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


def get_additional_render_urls(render_job_id: int) -> tuple[str, str]:
    mask_key = f'renders/{render_job_id}/mask.png'
    bbox_key = f'renders/{render_job_id}/bbox.png'
    return storage.presign_get(mask_key), storage.presign_get(bbox_key)


def create_yolo_dataset_job(db: Session, payload: CreateYoloDatasetRequest) -> YoloDatasetJob:
    if payload.split_train_count + payload.split_val_count != payload.count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='split_train_count + split_val_count must equal count',
        )
    if payload.randomization_preset != 'medium':
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Only randomization_preset=medium is supported',
        )

    scene = db.query(Scene).filter(Scene.id == payload.scene_id).first()
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Scene not found')

    _ensure_render_worker_available()
    job = YoloDatasetJob(
        scene_id=payload.scene_id,
        status=DatasetStatus.queued,
        progress=0,
        config=payload.model_dump(),
        summary=None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    run_yolo_dataset_job.delay(job.id)
    return job


def get_yolo_dataset_job(db: Session, dataset_job_id: int) -> YoloDatasetJob:
    job = db.query(YoloDatasetJob).filter(YoloDatasetJob.id == dataset_job_id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset job not found')
    return job


def list_yolo_dataset_jobs(db: Session, limit: int = 50, status_filter: DatasetStatus | None = None) -> list[YoloDatasetJob]:
    query = db.query(YoloDatasetJob)
    if status_filter is not None:
        query = query.filter(YoloDatasetJob.status == status_filter)
    return query.order_by(YoloDatasetJob.updated_at.desc(), YoloDatasetJob.id.desc()).limit(limit).all()


def get_yolo_dataset_result(db: Session, dataset_job_id: int) -> tuple[str, dict]:
    job = get_yolo_dataset_job(db, dataset_job_id)
    if job.status != DatasetStatus.succeeded:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Dataset generation is not completed yet')
    summary = job.summary or {}

    # Backward compatibility: old jobs store a prebuilt ZIP key.
    if job.result_key:
        return storage.presign_get(job.result_key), summary

    dataset_prefix = summary.get('dataset_prefix')
    if not isinstance(dataset_prefix, str) or not dataset_prefix:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Dataset artifacts are unavailable')

    temp_dir = tempfile.mkdtemp(prefix=f'dataset_download_{dataset_job_id}_')
    try:
        zip_path = os.path.join(temp_dir, 'dataset.zip')
        files_count = _build_dataset_zip_from_storage(dataset_prefix=dataset_prefix, zip_path=zip_path)
        if files_count == 0:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Dataset artifacts are empty')

        zip_key = f'datasets/yolo/{dataset_job_id}/downloads/dataset.zip'
        storage.upload_file(zip_path, zip_key, content_type='application/zip')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return storage.presign_get(zip_key), summary


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


def _set_dataset_job_status(
    dataset_job_id: int,
    *,
    status_value: DatasetStatus,
    progress: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    result_key: str | None = None,
    summary: dict | None = None,
) -> None:
    db = SessionLocal()
    try:
        job = db.query(YoloDatasetJob).filter(YoloDatasetJob.id == dataset_job_id).first()
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
        if summary is not None:
            job.summary = summary
        if job.started_at is None and status_value == DatasetStatus.running:
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
    mask_output_path: str | None = None,
    bbox_output_path: str | None = None,
    bbox_meta_output_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    samples: list[dict] | None = None,
    on_background_transition: Callable[[], None] | None = None,
    on_saved_output: Callable[[str], None] | None = None,
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
                    'mask_output_path': mask_output_path,
                    'bbox_output_path': bbox_output_path,
                    'bbox_meta_output_path': bbox_meta_output_path,
                    'width': width or settings.render_width,
                    'height': height or settings.render_height,
                    'require_gpu': settings.render_require_gpu,
                    'samples': settings.render_samples,
                    'use_adaptive_sampling': settings.render_use_adaptive_sampling,
                    'adaptive_threshold': settings.render_adaptive_threshold,
                    'use_denoising': settings.render_use_denoising,
                    'denoiser': settings.render_denoiser,
                    'max_bounces': settings.render_max_bounces,
                    'scene_config': scene_config,
                    'dataset_samples': samples,
                },
                fh,
            )

        script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'render_with_blenderproc.py'
        cmd = ['blenderproc', 'run', str(script_path), config_local]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        start = time.time()
        transitioned_background = False
        output_lines: list[str] = []
        saved_line_regex = re.compile(r"Saved:\s+['\"]([^'\"]+)['\"]")

        def process_output_line(line: str) -> None:
            cleaned = line.rstrip('\n')
            if not cleaned:
                return
            output_lines.append(cleaned)
            if on_saved_output is None:
                return
            match = saved_line_regex.search(cleaned)
            if not match:
                return
            on_saved_output(match.group(1))

        while True:
            elapsed = time.time() - start

            if not transitioned_background and elapsed > settings.render_timeout_seconds:
                transitioned_background = True
                if on_background_transition is not None:
                    on_background_transition()

            if process.stdout:
                ready, _, _ = select.select([process.stdout], [], [], 0.2)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        process_output_line(line)
                        continue

            return_code = process.poll()
            if return_code is not None:
                if process.stdout:
                    rest = process.stdout.read() or ''
                    for line in rest.splitlines():
                        process_output_line(line)
                if return_code != 0:
                    tail = '\n'.join(output_lines[-30:])
                    if settings.render_require_gpu and 'GPU is required but no GPU device found' in tail:
                        raise GpuUnavailableError('GPU is required but no GPU device found')
                    raise RenderTaskError(tail or 'blenderproc failed')
                return

            time.sleep(0.05)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@celery_app.task(name='app.services.renders.run_render_job')
def run_render_job(render_job_id: int) -> None:
    db = SessionLocal()
    temp_output = tempfile.mktemp(prefix=f'render_{render_job_id}_', suffix='.png')
    temp_mask = tempfile.mktemp(prefix=f'mask_{render_job_id}_', suffix='.png')
    temp_bbox = tempfile.mktemp(prefix=f'bbox_{render_job_id}_', suffix='.png')
    job_started = time.perf_counter()

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
        _run_gpu_probe()
        _set_job_status(render_job_id, status_value=RenderStatus.running, progress=35)

        render_started = time.perf_counter()
        _run_blenderproc_render(
            object_key=object_asset.original_key,
            environment_key=env_asset.original_key,
            skybox_key=skybox_asset.original_key if skybox_asset else None,
            scene_config=job.scene_config_used,
            output_path=temp_output,
            mask_output_path=temp_mask,
            bbox_output_path=temp_bbox,
            on_background_transition=lambda: _set_job_status(
                render_job_id, status_value=RenderStatus.running_background, progress=55
            ),
        )
        logger.info(
            'Render job %s: blenderproc stage completed in %.2fs',
            render_job_id,
            time.perf_counter() - render_started,
        )

        result_key = f'renders/{render_job_id}/final.png'
        mask_key = f'renders/{render_job_id}/mask.png'
        bbox_key = f'renders/{render_job_id}/bbox.png'
        storage.upload_file(temp_output, result_key, content_type='image/png')
        storage.upload_file(temp_mask, mask_key, content_type='image/png')
        storage.upload_file(temp_bbox, bbox_key, content_type='image/png')
        _set_job_status(render_job_id, status_value=RenderStatus.succeeded, progress=100, result_key=result_key)
        logger.info(
            'Render job %s: total pipeline completed in %.2fs',
            render_job_id,
            time.perf_counter() - job_started,
        )
    except GpuUnavailableError as err:
        _set_job_status(
            render_job_id,
            status_value=RenderStatus.failed,
            error_code='GPU_UNAVAILABLE',
            error_message=str(err),
            progress=100,
        )
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
        if os.path.exists(temp_mask):
            os.remove(temp_mask)
        if os.path.exists(temp_bbox):
            os.remove(temp_bbox)
        db.close()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _axis_range(source: dict | None, axis: str, fallback: float) -> float:
    value = fallback if source is None else source.get(axis, fallback)
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return abs(float(fallback))


def _range_value(value: float | int | None, fallback: float) -> float:
    if value is None:
        return abs(float(fallback))
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return abs(float(fallback))


def _jitter_scene_config(base_scene_config: dict, rng: random.Random) -> dict:
    scene_config = deepcopy(base_scene_config)
    jitter_cfg = scene_config.get('generation_jitter') or {}
    camera_jitter = jitter_cfg.get('camera') or {}
    environment_jitter = jitter_cfg.get('environment') or {}

    camera = scene_config['camera']
    camera_pos_range = camera_jitter.get('position') or {}
    camera_target_range = camera_jitter.get('target') or {}

    camera['position']['x'] += rng.uniform(-_axis_range(camera_pos_range, 'x', 0.8), _axis_range(camera_pos_range, 'x', 0.8))
    camera['position']['y'] += rng.uniform(-_axis_range(camera_pos_range, 'y', 0.8), _axis_range(camera_pos_range, 'y', 0.8))
    camera['position']['z'] += rng.uniform(-_axis_range(camera_pos_range, 'z', 0.4), _axis_range(camera_pos_range, 'z', 0.4))
    camera['target']['x'] += rng.uniform(-_axis_range(camera_target_range, 'x', 0.25), _axis_range(camera_target_range, 'x', 0.25))
    camera['target']['y'] += rng.uniform(-_axis_range(camera_target_range, 'y', 0.25), _axis_range(camera_target_range, 'y', 0.25))
    camera['target']['z'] += rng.uniform(-_axis_range(camera_target_range, 'z', 0.25), _axis_range(camera_target_range, 'z', 0.25))
    camera['fov_degrees'] = _clamp(
        float(camera['fov_degrees']) + rng.uniform(-_range_value(camera_jitter.get('fov_degrees'), 5.0), _range_value(camera_jitter.get('fov_degrees'), 5.0)),
        30.0,
        80.0,
    )

    distance_to_object_range = _range_value(camera_jitter.get('distance_to_object'), 0.0)
    if distance_to_object_range > 0.0:
        object_pos = scene_config.get('object_transform', {}).get('position', {})
        obj_x = float(object_pos.get('x', 0.0))
        obj_y = float(object_pos.get('y', 0.0))
        obj_z = float(object_pos.get('z', 0.0))

        cam_x = float(camera['position']['x'])
        cam_y = float(camera['position']['y'])
        cam_z = float(camera['position']['z'])

        vx = cam_x - obj_x
        vy = cam_y - obj_y
        vz = cam_z - obj_z
        current_distance = math.sqrt(vx * vx + vy * vy + vz * vz)
        if current_distance > 1e-6:
            target_distance = _clamp(
                current_distance + rng.uniform(-distance_to_object_range, distance_to_object_range),
                0.1,
                1000.0,
            )
            scale = target_distance / current_distance
            camera['position']['x'] = obj_x + vx * scale
            camera['position']['y'] = obj_y + vy * scale
            camera['position']['z'] = obj_z + vz * scale

    obj = scene_config['object_transform']
    obj['position']['x'] += rng.uniform(-0.5, 0.5)
    obj['position']['y'] += rng.uniform(-0.5, 0.5)
    obj['position']['z'] += rng.uniform(-0.5, 0.5)
    obj['rotation']['x'] += rng.uniform(-25.0, 25.0)
    obj['rotation']['y'] += rng.uniform(-25.0, 25.0)
    obj['rotation']['z'] += rng.uniform(-25.0, 25.0)

    scale_mul = rng.uniform(0.85, 1.15)
    obj['scale']['x'] = _clamp(float(obj['scale']['x']) * scale_mul, 0.05, 10.0)
    obj['scale']['y'] = _clamp(float(obj['scale']['y']) * scale_mul, 0.05, 10.0)
    obj['scale']['z'] = _clamp(float(obj['scale']['z']) * scale_mul, 0.05, 10.0)

    env = scene_config['environment_transform']
    env_position_range = environment_jitter.get('position') or {}
    env_rotation_range = environment_jitter.get('rotation') or {}
    env['position']['x'] += rng.uniform(-_axis_range(env_position_range, 'x', 0.0), _axis_range(env_position_range, 'x', 0.0))
    env['position']['y'] += rng.uniform(-_axis_range(env_position_range, 'y', 0.0), _axis_range(env_position_range, 'y', 0.0))
    env['position']['z'] += rng.uniform(-_axis_range(env_position_range, 'z', 0.0), _axis_range(env_position_range, 'z', 0.0))
    env['rotation']['x'] += rng.uniform(-_axis_range(env_rotation_range, 'x', 0.0), _axis_range(env_rotation_range, 'x', 0.0))
    env['rotation']['y'] += rng.uniform(-_axis_range(env_rotation_range, 'y', 0.0), _axis_range(env_rotation_range, 'y', 0.0))
    env['rotation']['z'] += rng.uniform(-_axis_range(env_rotation_range, 'z', 0.0), _axis_range(env_rotation_range, 'z', 0.0))

    return scene_config


def _bbox_meta_to_yolo_line(meta: dict) -> str | None:
    if not meta.get('visible'):
        return None
    width = float(meta['width'])
    height = float(meta['height'])
    xmin = float(meta['xmin'])
    ymin = float(meta['ymin'])
    xmax = float(meta['xmax'])
    ymax = float(meta['ymax'])

    box_w = max(0.0, (xmax - xmin + 1.0) / width)
    box_h = max(0.0, (ymax - ymin + 1.0) / height)
    center_x = ((xmin + xmax + 1.0) / 2.0) / width
    center_y = ((ymin + ymax + 1.0) / 2.0) / height

    center_x = _clamp(center_x, 0.0, 1.0)
    center_y = _clamp(center_y, 0.0, 1.0)
    box_w = _clamp(box_w, 0.0, 1.0)
    box_h = _clamp(box_h, 0.0, 1.0)
    return f'0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n'


def _guess_content_type(file_path: str) -> str:
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or 'application/octet-stream'


def _upload_tree_to_storage(src_dir: str, prefix: str) -> list[str]:
    uploaded_keys: list[str] = []
    for root, _, files in os.walk(src_dir):
        for file_name in files:
            abs_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(abs_path, src_dir).replace(os.sep, '/')
            key = f'{prefix}/{rel_path}'
            storage.upload_file(abs_path, key, content_type=_guess_content_type(rel_path))
            uploaded_keys.append(key)
    uploaded_keys.sort()
    return uploaded_keys


def _list_storage_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None
    while True:
        params: dict[str, object] = {
            'Bucket': settings.s3_bucket,
            'Prefix': f'{prefix.rstrip("/")}/',
        }
        if continuation_token:
            params['ContinuationToken'] = continuation_token
        payload = storage.client.list_objects_v2(**params)
        for item in payload.get('Contents', []):
            key = item.get('Key')
            if isinstance(key, str) and not key.endswith('/'):
                keys.append(key)
        if not payload.get('IsTruncated'):
            break
        continuation_token = payload.get('NextContinuationToken')
        if not continuation_token:
            break
    keys.sort()
    return keys


def _build_dataset_zip_from_storage(dataset_prefix: str, zip_path: str) -> int:
    keys = _list_storage_keys(dataset_prefix)
    prefix = f'{dataset_prefix.rstrip("/")}/'
    files_count = 0

    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for key in keys:
            if not key.startswith(prefix):
                continue
            rel_path = key[len(prefix):]
            if not rel_path:
                continue
            obj = storage.client.get_object(Bucket=settings.s3_bucket, Key=key)
            body = obj.get('Body')
            if body is None:
                continue
            data = body.read()
            zf.writestr(rel_path, data)
            files_count += 1

    return files_count


@celery_app.task(name='app.services.renders.run_yolo_dataset_job')
def run_yolo_dataset_job(dataset_job_id: int) -> None:
    db = SessionLocal()
    workspace_dir = tempfile.mkdtemp(prefix=f'yolo_dataset_{dataset_job_id}_')

    try:
        _set_dataset_job_status(dataset_job_id, status_value=DatasetStatus.running, progress=5)
        job = db.query(YoloDatasetJob).filter(YoloDatasetJob.id == dataset_job_id).first()
        if job is None:
            return

        config = job.config or {}
        scene = db.query(Scene).filter(Scene.id == job.scene_id).first()
        if scene is None:
            raise RenderTaskError('Scene not found for dataset job')

        count = int(config['count'])
        train_count = int(config['split_train_count'])
        val_count = int(config['split_val_count'])
        if train_count + val_count != count:
            raise RenderTaskError('Invalid split configuration')

        include_debug = bool(config.get('include_debug', True))
        width = int(config['width'])
        height = int(config['height'])
        base_scene_config = config['scene_config_snapshot']
        parallel_workers = max(1, min(int(settings.dataset_render_parallelism), count))
        logger.info(
            'Dataset job %s START: count=%s train=%s val=%s include_debug=%s size=%sx%s parallel_workers=%s',
            dataset_job_id,
            count,
            train_count,
            val_count,
            include_debug,
            width,
            height,
            parallel_workers,
        )

        object_asset, env_asset, skybox_asset = _load_scene_assets(db, scene, base_scene_config)
        _run_gpu_probe()
        _set_dataset_job_status(dataset_job_id, status_value=DatasetStatus.running, progress=15)

        dataset_root = os.path.join(workspace_dir, 'dataset')
        os.makedirs(os.path.join(dataset_root, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'labels', 'val'), exist_ok=True)

        debug_mask_base = os.path.join(dataset_root, 'debug', 'mask')
        debug_bbox_base = os.path.join(dataset_root, 'debug', 'bbox')
        if include_debug:
            os.makedirs(os.path.join(debug_mask_base, 'train'), exist_ok=True)
            os.makedirs(os.path.join(debug_mask_base, 'val'), exist_ok=True)
            os.makedirs(os.path.join(debug_bbox_base, 'train'), exist_ok=True)
            os.makedirs(os.path.join(debug_bbox_base, 'val'), exist_ok=True)

        aux_mask_base = os.path.join(workspace_dir, 'aux_mask')
        aux_bbox_base = os.path.join(workspace_dir, 'aux_bbox')
        aux_meta_base = os.path.join(workspace_dir, 'meta')
        os.makedirs(aux_mask_base, exist_ok=True)
        os.makedirs(aux_bbox_base, exist_ok=True)
        os.makedirs(aux_meta_base, exist_ok=True)

        rng = random.Random(dataset_job_id)
        dataset_samples: list[dict] = []
        sample_index: list[tuple[str, str]] = []
        sample_output_to_meta: dict[str, tuple[int, str, str]] = {}
        for idx in range(count):
            split = 'train' if idx < train_count else 'val'
            stem = f'{idx:05d}'
            image_path = os.path.join(dataset_root, 'images', split, f'{stem}.png')
            label_path = os.path.join(dataset_root, 'labels', split, f'{stem}.txt')

            if include_debug:
                mask_path = os.path.join(debug_mask_base, split, f'{stem}.png')
                bbox_path = os.path.join(debug_bbox_base, split, f'{stem}.png')
            else:
                mask_path = os.path.join(aux_mask_base, f'{stem}.png')
                bbox_path = os.path.join(aux_bbox_base, f'{stem}.png')
            meta_path = os.path.join(aux_meta_base, f'{stem}.json')

            jittered_scene = _jitter_scene_config(base_scene_config, rng)
            dataset_samples.append(
                {
                    'scene_config': jittered_scene,
                    'output_path': image_path,
                    'mask_output_path': mask_path,
                    'bbox_output_path': bbox_path,
                    'bbox_meta_output_path': meta_path,
                }
            )
            sample_index.append((split, label_path))
            sample_output_to_meta[os.path.normpath(image_path)] = (idx + 1, split, stem)

        logger.info('Dataset job %s: prepared %s scene samples', dataset_job_id, count)

        _set_dataset_job_status(dataset_job_id, status_value=DatasetStatus.running, progress=25)
        completed_output_paths: set[str] = set()
        progress_lock = threading.Lock()
        chunk_map: dict[int, list[dict]] = {idx: [] for idx in range(parallel_workers)}
        for idx, sample in enumerate(dataset_samples):
            chunk_map[idx % parallel_workers].append(sample)
        chunks = [chunk_map[idx] for idx in range(parallel_workers) if chunk_map[idx]]

        def log_image_start(sample: dict) -> None:
            normalized_path = os.path.normpath(sample['output_path'])
            meta = sample_output_to_meta.get(normalized_path)
            if meta is None:
                return
            ordinal, split, stem = meta
            logger.info(
                'Dataset job %s | IMAGE %s/%s START | split=%s file=%s',
                dataset_job_id,
                ordinal,
                count,
                split,
                f'{stem}.png',
            )

        def mark_image_done(saved_path: str) -> None:
            normalized_path = os.path.normpath(saved_path)
            meta = sample_output_to_meta.get(normalized_path)
            if meta is None:
                return
            with progress_lock:
                if normalized_path in completed_output_paths:
                    return
                completed_output_paths.add(normalized_path)
                done = len(completed_output_paths)
                progress_pct = (done / count) * 100.0
                job_progress = 25 + int((done * 60) / count)
                job_progress = max(25, min(85, job_progress))
                _set_dataset_job_status(dataset_job_id, status_value=DatasetStatus.running, progress=job_progress)
            ordinal, split, stem = meta
            logger.info(
                'Dataset job %s | IMAGE %s/%s END | done=%.2f%% | split=%s file=%s | progress=%s%%',
                dataset_job_id,
                ordinal,
                count,
                progress_pct,
                split,
                f'{stem}.png',
                job_progress,
            )

        def render_chunk(chunk_idx: int, chunk_samples: list[dict]) -> None:
            cursor = 0
            log_image_start(chunk_samples[cursor])

            def on_saved_output(saved_path: str) -> None:
                nonlocal cursor
                mark_image_done(saved_path)
                cursor += 1
                if cursor < len(chunk_samples):
                    log_image_start(chunk_samples[cursor])

            _run_blenderproc_render(
                object_key=object_asset.original_key,
                environment_key=env_asset.original_key,
                skybox_key=skybox_asset.original_key if skybox_asset else None,
                scene_config=base_scene_config,
                output_path=chunk_samples[0]['output_path'],
                width=width,
                height=height,
                samples=chunk_samples,
                on_saved_output=on_saved_output,
            )
            logger.info(
                'Dataset job %s: chunk %s finished (%s images)',
                dataset_job_id,
                chunk_idx + 1,
                len(chunk_samples),
            )

        logger.info('Dataset job %s: launching %s parallel render chunks', dataset_job_id, len(chunks))
        with ThreadPoolExecutor(max_workers=len(chunks), thread_name_prefix=f'dataset_{dataset_job_id}') as pool:
            futures = [pool.submit(render_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
            for future in futures:
                future.result()

        _set_dataset_job_status(dataset_job_id, status_value=DatasetStatus.running, progress=90)
        logger.info('Dataset job %s: render stage completed. rendered_images=%s/%s', dataset_job_id, len(completed_output_paths), count)

        empty_labels = 0
        for idx, (_, label_path) in enumerate(sample_index):
            meta_path = dataset_samples[idx]['bbox_meta_output_path']
            with open(meta_path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
            yolo_line = _bbox_meta_to_yolo_line(meta)
            if yolo_line is None:
                empty_labels += 1
                with open(label_path, 'w', encoding='utf-8') as fh:
                    fh.write('')
            else:
                with open(label_path, 'w', encoding='utf-8') as fh:
                    fh.write(yolo_line)
        _set_dataset_job_status(dataset_job_id, status_value=DatasetStatus.running, progress=95)
        logger.info('Dataset job %s: labels generated, empty_labels=%s', dataset_job_id, empty_labels)

        with open(os.path.join(dataset_root, 'data.yaml'), 'w', encoding='utf-8') as fh:
            fh.write('path: .\n')
            fh.write('train: images/train\n')
            fh.write('val: images/val\n')
            fh.write('nc: 1\n')
            fh.write('names:\n')
            fh.write('  0: object\n')

        preview_indices: list[int] = []
        preview_target = min(8, count)
        if preview_target == 1:
            preview_indices = [0]
        elif preview_target > 1:
            for i in range(preview_target):
                idx = round(i * (count - 1) / (preview_target - 1))
                if idx not in preview_indices:
                    preview_indices.append(idx)

        _set_dataset_job_status(dataset_job_id, status_value=DatasetStatus.running, progress=97)
        dataset_prefix = f'datasets/yolo/{dataset_job_id}/dataset'
        uploaded_keys = _upload_tree_to_storage(dataset_root, dataset_prefix)
        uploaded_keys_set = set(uploaded_keys)
        logger.info('Dataset job %s: dataset files uploaded (%s)', dataset_job_id, len(uploaded_keys))

        preview_pairs: list[dict] = []
        preview_candidates: list[dict] = []
        for idx in range(count):
            split, _ = sample_index[idx]
            image_path = dataset_samples[idx]['output_path']
            stem = os.path.splitext(os.path.basename(image_path))[0]
            image_key = f'{dataset_prefix}/images/{split}/{stem}.png'
            bbox_key = f'{dataset_prefix}/debug/bbox/{split}/{stem}.png'
            mask_key = f'{dataset_prefix}/debug/mask/{split}/{stem}.png'
            if image_key in uploaded_keys_set and bbox_key in uploaded_keys_set and mask_key in uploaded_keys_set:
                preview_candidates.append(
                    {
                        'image_key': image_key,
                        'bbox_key': bbox_key,
                        'mask_key': mask_key,
                    }
                )

        if preview_candidates:
            pick_count = min(2, len(preview_candidates))
            preview_pairs = random.sample(preview_candidates, k=pick_count)

        preview_image_keys: list[str] = []
        if preview_pairs:
            preview_image_keys = [pair['image_key'] for pair in preview_pairs]
        else:
            for idx in preview_indices:
                image_path = dataset_samples[idx]['output_path']
                rel_path = os.path.relpath(image_path, dataset_root).replace(os.sep, '/')
                key = f'{dataset_prefix}/{rel_path}'
                if key in uploaded_keys_set:
                    preview_image_keys.append(key)

        summary = {
            'images_total': count,
            'train_count': train_count,
            'val_count': val_count,
            'empty_labels_count': empty_labels,
            'class_names': ['object'],
            'dataset_prefix': dataset_prefix,
            'preview_image_keys': preview_image_keys,
            'preview_pairs': preview_pairs,
        }
        _set_dataset_job_status(
            dataset_job_id,
            status_value=DatasetStatus.succeeded,
            progress=100,
            summary=summary,
        )
        logger.info(
            'Dataset job %s END: completed. images_total=%s train=%s val=%s empty_labels=%s',
            dataset_job_id,
            count,
            train_count,
            val_count,
            empty_labels,
        )
    except GpuUnavailableError as err:
        _set_dataset_job_status(
            dataset_job_id,
            status_value=DatasetStatus.failed,
            progress=100,
            error_code='GPU_UNAVAILABLE',
            error_message=str(err),
        )
        logger.error('Dataset job %s END: failed. code=GPU_UNAVAILABLE message=%s', dataset_job_id, err)
    except RenderTaskError as err:
        _set_dataset_job_status(
            dataset_job_id,
            status_value=DatasetStatus.failed,
            progress=100,
            error_code='DATASET_FAILED',
            error_message=str(err),
        )
        logger.error('Dataset job %s END: failed. code=DATASET_FAILED message=%s', dataset_job_id, err)
    except Exception as err:  # noqa: BLE001
        _set_dataset_job_status(
            dataset_job_id,
            status_value=DatasetStatus.failed,
            progress=100,
            error_code='UNEXPECTED_ERROR',
            error_message=str(err),
        )
        logger.exception('Dataset job %s END: failed. code=UNEXPECTED_ERROR', dataset_job_id)
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        db.close()
