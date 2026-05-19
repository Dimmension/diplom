from __future__ import annotations

import base64
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import io
import random
import threading
import uuid
from datetime import datetime, timezone
from typing import Any
from typing_extensions import TypedDict

import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import DatasetStatus, LlmEnhanceJob, YoloDatasetJob
from app.services.render_queue import celery_app
from app.services.storage_helpers import list_object_keys
from app.storage import storage

settings = get_settings()
_thread_local = threading.local()


class EnhanceState(TypedDict, total=False):
    image_bytes: bytes
    mime_type: str
    mask_bytes: bytes
    mask_mime_type: str
    llm_prompt: str
    enhanced_image_bytes: bytes


def _storage_list_keys_adapter(prefix: str) -> list[str]:
    return list_object_keys(storage_client=storage.client, bucket=settings.s3_bucket, prefix=prefix)


def _storage_get_object_adapter(key: str) -> dict[str, Any]:
    return storage.client.get_object(Bucket=settings.s3_bucket, Key=key)


def _storage_upload_bytes_adapter(payload: bytes, key: str, *, content_type: str) -> None:
    storage.upload_bytes(payload, key, content_type=content_type)


def _storage_presign_get_adapter(key: str) -> str:
    return storage.presign_get(key)


def _create_openai_client_adapter() -> Any:
    from openai import OpenAI

    return OpenAI()


def _openai_edit_image_adapter(image_client: Any, *, image: io.BytesIO, mask: io.BytesIO, prompt: str) -> Any:
    return image_client.images.edit(
        model='gpt-image-1.5',
        image=image,
        mask=mask,
        prompt=prompt,
        size='640x640',
        quality='high',
        output_format='png',
    )


def _http_get_bytes_adapter(url: str, *, timeout: int) -> bytes:
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _build_enhancer_graph(llm_model: str):
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, StateGraph
        image_client = _create_openai_client_adapter()
    except ImportError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='LLM enhancement dependencies are missing. Rebuild backend image with updated requirements.',
        ) from err

    llm = ChatOpenAI(model=llm_model, temperature=0)

    def build_edit_prompt(state: EnhanceState) -> dict[str, Any]:
        image_bytes = state['image_bytes']
        mime_type = state.get('mime_type') or 'image/png'
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        prompt = (
            'You are an expert in computer vision preprocessing for object detection datasets. '
            'Return only a concise image-edit prompt (plain text, no JSON, no markdown) for an image-to-image model. '
            'Primary objective: preserve object geometry exactly (shape, pose, silhouette, boundaries, scale, and pixel alignment). '
            'Everything else is secondary. '
            'Target look: photo captured by a real camera/photo camera, highly photorealistic, high-quality, natural lighting and textures, non-CGI appearance. '
            'Never request crop, resize, rotation, warp, perspective change, object relocation, or new objects. '
            'Never change composition. Keep bbox/mask alignment valid.'
        )

        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=[
                        {'type': 'text', 'text': 'Produce only the final edit prompt string for camera-like photorealistic enhancement.'},
                        {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_b64}'}},
                    ]
                ),
            ]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        llm_prompt = raw.strip()
        if not llm_prompt:
            llm_prompt = (
                'Enhance this render into a realistic camera photo while strictly preserving object geometry and composition; '
                'improve photorealism, natural light, realistic texture, and high-quality detail.'
            )
        return {'llm_prompt': llm_prompt}

    def _mime_to_name(mime_type: str) -> str:
        if 'jpeg' in mime_type or 'jpg' in mime_type:
            return 'input.jpg'
        if 'webp' in mime_type:
            return 'input.webp'
        return 'input.png'

    def edit_with_image_model(state: EnhanceState) -> dict[str, Any]:
        mime_type = state.get('mime_type') or 'image/png'
        file_like = io.BytesIO(state['image_bytes'])
        file_like.name = _mime_to_name(mime_type)
        mask_mime_type = state.get('mask_mime_type') or 'image/png'
        mask_file_like = io.BytesIO(state['mask_bytes'])
        mask_file_like.name = _mime_to_name(mask_mime_type)
        response = _openai_edit_image_adapter(
            image_client,
            image=file_like,
            mask=mask_file_like,
            prompt=state['llm_prompt'],
        )

        data0 = response.data[0]
        if getattr(data0, 'b64_json', None):
            return {'enhanced_image_bytes': base64.b64decode(data0.b64_json)}
        if getattr(data0, 'url', None):
            return {'enhanced_image_bytes': _http_get_bytes_adapter(data0.url, timeout=120)}
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Image model returned no image payload')

    graph_builder = StateGraph(EnhanceState)
    graph_builder.add_node('build_edit_prompt', build_edit_prompt)
    graph_builder.add_node('edit_with_image_model', edit_with_image_model)
    graph_builder.add_edge(START, 'build_edit_prompt')
    graph_builder.add_edge('build_edit_prompt', 'edit_with_image_model')
    graph_builder.add_edge('edit_with_image_model', END)
    return graph_builder.compile()


def _get_thread_enhancer_graph(llm_model: str):
    cache = getattr(_thread_local, 'enhancer_graph_cache', None)
    if not isinstance(cache, dict):
        cache = {}
        _thread_local.enhancer_graph_cache = cache
    graph = cache.get(llm_model)
    if graph is None:
        graph = _build_enhancer_graph(llm_model=llm_model)
        cache[llm_model] = graph
    return graph


def _resolve_llm_enhance_parallelism(total_items: int) -> int:
    if total_items <= 1:
        return 1
    configured = int(settings.llm_enhance_parallelism or 4)
    return max(1, min(configured, total_items))


def _enhance_triplet_with_graph(triplet: dict[str, str], *, llm_model: str) -> dict[str, Any]:
    graph = _get_thread_enhancer_graph(llm_model=llm_model)
    image_bytes, mime_type = _read_object_bytes(triplet['image_key'])
    mask_bytes, mask_mime_type = _read_object_bytes(triplet['mask_key'])
    final_state = graph.invoke(
        {
            'image_bytes': image_bytes,
            'mime_type': mime_type,
            'mask_bytes': mask_bytes,
            'mask_mime_type': mask_mime_type,
        }
    )
    enhanced_bytes = final_state.get('enhanced_image_bytes')
    if not isinstance(enhanced_bytes, (bytes, bytearray)):
        raise RuntimeError('Enhancer graph returned no output image bytes')
    return {
        'enhanced_bytes': bytes(enhanced_bytes),
        'edit_prompt': str(final_state.get('llm_prompt') or ''),
    }


def _list_keys(prefix: str) -> list[str]:
    return _storage_list_keys_adapter(prefix)


def _read_object_bytes(key: str) -> tuple[bytes, str]:
    payload = _storage_get_object_adapter(key)
    body = payload.get('Body')
    if body is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'S3 object body is empty: {key}')
    data = body.read()
    content_type = str(payload.get('ContentType') or 'application/octet-stream')
    return data, content_type


def _select_random_triplets(dataset_prefix: str, sample_count: int) -> list[dict[str, str]]:
    triplets = _collect_triplets(dataset_prefix=dataset_prefix)
    take = min(sample_count, len(triplets))
    return random.sample(triplets, k=take)


def _collect_triplets(dataset_prefix: str) -> list[dict[str, str]]:
    image_keys = _list_keys(f'{dataset_prefix}/images')
    bbox_keys = set(_list_keys(f'{dataset_prefix}/debug/bbox'))
    mask_keys = set(_list_keys(f'{dataset_prefix}/debug/mask'))

    triplets: list[dict[str, str]] = []
    images_prefix = f'{dataset_prefix.rstrip("/")}/images/'

    for image_key in image_keys:
        if not image_key.startswith(images_prefix):
            continue
        rel = image_key[len(images_prefix):]
        if '/' not in rel:
            continue
        split, file_name = rel.split('/', 1)
        bbox_key = f'{dataset_prefix}/debug/bbox/{split}/{file_name}'
        mask_key = f'{dataset_prefix}/debug/mask/{split}/{file_name}'
        if bbox_key in bbox_keys and mask_key in mask_keys:
            triplets.append(
                {
                    'image_key': image_key,
                    'bbox_key': bbox_key,
                    'mask_key': mask_key,
                }
            )

    if not triplets:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='No image/bbox/mask triplets were found in dataset artifacts. Generate dataset with debug outputs.',
        )

    return triplets


def _get_dataset_prefix_or_raise(dataset_job_id: int, db: Session) -> str:
    job = db.query(YoloDatasetJob).filter(YoloDatasetJob.id == dataset_job_id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset job not found')
    if job.status != DatasetStatus.succeeded:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Dataset generation is not completed yet')

    summary = job.summary or {}
    dataset_prefix = summary.get('dataset_prefix')
    if not isinstance(dataset_prefix, str) or not dataset_prefix:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Dataset artifacts are unavailable for this job. Regenerate dataset with current pipeline.',
        )
    return dataset_prefix


def _set_enhance_job_status(
    enhance_job_id: int,
    *,
    status_value: DatasetStatus,
    progress: int | None = None,
    error_message: str | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    db = SessionLocal()
    try:
        job = db.query(LlmEnhanceJob).filter(LlmEnhanceJob.id == enhance_job_id).first()
        if job is None:
            return
        job.status = status_value
        if progress is not None:
            job.progress = progress
        if error_message is not None:
            job.error_message = error_message
        if summary is not None:
            job.summary = summary
        if job.started_at is None and status_value == DatasetStatus.running:
            job.started_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def enhance_dataset_samples_with_langgraph(
    db: Session,
    dataset_job_id: int,
    sample_count: int = 2,
    llm_model: str = 'gpt-5.4',
) -> list[dict[str, Any]]:
    dataset_prefix = _get_dataset_prefix_or_raise(dataset_job_id=dataset_job_id, db=db)
    triplets = _select_random_triplets(dataset_prefix=dataset_prefix, sample_count=sample_count)
    workers = _resolve_llm_enhance_parallelism(len(triplets))

    response_items_by_index: dict[int, dict[str, Any]] = {}
    batch_id = uuid.uuid4().hex

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_meta: dict[Future[dict[str, Any]], tuple[int, dict[str, str]]] = {
            pool.submit(_enhance_triplet_with_graph, triplet, llm_model=llm_model): (index, triplet)
            for index, triplet in enumerate(triplets, start=1)
        }
        for future in as_completed(future_to_meta):
            index, triplet = future_to_meta[future]
            try:
                result = future.result()
            except HTTPException:
                raise
            except Exception as err:  # noqa: BLE001
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)) from err

            enhanced_key = f'datasets/yolo/{dataset_job_id}/enhanced/{batch_id}/sample_{index:02d}.png'
            _storage_upload_bytes_adapter(result['enhanced_bytes'], enhanced_key, content_type='image/png')
            response_items_by_index[index] = {
                'image_url': _storage_presign_get_adapter(triplet['image_key']),
                'bbox_url': _storage_presign_get_adapter(triplet['bbox_key']),
                'mask_url': _storage_presign_get_adapter(triplet['mask_key']),
                'enhanced_image_url': _storage_presign_get_adapter(enhanced_key),
                'enhancement_plan': {
                    'llm_model': llm_model,
                    'image_model': 'gpt-image-1.5',
                    'edit_prompt': result['edit_prompt'],
                },
            }

    return [response_items_by_index[idx] for idx in sorted(response_items_by_index.keys())]


def create_llm_enhance_job(db: Session, dataset_job_id: int, llm_model: str) -> LlmEnhanceJob:
    _ = _get_dataset_prefix_or_raise(dataset_job_id=dataset_job_id, db=db)
    job = LlmEnhanceJob(
        dataset_job_id=dataset_job_id,
        status=DatasetStatus.queued,
        progress=0,
        config={'llm_model': llm_model},
        summary=None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    run_llm_enhance_job.delay(job.id)
    return job


def get_llm_enhance_job(db: Session, enhance_job_id: int) -> LlmEnhanceJob:
    job = db.query(LlmEnhanceJob).filter(LlmEnhanceJob.id == enhance_job_id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='LLM enhance job not found')
    return job


def get_llm_enhance_job_result(db: Session, enhance_job_id: int) -> dict[str, Any]:
    job = get_llm_enhance_job(db, enhance_job_id)
    if job.status != DatasetStatus.succeeded:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='LLM enhance job is not completed yet')
    summary = job.summary or {}
    sample_pairs = summary.get('sample_pairs') or []

    items: list[dict[str, Any]] = []
    for pair in sample_pairs:
        if not isinstance(pair, dict):
            continue
        image_key = pair.get('image_key')
        bbox_key = pair.get('bbox_key')
        mask_key = pair.get('mask_key')
        enhanced_key = pair.get('enhanced_key')
        plan = pair.get('enhancement_plan') or {}
        if not all(isinstance(x, str) for x in (image_key, bbox_key, mask_key, enhanced_key)):
            continue
        items.append(
            {
                'image_url': _storage_presign_get_adapter(image_key),
                'bbox_url': _storage_presign_get_adapter(bbox_key),
                'mask_url': _storage_presign_get_adapter(mask_key),
                'enhanced_image_url': _storage_presign_get_adapter(enhanced_key),
                'enhancement_plan': plan,
            }
        )

    return {
        'enhance_job_id': job.id,
        'dataset_job_id': job.dataset_job_id,
        'status': job.status,
        'processed_images': int(summary.get('processed_images') or 0),
        'total_images': int(summary.get('total_images') or 0),
        'sample_items': items,
    }


@celery_app.task(name='app.services.llm_enhancer.run_llm_enhance_job')
def run_llm_enhance_job(enhance_job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(LlmEnhanceJob).filter(LlmEnhanceJob.id == enhance_job_id).first()
        if job is None:
            return

        _set_enhance_job_status(enhance_job_id, status_value=DatasetStatus.running, progress=2)

        llm_model = str((job.config or {}).get('llm_model') or 'gpt-5.4')
        dataset_prefix = _get_dataset_prefix_or_raise(dataset_job_id=job.dataset_job_id, db=db)
        triplets = _collect_triplets(dataset_prefix=dataset_prefix)
        total = len(triplets)
        workers = _resolve_llm_enhance_parallelism(total)

        run_id = uuid.uuid4().hex
        enhanced_prefix = f'datasets/yolo/{job.dataset_job_id}/enhanced_full/{run_id}'

        sample_for_preview = random.sample(triplets, k=min(2, total))
        sample_keys = {(x['image_key'], x['bbox_key'], x['mask_key']) for x in sample_for_preview}
        sample_pairs_with_index: list[tuple[int, dict[str, Any]]] = []
        processed_count = 0
        images_prefix = f'{dataset_prefix.rstrip("/")}/images/'

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_meta: dict[Future[dict[str, Any]], tuple[int, dict[str, str]]] = {
                pool.submit(_enhance_triplet_with_graph, triplet, llm_model=llm_model): (idx, triplet)
                for idx, triplet in enumerate(triplets, start=1)
            }

            for future in as_completed(future_to_meta):
                idx, triplet = future_to_meta[future]
                result = future.result()
                rel = triplet['image_key'][len(images_prefix):]
                enhanced_key = f'{enhanced_prefix}/images/{rel}'
                _storage_upload_bytes_adapter(result['enhanced_bytes'], enhanced_key, content_type='image/png')

                if (triplet['image_key'], triplet['bbox_key'], triplet['mask_key']) in sample_keys:
                    sample_pairs_with_index.append(
                        (
                            idx,
                            {
                                'image_key': triplet['image_key'],
                                'bbox_key': triplet['bbox_key'],
                                'mask_key': triplet['mask_key'],
                                'enhanced_key': enhanced_key,
                                'enhancement_plan': {
                                    'llm_model': llm_model,
                                    'image_model': 'gpt-image-1.5',
                                    'edit_prompt': result['edit_prompt'],
                                },
                            },
                        )
                    )

                processed_count += 1
                progress = 2 + int((processed_count / max(total, 1)) * 96)
                _set_enhance_job_status(
                    enhance_job_id,
                    status_value=DatasetStatus.running,
                    progress=min(progress, 98),
                )

        sample_pairs = [item for _, item in sorted(sample_pairs_with_index, key=lambda pair: pair[0])]

        _set_enhance_job_status(
            enhance_job_id,
            status_value=DatasetStatus.succeeded,
            progress=100,
            summary={
                'enhanced_prefix': enhanced_prefix,
                'processed_images': total,
                'total_images': total,
                'sample_pairs': sample_pairs,
            },
        )
    except HTTPException as err:
        _set_enhance_job_status(
            enhance_job_id,
            status_value=DatasetStatus.failed,
            progress=100,
            error_message=str(err.detail),
        )
    except Exception as err:  # noqa: BLE001
        _set_enhance_job_status(
            enhance_job_id,
            status_value=DatasetStatus.failed,
            progress=100,
            error_message=str(err),
        )
    finally:
        db.close()
