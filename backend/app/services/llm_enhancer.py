from __future__ import annotations

import base64
import io
import random
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
from app.storage import storage

settings = get_settings()


class EnhanceState(TypedDict, total=False):
    image_bytes: bytes
    mime_type: str
    llm_prompt: str
    enhanced_image_bytes: bytes


def _build_enhancer_graph(llm_model: str):
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, StateGraph
        from openai import OpenAI
    except ImportError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='LLM enhancement dependencies are missing. Rebuild backend image with updated requirements.',
        ) from err

    llm = ChatOpenAI(model=llm_model, temperature=0)
    image_client = OpenAI()

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
            'transform into anime-drone girls'
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
        response = image_client.images.edit(
            model='gpt-image-2',
            image=file_like,
            prompt=state['llm_prompt'],
            output_format='png',
        )

        data0 = response.data[0]
        if getattr(data0, 'b64_json', None):
            return {'enhanced_image_bytes': base64.b64decode(data0.b64_json)}
        if getattr(data0, 'url', None):
            img = httpx.get(data0.url, timeout=120)
            img.raise_for_status()
            return {'enhanced_image_bytes': img.content}
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Image model returned no image payload')

    graph_builder = StateGraph(EnhanceState)
    graph_builder.add_node('build_edit_prompt', build_edit_prompt)
    graph_builder.add_node('edit_with_image_model', edit_with_image_model)
    graph_builder.add_edge(START, 'build_edit_prompt')
    graph_builder.add_edge('build_edit_prompt', 'edit_with_image_model')
    graph_builder.add_edge('edit_with_image_model', END)
    return graph_builder.compile()


def _list_keys(prefix: str) -> list[str]:
    base_prefix = f"{prefix.rstrip('/')}/"
    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        params: dict[str, Any] = {
            'Bucket': settings.s3_bucket,
            'Prefix': base_prefix,
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


def _read_object_bytes(key: str) -> tuple[bytes, str]:
    payload = storage.client.get_object(Bucket=settings.s3_bucket, Key=key)
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
    graph = _build_enhancer_graph(llm_model=llm_model)

    response_items: list[dict[str, Any]] = []
    batch_id = uuid.uuid4().hex

    for index, triplet in enumerate(triplets, start=1):
        image_bytes, mime_type = _read_object_bytes(triplet['image_key'])

        final_state = graph.invoke(
            {
                'image_bytes': image_bytes,
                'mime_type': mime_type,
            }
        )
        enhanced_bytes = final_state.get('enhanced_image_bytes')
        if not isinstance(enhanced_bytes, (bytes, bytearray)):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Enhancer graph returned no output image bytes')

        enhanced_key = f'datasets/yolo/{dataset_job_id}/enhanced/{batch_id}/sample_{index:02d}.png'
        storage.upload_bytes(bytes(enhanced_bytes), enhanced_key, content_type='image/png')

        response_items.append(
            {
                'image_url': storage.presign_get(triplet['image_key']),
                'bbox_url': storage.presign_get(triplet['bbox_key']),
                'mask_url': storage.presign_get(triplet['mask_key']),
                'enhanced_image_url': storage.presign_get(enhanced_key),
                'enhancement_plan': {
                    'llm_model': llm_model,
                    'image_model': 'gpt-image-1.5',
                    'edit_prompt': final_state.get('llm_prompt') or '',
                },
            }
        )

    return response_items


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
                'image_url': storage.presign_get(image_key),
                'bbox_url': storage.presign_get(bbox_key),
                'mask_url': storage.presign_get(mask_key),
                'enhanced_image_url': storage.presign_get(enhanced_key),
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
        graph = _build_enhancer_graph(llm_model=llm_model)

        run_id = uuid.uuid4().hex
        enhanced_prefix = f'datasets/yolo/{job.dataset_job_id}/enhanced_full/{run_id}'

        sample_for_preview = random.sample(triplets, k=min(2, total))
        sample_keys = {(x['image_key'], x['bbox_key'], x['mask_key']) for x in sample_for_preview}
        sample_pairs: list[dict[str, Any]] = []

        for idx, triplet in enumerate(triplets, start=1):
            image_bytes, mime_type = _read_object_bytes(triplet['image_key'])
            final_state = graph.invoke({'image_bytes': image_bytes, 'mime_type': mime_type})
            enhanced_bytes = final_state.get('enhanced_image_bytes')
            if not isinstance(enhanced_bytes, (bytes, bytearray)):
                raise RuntimeError('Enhancer graph returned no output image bytes')

            rel = triplet['image_key'][len(f'{dataset_prefix.rstrip("/")}/images/'):]
            enhanced_key = f'{enhanced_prefix}/images/{rel}'
            storage.upload_bytes(bytes(enhanced_bytes), enhanced_key, content_type='image/png')

            if (triplet['image_key'], triplet['bbox_key'], triplet['mask_key']) in sample_keys:
                sample_pairs.append(
                    {
                        'image_key': triplet['image_key'],
                        'bbox_key': triplet['bbox_key'],
                        'mask_key': triplet['mask_key'],
                        'enhanced_key': enhanced_key,
                        'enhancement_plan': {
                            'llm_model': llm_model,
                            'image_model': 'gpt-image-1.5',
                            'edit_prompt': final_state.get('llm_prompt') or '',
                        },
                    }
                )

            progress = 2 + int((idx / max(total, 1)) * 96)
            _set_enhance_job_status(
                enhance_job_id,
                status_value=DatasetStatus.running,
                progress=min(progress, 98),
            )

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
