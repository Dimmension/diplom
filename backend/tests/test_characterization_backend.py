from __future__ import annotations

import io
import json
import os
import random
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models import DatasetStatus
from app.services import llm_enhancer, renders


class _FakeStdout:
    def readline(self) -> str:
        return ''

    def read(self) -> str:
        return ''


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = _FakeStdout()

    def poll(self) -> int:
        return 0


def _write_zip(local_path: str, *, manifest_value: str | None, members: dict[str, str]) -> None:
    with zipfile.ZipFile(local_path, 'w') as zf:
        if manifest_value is not None:
            zf.writestr(renders.ZIP_ENTRYPOINT_MANIFEST, manifest_value)
        for rel_path, payload in members.items():
            zf.writestr(rel_path, payload)


def _minimal_scene_config() -> dict:
    return {
        'camera': {
            'position': {'x': 1.0, 'y': 1.0, 'z': 1.0},
            'target': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'fov_degrees': 50.0,
        },
        'object_transform': {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'scale': {'x': 1.0, 'y': 1.0, 'z': 1.0},
        },
        'environment_transform': {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'scale': {'x': 1.0, 'y': 1.0, 'z': 1.0},
        },
        'generation_jitter': {
            'camera': {
                'position': {'x': 0.8, 'y': 0.8, 'z': 0.4},
                'target': {'x': 0.25, 'y': 0.25, 'z': 0.25},
                'fov_degrees': 5.0,
                'distance_to_object': 0.0,
            },
            'environment': {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            },
        },
        'skybox_asset_id': None,
    }


def test_zip_entrypoint_manifest_is_used_for_render_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_config: dict = {}
    output_path = str(tmp_path / 'result.png')

    zip_payloads = {
        'object.zip': ('mesh/object.glb', {'mesh/object.glb': 'obj data'}),
        'environment.zip': ('world/env.glb', {'world/env.glb': 'env data'}),
    }

    def fake_download_file(key: str, local_path: str) -> None:
        manifest, members = zip_payloads[key]
        _write_zip(local_path, manifest_value=manifest, members=members)

    def fake_popen(cmd: list[str], **_: object) -> _FakeProcess:
        with open(cmd[-1], 'r', encoding='utf-8') as fh:
            captured_config.update(json.load(fh))
        return _FakeProcess()

    monkeypatch.setattr(renders.storage, 'download_file', fake_download_file)
    monkeypatch.setattr(renders.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(renders.select, 'select', lambda *_args, **_kwargs: ([], [], []))
    monkeypatch.setattr(renders.time, 'sleep', lambda *_args, **_kwargs: None)

    renders._run_blenderproc_render(
        object_key='object.zip',
        environment_key='environment.zip',
        skybox_key=None,
        scene_config=_minimal_scene_config(),
        output_path=output_path,
    )

    assert captured_config['object_path'].endswith('/object_src/mesh/object.glb')
    assert captured_config['environment_path'].endswith('/environment_src/world/env.glb')
    assert captured_config['skybox_path'] is None
    assert captured_config['output_path'] == output_path


def test_zip_entrypoint_rejects_empty_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_download_file(key: str, local_path: str) -> None:
        if key == 'object.zip':
            _write_zip(local_path, manifest_value='  \n', members={'mesh/object.glb': 'obj data'})
            return
        _write_zip(local_path, manifest_value='world/env.glb', members={'world/env.glb': 'env data'})

    monkeypatch.setattr(renders.storage, 'download_file', fake_download_file)

    with pytest.raises(renders.RenderTaskError, match='empty entrypoint manifest'):
        renders._run_blenderproc_render(
            object_key='object.zip',
            environment_key='environment.zip',
            skybox_key=None,
            scene_config=_minimal_scene_config(),
            output_path=str(tmp_path / 'result.png'),
        )


def test_zip_entrypoint_rejects_path_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_download_file(key: str, local_path: str) -> None:
        if key == 'object.zip':
            _write_zip(local_path, manifest_value='../escape.glb', members={'escape.glb': 'obj data'})
            return
        _write_zip(local_path, manifest_value='world/env.glb', members={'world/env.glb': 'env data'})

    monkeypatch.setattr(renders.storage, 'download_file', fake_download_file)

    with pytest.raises(renders.RenderTaskError, match='escapes extracted archive root'):
        renders._run_blenderproc_render(
            object_key='object.zip',
            environment_key='environment.zip',
            skybox_key=None,
            scene_config=_minimal_scene_config(),
            output_path=str(tmp_path / 'result.png'),
        )


def test_jitter_math_remains_deterministic_for_seeded_rng() -> None:
    base_config = {
        'camera': {
            'position': {'x': 1.0, 'y': 2.0, 'z': 3.0},
            'target': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'fov_degrees': 50.0,
        },
        'object_transform': {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'scale': {'x': 1.0, 'y': 1.0, 'z': 1.0},
        },
        'environment_transform': {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'scale': {'x': 1.0, 'y': 1.0, 'z': 1.0},
        },
        'generation_jitter': {
            'camera': {
                'position': {'x': 1.5, 'y': 2.5, 'z': 3.5},
                'target': {'x': 0.5, 'y': 1.5, 'z': 2.5},
                'fov_degrees': 9.0,
                'distance_to_object': 1.2,
            },
            'environment': {
                'position': {'x': 0.1, 'y': 0.2, 'z': 0.3},
                'rotation': {'x': 1.0, 'y': 2.0, 'z': 3.0},
            },
        },
    }
    original = deepcopy(base_config)

    actual = renders._jitter_scene_config(base_config, random.Random(123))

    assert base_config == original
    assert actual['camera']['fov_degrees'] == pytest.approx(50.65163672061068)
    assert actual['camera']['position']['x'] == pytest.approx(-0.2847976798528559)
    assert actual['camera']['position']['y'] == pytest.approx(-0.05320948630244352)
    assert actual['camera']['position']['z'] == pytest.approx(1.952329419482382)
    assert actual['object_transform']['scale']['x'] == pytest.approx(0.9808827380245655)
    assert actual['object_transform']['scale']['y'] == pytest.approx(0.9808827380245655)
    assert actual['object_transform']['scale']['z'] == pytest.approx(0.9808827380245655)
    assert actual['environment_transform']['rotation']['z'] == pytest.approx(2.4323919137069936)


def test_dataset_summary_prefers_preview_pairs_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_job_id = 77
    scene_config = _minimal_scene_config()
    job = SimpleNamespace(
        id=dataset_job_id,
        scene_id=11,
        config={
            'count': 4,
            'split_train_count': 3,
            'split_val_count': 1,
            'include_debug': True,
            'width': 256,
            'height': 256,
            'scene_config_snapshot': scene_config,
        },
    )
    scene = SimpleNamespace(id=11)
    db_closed = {'value': False}
    status_updates: list[dict] = []

    class FakeQuery:
        def __init__(self, value: object) -> None:
            self.value = value

        def filter(self, *_args: object, **_kwargs: object) -> 'FakeQuery':
            return self

        def first(self) -> object:
            return self.value

    class FakeSession:
        def query(self, model: object) -> FakeQuery:
            if model is renders.YoloDatasetJob:
                return FakeQuery(job)
            if model is renders.Scene:
                return FakeQuery(scene)
            raise AssertionError(f'Unexpected query model: {model}')

        def close(self) -> None:
            db_closed['value'] = True

    def fake_run_blenderproc_render(*, samples: list[dict], on_saved_output, width: int, height: int, **_kwargs: object) -> None:
        for sample in samples:
            Path(sample['output_path']).parent.mkdir(parents=True, exist_ok=True)
            Path(sample['mask_output_path']).parent.mkdir(parents=True, exist_ok=True)
            Path(sample['bbox_output_path']).parent.mkdir(parents=True, exist_ok=True)
            Path(sample['output_path']).write_bytes(b'png')
            Path(sample['mask_output_path']).write_bytes(b'png')
            Path(sample['bbox_output_path']).write_bytes(b'png')
            with open(sample['bbox_meta_output_path'], 'w', encoding='utf-8') as fh:
                json.dump({'visible': True, 'width': width, 'height': height, 'xmin': 10, 'ymin': 10, 'xmax': 40, 'ymax': 60}, fh)
            if on_saved_output is not None:
                on_saved_output(sample['output_path'])

    def fake_upload_tree(src_dir: str, prefix: str) -> list[str]:
        keys: list[str] = []
        for root, _, files in os.walk(src_dir):
            for file_name in files:
                abs_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(abs_path, src_dir).replace(os.sep, '/')
                keys.append(f'{prefix}/{rel_path}')
        keys.sort()
        return keys

    monkeypatch.setattr(renders, 'SessionLocal', lambda: FakeSession())
    monkeypatch.setattr(renders, '_load_scene_assets', lambda *_args, **_kwargs: (SimpleNamespace(original_key='obj.glb'), SimpleNamespace(original_key='env.glb'), None))
    monkeypatch.setattr(renders, '_run_gpu_probe', lambda: None)
    monkeypatch.setattr(renders, '_run_blenderproc_render', fake_run_blenderproc_render)
    monkeypatch.setattr(renders, '_upload_tree_to_storage', fake_upload_tree)
    monkeypatch.setattr(renders.random, 'sample', lambda seq, k: list(seq)[:k])
    monkeypatch.setattr(
        renders,
        '_set_dataset_job_status',
        lambda _job_id, **kwargs: status_updates.append(kwargs),
    )

    renders.run_yolo_dataset_job(dataset_job_id)

    succeeded_updates = [x for x in status_updates if x.get('status_value') == DatasetStatus.succeeded]
    assert succeeded_updates, 'Expected final succeeded status update'
    summary = succeeded_updates[-1]['summary']

    assert summary['dataset_prefix'] == f'datasets/yolo/{dataset_job_id}/dataset'
    assert summary['images_total'] == 4
    assert summary['train_count'] == 3
    assert summary['val_count'] == 1
    assert len(summary['preview_pairs']) == 2
    assert summary['preview_image_keys'] == [pair['image_key'] for pair in summary['preview_pairs']]
    assert db_closed['value'] is True


def test_collect_triplets_requires_matching_bbox_and_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = 'datasets/yolo/999/dataset'

    def fake_list_keys(list_prefix: str) -> list[str]:
        mapping = {
            f'{prefix}/images': [
                f'{prefix}/images/train/00001.png',
                f'{prefix}/images/train/00002.png',
                f'{prefix}/images/without-split.png',
                f'other-prefix/images/train/00003.png',
            ],
            f'{prefix}/debug/bbox': [
                f'{prefix}/debug/bbox/train/00001.png',
                f'{prefix}/debug/bbox/train/00002.png',
            ],
            f'{prefix}/debug/mask': [
                f'{prefix}/debug/mask/train/00001.png',
            ],
        }
        return mapping[list_prefix]

    monkeypatch.setattr(llm_enhancer, '_list_keys', fake_list_keys)

    actual = llm_enhancer._collect_triplets(prefix)

    assert actual == [
        {
            'image_key': f'{prefix}/images/train/00001.png',
            'bbox_key': f'{prefix}/debug/bbox/train/00001.png',
            'mask_key': f'{prefix}/debug/mask/train/00001.png',
        }
    ]


def test_collect_triplets_raises_when_no_complete_triplet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_enhancer, '_list_keys', lambda _prefix: [])

    with pytest.raises(HTTPException) as raised:
        llm_enhancer._collect_triplets('datasets/yolo/888/dataset')

    assert raised.value.status_code == 409
    assert 'No image/bbox/mask triplets' in str(raised.value.detail)


def test_openai_edit_image_adapter_passes_mask_and_quality() -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeImagesClient:
        def edit(self, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            return object()

    client = SimpleNamespace(images=_FakeImagesClient())
    image = io.BytesIO(b'image')
    image.name = 'scene.png'
    mask = io.BytesIO(b'mask')
    mask.name = 'mask.png'

    llm_enhancer._openai_edit_image_adapter(
        client,
        image=image,
        mask=mask,
        prompt='replace only masked area',
    )

    assert captured_kwargs['model'] == 'gpt-image-1.5'
    assert captured_kwargs['image'] is image
    assert captured_kwargs['mask'] is mask
    assert captured_kwargs['prompt'] == 'replace only masked area'
    assert captured_kwargs['size'] == '640x640'
    assert captured_kwargs['quality'] == 'high'
    assert captured_kwargs['output_format'] == 'png'
