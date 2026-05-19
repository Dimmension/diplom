from __future__ import annotations

from typing import Any


def list_object_keys(*, storage_client: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None
    normalized_prefix = f'{prefix.rstrip("/")}/'

    while True:
        params: dict[str, object] = {
            'Bucket': bucket,
            'Prefix': normalized_prefix,
        }
        if continuation_token:
            params['ContinuationToken'] = continuation_token

        payload = storage_client.list_objects_v2(**params)
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
