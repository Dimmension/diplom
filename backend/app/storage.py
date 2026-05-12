from __future__ import annotations

import boto3
from urllib.parse import urlparse, urlunparse

from app.core.config import get_settings

settings = get_settings()


class S3Storage:
    def __init__(self) -> None:
        self._client = boto3.client(
            's3',
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )

    @property
    def client(self):
        return self._client

    def upload_file(self, local_path: str, key: str, content_type: str = 'application/octet-stream') -> None:
        self._client.upload_file(local_path, settings.s3_bucket, key, ExtraArgs={'ContentType': content_type})

    def upload_bytes(self, payload: bytes, key: str, content_type: str = 'application/octet-stream') -> None:
        self._client.put_object(Bucket=settings.s3_bucket, Key=key, Body=payload, ContentType=content_type)

    def download_file(self, key: str, local_path: str) -> None:
        self._client.download_file(settings.s3_bucket, key, local_path)

    def presign_get(self, key: str) -> str:
        presigned = self._client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.s3_bucket, 'Key': key},
            ExpiresIn=settings.s3_presign_expire_seconds,
        )
        if settings.s3_public_endpoint_url:
            src = urlparse(settings.s3_endpoint_url)
            dst = urlparse(settings.s3_public_endpoint_url)
            parsed = urlparse(presigned)
            if parsed.netloc == src.netloc:
                parsed = parsed._replace(scheme=dst.scheme or parsed.scheme, netloc=dst.netloc or parsed.netloc)
                return urlunparse(parsed)
        return presigned


storage = S3Storage()
