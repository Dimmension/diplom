from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = '3D Render Service'
    api_prefix: str = '/v1'

    database_url: str = 'postgresql+psycopg2://postgres:postgres@postgres:5432/renderdb'
    redis_url: str = 'redis://redis:6379/0'

    s3_endpoint_url: str = 'http://minio:9000'
    s3_public_endpoint_url: str = 'http://localhost:9000'
    s3_access_key: str = 'minioadmin'
    s3_secret_key: str = 'minioadmin'
    s3_region: str = 'us-east-1'
    s3_bucket: str = 'render-assets'
    s3_presign_expire_seconds: int = 3600

    max_upload_size_bytes: int = 1_000_000_000
    render_width: int = 1920
    render_height: int = 1080
    render_timeout_seconds: int = 60
    render_require_gpu: bool = False
    celery_worker_concurrency: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
