from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str
    api_prefix: str

    database_url: str
    redis_url: str

    s3_endpoint_url: str
    s3_public_endpoint_url: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str
    s3_bucket: str
    s3_presign_expire_seconds: int

    max_upload_size_bytes: int
    render_width: int
    render_height: int
    render_timeout_seconds: int
    render_require_gpu: bool
    render_samples: int
    render_use_adaptive_sampling: bool
    render_adaptive_threshold: float
    render_use_denoising: bool
    render_denoiser: str
    render_max_bounces: int
    celery_worker_concurrency: int
    dataset_render_parallelism: int


@lru_cache
def get_settings() -> Settings:
    return Settings()
