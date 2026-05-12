from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery('render_worker', broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    include=['app.services.renders'],
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)
