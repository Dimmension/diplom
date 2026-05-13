import logging
import os

from celery.signals import worker_ready

from app.services.render_queue import celery_app

logger = logging.getLogger(__name__)


@worker_ready.connect
def validate_gpu_on_worker_start(**_: object) -> None:
    from app.services.renders import GpuUnavailableError, _run_gpu_probe

    try:
        _run_gpu_probe()
        logger.info('GPU preflight succeeded: worker is ready for strict GPU renders')
    except GpuUnavailableError as err:
        logger.error('GPU preflight failed during worker startup: %s', err)
        os._exit(1)

__all__ = ('celery_app',)
