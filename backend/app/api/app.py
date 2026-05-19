from fastapi import FastAPI

from app.api.routes import assets, datasets, health, llm, renders, scenes
from app.api.startup import run_startup_checks
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.on_event('startup')
def startup() -> None:
    run_startup_checks()


app.include_router(health.router)
app.include_router(assets.router, prefix=settings.api_prefix)
app.include_router(scenes.router, prefix=settings.api_prefix)
app.include_router(renders.router, prefix=settings.api_prefix)
app.include_router(datasets.router, prefix=settings.api_prefix)
app.include_router(llm.router, prefix=settings.api_prefix)
