from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas import CreateSceneRequest, SceneResponse, UpdateSceneConfigRequest
from app.services.scenes import create_scene, update_scene_config

router = APIRouter()


@router.post('/scenes', response_model=SceneResponse)
def create_scene_endpoint(payload: CreateSceneRequest, db: Session = Depends(get_db)) -> SceneResponse:
    scene = create_scene(db, payload.object_asset_id, payload.environment_asset_id, payload.skybox_asset_id)
    return SceneResponse(scene_id=scene.id, scene_config=scene.scene_config)


@router.patch('/scenes/{scene_id}/config', response_model=SceneResponse)
def update_scene_config_endpoint(scene_id: int, payload: UpdateSceneConfigRequest, db: Session = Depends(get_db)) -> SceneResponse:
    scene = update_scene_config(db, scene_id, payload.scene_config)
    return SceneResponse(scene_id=scene.id, scene_config=scene.scene_config)
