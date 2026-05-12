from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models import Asset, AssetKind, Scene
from app.schemas import SceneConfig


def _default_scene_config() -> SceneConfig:
    return SceneConfig()


def _validate_skybox_asset(db: Session, skybox_asset_id: int | None) -> None:
    if skybox_asset_id is None:
        return
    skybox_asset = db.query(Asset).filter(Asset.id == skybox_asset_id).first()
    if skybox_asset is None or skybox_asset.kind != AssetKind.skybox:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Skybox asset not found')


def create_scene(db: Session, object_asset_id: int, environment_asset_id: int, skybox_asset_id: int | None = None) -> Scene:
    object_asset = db.query(Asset).filter(Asset.id == object_asset_id).first()
    environment_asset = db.query(Asset).filter(Asset.id == environment_asset_id).first()

    if object_asset is None or object_asset.kind != AssetKind.object:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Object asset not found')
    if environment_asset is None or environment_asset.kind != AssetKind.environment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Environment asset not found')
    _validate_skybox_asset(db, skybox_asset_id)

    scene_config = _default_scene_config()
    scene_config.skybox_asset_id = skybox_asset_id

    scene = Scene(
        object_asset_id=object_asset_id,
        environment_asset_id=environment_asset_id,
        scene_config=scene_config.model_dump(),
    )
    db.add(scene)
    db.commit()
    db.refresh(scene)
    return scene


def update_scene_config(db: Session, scene_id: int, scene_config: SceneConfig) -> Scene:
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if scene is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Scene not found')

    _validate_skybox_asset(db, scene_config.skybox_asset_id)

    scene.scene_config = scene_config.model_dump()
    db.commit()
    db.refresh(scene)
    return scene
