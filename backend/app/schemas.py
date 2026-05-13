from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AssetKindSchema(str, Enum):
    object = 'object'
    environment = 'environment'
    skybox = 'skybox'


class RenderStatusSchema(str, Enum):
    queued = 'queued'
    running = 'running'
    running_background = 'running_background'
    succeeded = 'succeeded'
    failed = 'failed'


class DatasetStatusSchema(str, Enum):
    queued = 'queued'
    running = 'running'
    succeeded = 'succeeded'
    failed = 'failed'


class Vec3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class CameraConfig(BaseModel):
    position: Vec3 = Field(default_factory=lambda: Vec3(x=3.0, y=-3.0, z=2.0))
    target: Vec3 = Field(default_factory=lambda: Vec3(x=0.0, y=0.0, z=0.0))
    fov_degrees: float = 50.0


class AssetTransform(BaseModel):
    position: Vec3 = Field(default_factory=Vec3)
    rotation: Vec3 = Field(default_factory=Vec3)
    scale: Vec3 = Field(default_factory=lambda: Vec3(x=1.0, y=1.0, z=1.0))


class SceneConfig(BaseModel):
    object_transform: AssetTransform = Field(default_factory=AssetTransform)
    environment_transform: AssetTransform = Field(default_factory=AssetTransform)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    skybox_asset_id: int | None = None


class AssetUploadResponse(BaseModel):
    asset_id: int
    kind: AssetKindSchema
    filename: str
    original_url: str
    preview_glb_url: str | None
    detected_format: str
    size_bytes: int
    created_at: datetime


class AssetListResponse(BaseModel):
    items: list[AssetUploadResponse]


class CreateSceneRequest(BaseModel):
    object_asset_id: int
    environment_asset_id: int
    skybox_asset_id: int | None = None


class SceneResponse(BaseModel):
    scene_id: int
    scene_config: SceneConfig


class UpdateSceneConfigRequest(BaseModel):
    scene_config: SceneConfig


class CreateRenderRequest(BaseModel):
    scene_id: int
    scene_config_snapshot: SceneConfig


class CreateRenderResponse(BaseModel):
    render_job_id: int
    status: RenderStatusSchema


class RenderStatusResponse(BaseModel):
    render_job_id: int
    status: RenderStatusSchema
    progress: int
    started_at: datetime | None
    updated_at: datetime
    error_code: str | None = None
    error_message: str | None = None


class RenderResultResponse(BaseModel):
    png_url: str
    mask_url: str
    bbox_url: str
    scene_config_used: SceneConfig
    scene_config_suggested: SceneConfig


class CreateYoloDatasetRequest(BaseModel):
    scene_id: int
    scene_config_snapshot: SceneConfig
    count: int = Field(default=10, ge=1, le=5000)
    width: int = Field(default=640, ge=64, le=4096)
    height: int = Field(default=640, ge=64, le=4096)
    split_train_count: int = Field(default=8, ge=1)
    split_val_count: int = Field(default=2, ge=1)
    randomization_preset: str = 'medium'
    include_debug: bool = True


class CreateYoloDatasetResponse(BaseModel):
    dataset_job_id: int
    status: DatasetStatusSchema


class YoloDatasetStatusResponse(BaseModel):
    dataset_job_id: int
    status: DatasetStatusSchema
    progress: int
    started_at: datetime | None
    updated_at: datetime
    error_code: str | None = None
    error_message: str | None = None


class YoloDatasetResultSummary(BaseModel):
    images_total: int
    train_count: int
    val_count: int
    empty_labels_count: int
    class_names: list[str]


class YoloDatasetResultResponse(BaseModel):
    zip_url: str
    summary: YoloDatasetResultSummary
