import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class AssetKind(str, enum.Enum):
    object = 'object'
    environment = 'environment'
    skybox = 'skybox'


class RenderStatus(str, enum.Enum):
    queued = 'queued'
    running = 'running'
    running_background = 'running_background'
    succeeded = 'succeeded'
    failed = 'failed'


class DatasetStatus(str, enum.Enum):
    queued = 'queued'
    running = 'running'
    succeeded = 'succeeded'
    failed = 'failed'


class Asset(Base):
    __tablename__ = 'assets'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    kind: Mapped[AssetKind] = mapped_column(Enum(AssetKind), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    detected_format: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    original_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    preview_glb_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Scene(Base):
    __tablename__ = 'scenes'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    object_asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'), nullable=False)
    environment_asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'), nullable=False)
    scene_config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    object_asset: Mapped[Asset] = relationship('Asset', foreign_keys=[object_asset_id])
    environment_asset: Mapped[Asset] = relationship('Asset', foreign_keys=[environment_asset_id])


class RenderJob(Base):
    __tablename__ = 'render_jobs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey('scenes.id'), nullable=False)
    status: Mapped[RenderStatus] = mapped_column(Enum(RenderStatus), nullable=False, default=RenderStatus.queued)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    scene_config_used: Mapped[dict] = mapped_column(JSONB, nullable=False)
    scene_config_suggested: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    scene: Mapped[Scene] = relationship('Scene')


class YoloDatasetJob(Base):
    __tablename__ = 'yolo_dataset_jobs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey('scenes.id'), nullable=False)
    status: Mapped[DatasetStatus] = mapped_column(Enum(DatasetStatus), nullable=False, default=DatasetStatus.queued)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    scene: Mapped[Scene] = relationship('Scene')


class LlmEnhanceJob(Base):
    __tablename__ = 'llm_enhance_jobs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_job_id: Mapped[int] = mapped_column(ForeignKey('yolo_dataset_jobs.id'), nullable=False)
    status: Mapped[DatasetStatus] = mapped_column(Enum(DatasetStatus), nullable=False, default=DatasetStatus.queued)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    dataset_job: Mapped[YoloDatasetJob] = relationship('YoloDatasetJob')
