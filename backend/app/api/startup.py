from sqlalchemy import text

from app.core.config import get_settings
from app.core.db import Base, engine
from app.storage import storage

settings = get_settings()


def _ensure_asset_kind_enum_values() -> None:
    if engine.dialect.name != 'postgresql':
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_enum e
                        JOIN pg_type t ON e.enumtypid = t.oid
                        WHERE t.typname = 'assetkind' AND e.enumlabel = 'skybox'
                    ) THEN
                        ALTER TYPE assetkind ADD VALUE 'skybox';
                    END IF;
                END $$;
                """
            )
        )


def run_startup_checks() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_asset_kind_enum_values()
    try:
        storage.client.head_bucket(Bucket=settings.s3_bucket)
    except Exception:  # noqa: BLE001
        storage.client.create_bucket(Bucket=settings.s3_bucket)
