# 3D Scene Configurator + BlenderProc Render

MVP сервис для загрузки ассетов (`object`, `environment`, optional `skybox`), интерактивной конфигурации сцены в вебе и финального рендера PNG через BlenderProc.

## Что реализовано

- Upload ассетов через веб:
  - `object/environment`: `GLB/GLTF/OBJ/FBX/ZIP`
  - `skybox`: `HDR/EXR`
- Нормализация превью для браузера: все форматы конвертируются в `GLB` (кроме исходного `GLB`)
- Поддержка ZIP-пакетов ассетов:
  - безопасная распаковка (проверка путей/объема),
  - автоматическая распаковка вложенных ZIP (до ограниченной глубины),
  - автоматический выбор entrypoint 3D-файла,
  - сохранение структуры папок (текстуры и связанные файлы),
  - корректный path resolving при рендере.
- WebGL viewport (React Three Fiber):
  - Orbit/Pan/Zoom
  - Gizmo для `translate/rotate/scale`
  - Skybox background (`HDR/EXR`) в viewport
  - Редактирование параметров объекта, окружения и камеры
- Очередь рендера: `FastAPI + Celery + Redis`
- Рендер в воркере через `BlenderProc`:
  - при наличии GPU используется GPU,
  - если GPU недоступен, по умолчанию включается fallback на CPU (`RENDER_REQUIRE_GPU=false`).
  - Скрипт рендера должен начинаться с `import blenderproc as bproc` (требование BlenderProc)
- Polling статуса задачи: `queued`, `running`, `running_background`, `succeeded`, `failed`
- Получение итогового PNG: превью и download URL
- Хранение бинарных файлов: S3-compatible (MinIO)
- Хранение метаданных: PostgreSQL

## Быстрый старт

```bash
docker compose up --build
```

После первого билда можно запускать без пересборки (код монтируется через `volumes`):

```bash
docker compose up
```

Если менялись `backend/Dockerfile` или python/blender зависимости, пересобирайте минимум `api` и `worker`:

```bash
docker compose up -d --build api worker
```

На этапе build backend-образа выполняется warmup `blenderproc`, чтобы managed Blender устанавливался в image, а не в runtime контейнера.
Образ также включает системные shared libraries для prebuilt Blender (`libSM`, X11/GL stack), чтобы избежать runtime/build ошибок вида `libSM.so.6: cannot open shared object file`.

Сервисы:
- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

Логи всех сервисов с датой и временем:

```bash
docker compose logs -f --timestamps
```

## API

- `POST /v1/assets/upload?kind=object|environment|skybox`
- `POST /v1/scenes`
- `PATCH /v1/scenes/{scene_id}/config`
- `POST /v1/renders`
- `GET /v1/renders/{render_job_id}`
- `GET /v1/renders/{render_job_id}/result`

## Ограничения v1

- GPU не обязателен для финального рендера по умолчанию (есть fallback на CPU).
- Для строгого режима GPU-only задайте `RENDER_REQUIRE_GPU=true`.
- Лимит размера файла: `1 GB`
- Финальное изображение: `1920x1080 PNG`
- Таймаут интерактивного режима: `60 сек`.
  - Если превышен, задача не останавливается, статус становится `running_background`, рендер продолжается до конца.

## Переменные окружения

Смотрите `.env.example`.
