FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG BLENDER_VERSION=4.3.2
ARG BLENDER_SERIES=4.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    xz-utils \
    ca-certificates \
    python3-numpy \
    libsm6 \
    libice6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxi6 \
    libxrandr2 \
    libxxf86vm1 \
    libxfixes3 \
    libxkbcommon0 \
    libfontconfig1 \
    libglib2.0-0 \
    libdbus-1-3 \
    libgl1 \
    libegl1 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/blender.tar.xz \
    https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && ln -sf /opt/blender/blender /usr/local/bin/blender \
    && rm /tmp/blender.tar.xz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN blenderproc run app/scripts/blenderproc_warmup.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
