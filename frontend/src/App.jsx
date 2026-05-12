import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createRender,
  createScene,
  getRenderResult,
  getRenderStatus,
  listAssets,
  updateSceneConfig,
  uploadAsset,
} from './api/client'
import SceneViewport from './components/SceneViewport'

const defaultConfig = {
  object_transform: {
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
  },
  environment_transform: {
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
  },
  camera: {
    position: { x: 1, y: 1, z: 1 },
    target: { x: 0, y: 0, z: 0 },
    fov_degrees: 50,
  },
  skybox_asset_id: null,
}

function nearlyEqual(a, b, eps = 0.0001) {
  return Math.abs(a - b) <= eps
}

function sameVec3(a, b) {
  return nearlyEqual(a.x, b.x) && nearlyEqual(a.y, b.y) && nearlyEqual(a.z, b.z)
}

function AxisInput({ label, value, onChange, step = 0.1 }) {
  return (
    <label className="axis-input">
      <span>{label}</span>
      <input type="number" value={value} step={step} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

function BusyDot() {
  return <span className="busy-dot" aria-hidden="true" />
}

function StatusCircle({ state }) {
  const icon = state === 'complete' ? '✓' : state === 'error' ? '!' : ''

  return (
    <span className={`status-circle ${state}`} aria-hidden="true">
      {state === 'active' ? <BusyDot /> : icon}
    </span>
  )
}

function StatusRow({ label, detail, state }) {
  return (
    <div className="status-row">
      <StatusCircle state={state} />
      <div>
        <p className="status-label">{label}</p>
        <p className="status-detail">{detail}</p>
      </div>
    </div>
  )
}

function CollapsibleBlock({ title, open, onToggle, children }) {
  return (
    <section className="panel-block collapsible-block">
      <button type="button" className="collapsible-trigger" onClick={onToggle} aria-expanded={open}>
        <span>{title}</span>
        <span className={`chevron ${open ? 'open' : ''}`}>⌄</span>
      </button>
      {open ? <div className="collapsible-content">{children}</div> : null}
    </section>
  )
}

function TransformEditor({ title, transform, onUpdate }) {
  return (
    <section className="panel-block">
      <h3>{title}</h3>
      <div className="axis-grid">
        <AxisInput label="Поз X" value={transform.position.x} onChange={(x) => onUpdate({ ...transform, position: { ...transform.position, x } })} />
        <AxisInput label="Поз Y" value={transform.position.y} onChange={(y) => onUpdate({ ...transform, position: { ...transform.position, y } })} />
        <AxisInput label="Поз Z" value={transform.position.z} onChange={(z) => onUpdate({ ...transform, position: { ...transform.position, z } })} />
        <AxisInput label="Вращ X" value={transform.rotation.x} onChange={(x) => onUpdate({ ...transform, rotation: { ...transform.rotation, x } })} />
        <AxisInput label="Вращ Y" value={transform.rotation.y} onChange={(y) => onUpdate({ ...transform, rotation: { ...transform.rotation, y } })} />
        <AxisInput label="Вращ Z" value={transform.rotation.z} onChange={(z) => onUpdate({ ...transform, rotation: { ...transform.rotation, z } })} />
        <AxisInput label="Масштаб X" value={transform.scale.x} onChange={(x) => onUpdate({ ...transform, scale: { ...transform.scale, x } })} />
        <AxisInput label="Масштаб Y" value={transform.scale.y} onChange={(y) => onUpdate({ ...transform, scale: { ...transform.scale, y } })} />
        <AxisInput label="Масштаб Z" value={transform.scale.z} onChange={(z) => onUpdate({ ...transform, scale: { ...transform.scale, z } })} />
      </div>
    </section>
  )
}

export default function App() {
  const [objectAsset, setObjectAsset] = useState(null)
  const [environmentAsset, setEnvironmentAsset] = useState(null)
  const [skyboxAsset, setSkyboxAsset] = useState(null)
  const [sceneId, setSceneId] = useState(null)
  const [sceneConfig, setSceneConfig] = useState(defaultConfig)
  const [selectedTarget, setSelectedTarget] = useState('object')
  const [transformMode, setTransformMode] = useState('translate')
  const [renderJobId, setRenderJobId] = useState(null)
  const [renderStatus, setRenderStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [activeTask, setActiveTask] = useState(null)
  const [sidebarHeight, setSidebarHeight] = useState(null)
  const [showParameters, setShowParameters] = useState(true)
  const [assetLibrary, setAssetLibrary] = useState({
    object: [],
    environment: [],
    skybox: [],
  })
  const [assetLibraryLoading, setAssetLibraryLoading] = useState(false)
  const [error, setError] = useState('')
  const sidebarRef = useRef(null)
  const sceneViewportRef = useRef(null)
  const sceneConfigRef = useRef(defaultConfig)

  const busy = activeTask !== null
  const canRender = objectAsset && environmentAsset && !busy

  async function reloadAssetKind(kind) {
    const items = await listAssets(kind, 50)
    setAssetLibrary((prev) => ({ ...prev, [kind]: items }))
  }

  async function reloadAssetLibrary() {
    setAssetLibraryLoading(true)
    try {
      const [objects, environments, skyboxes] = await Promise.all([
        listAssets('object', 50),
        listAssets('environment', 50),
        listAssets('skybox', 50),
      ])
      setAssetLibrary({
        object: objects,
        environment: environments,
        skybox: skyboxes,
      })
    } finally {
      setAssetLibraryLoading(false)
    }
  }

  async function handleUpload(kind, file) {
    setError('')
    setActiveTask(`upload-${kind}`)
    try {
      const asset = await uploadAsset(kind, file)
      if (kind === 'object') {
        setObjectAsset(asset)
      } else if (kind === 'environment') {
        setEnvironmentAsset(asset)
      } else {
        setSkyboxAsset(asset)
        updateSceneConfigLocal((prev) => ({ ...prev, skybox_asset_id: asset.asset_id }))
      }
      await reloadAssetKind(kind)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setActiveTask(null)
    }
  }

  function selectExistingAsset(kind, assetIdRaw) {
    const assetId = Number(assetIdRaw)
    if (!assetId) {
      if (kind === 'object') {
        setObjectAsset(null)
        return
      }
      if (kind === 'environment') {
        setEnvironmentAsset(null)
        return
      }
      if (kind === 'skybox') {
        setSkyboxAsset(null)
        updateSceneConfigLocal((prev) => ({ ...prev, skybox_asset_id: null }))
      }
      return
    }

    const selected = assetLibrary[kind].find((item) => item.asset_id === assetId)
    if (!selected) return

    if (kind === 'object') {
      setObjectAsset(selected)
      return
    }
    if (kind === 'environment') {
      setEnvironmentAsset(selected)
      return
    }

    setSkyboxAsset(selected)
    updateSceneConfigLocal((prev) => ({ ...prev, skybox_asset_id: selected.asset_id }))
  }

  function formatAssetOption(asset) {
    const created = new Date(asset.created_at).toLocaleString('ru-RU')
    return `#${asset.asset_id} ${asset.filename} (${created})`
  }

  function updateSceneConfigLocal(updater) {
    setSceneConfig((prev) => {
      const next = updater(prev)
      sceneConfigRef.current = next
      return next
    })
  }

  async function startRender() {
    if (!objectAsset || !environmentAsset) return
    setActiveTask('render')
    setError('')
    setResult(null)
    try {
      let targetSceneId = sceneId
      if (!targetSceneId) {
        const sceneData = await createScene({
          object_asset_id: objectAsset.asset_id,
          environment_asset_id: environmentAsset.asset_id,
          skybox_asset_id: skyboxAsset?.asset_id ?? null,
        })
        targetSceneId = sceneData.scene_id
        setSceneId(targetSceneId)
        setRenderJobId(null)
        setRenderStatus(null)
      }

      const cameraFromViewport = sceneViewportRef.current?.getCameraTransform?.()
      const configSnapshot = cameraFromViewport
        ? { ...sceneConfigRef.current, camera: cameraFromViewport }
        : sceneConfigRef.current

      sceneConfigRef.current = configSnapshot
      setSceneConfig(configSnapshot)

      await updateSceneConfig(targetSceneId, configSnapshot)
      const data = await createRender({ scene_id: targetSceneId, scene_config_snapshot: configSnapshot })
      setRenderJobId(data.render_job_id)
      setRenderStatus(data.status)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
      setActiveTask(null)
    }
  }

  useEffect(() => {
    if (!renderJobId) return undefined

    const timer = setInterval(async () => {
      try {
        const statusPayload = await getRenderStatus(renderJobId)
        setRenderStatus(statusPayload.status)

        if (statusPayload.status === 'succeeded') {
          const res = await getRenderResult(renderJobId)
          setResult(res)
          setActiveTask(null)
          clearInterval(timer)
        }

        if (statusPayload.status === 'failed') {
          setError(statusPayload.error_message || 'Ошибка рендера')
          setActiveTask(null)
          clearInterval(timer)
        }
      } catch (err) {
        setError(err.response?.data?.detail || err.message)
        setActiveTask(null)
        clearInterval(timer)
      }
    }, 2000)

    return () => clearInterval(timer)
  }, [renderJobId])

  useEffect(() => {
    const sidebarNode = sidebarRef.current
    if (!sidebarNode || typeof ResizeObserver === 'undefined') return undefined

    const updateHeight = () => {
      setSidebarHeight(Math.round(sidebarNode.getBoundingClientRect().height))
    }

    updateHeight()
    const observer = new ResizeObserver(updateHeight)
    observer.observe(sidebarNode)
    window.addEventListener('resize', updateHeight)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', updateHeight)
    }
  }, [])

  useEffect(() => {
    reloadAssetLibrary().catch((err) => {
      setError(err.response?.data?.detail || err.message)
    })
  }, [])

  const statusLabel = useMemo(() => {
    if (!renderStatus) return 'Ожидание'

    if (renderStatus === 'queued') return 'В очереди'
    if (renderStatus === 'running') return 'Выполняется'
    if (renderStatus === 'succeeded') return 'Готово'
    if (renderStatus === 'failed') return 'Ошибка'

    return renderStatus
  }, [renderStatus])

  const workflowRows = useMemo(() => {
    const renderState = renderStatus === 'failed'
      ? 'error'
      : renderStatus === 'succeeded'
        ? 'complete'
        : activeTask === 'render' || renderStatus === 'queued' || renderStatus === 'running'
          ? 'active'
          : 'pending'

    return [
      {
        label: 'Объект загружен',
        detail: objectAsset ? objectAsset.filename : 'Загрузите модель объекта',
        state: activeTask === 'upload-object' ? 'active' : objectAsset ? 'complete' : 'pending',
      },
      {
        label: 'Окружение загружено',
        detail: environmentAsset ? environmentAsset.filename : 'Загрузите модель окружения',
        state: activeTask === 'upload-environment' ? 'active' : environmentAsset ? 'complete' : 'pending',
      },
      {
        label: 'Сцена подготовлена',
        detail: sceneId ? `Сцена ${sceneId}` : 'Сцена создастся автоматически при первом рендере',
        state: sceneId ? 'complete' : activeTask === 'render' ? 'active' : 'pending',
      },
      {
        label: 'Задача рендера',
        detail: renderJobId ? `Статус: ${statusLabel}` : 'Запустите финальный рендер, когда будете готовы',
        state: renderState,
      },
    ]
  }, [
    activeTask,
    environmentAsset,
    objectAsset,
    renderJobId,
    renderStatus,
    sceneId,
    statusLabel,
  ])

  return (
    <div className="page">
      <section className="layout">
        <main className="main">
          <div className="viewport-wrap" style={sidebarHeight ? { height: `${sidebarHeight}px` } : undefined}>
            <SceneViewport
              ref={sceneViewportRef}
              objectPreviewUrl={objectAsset?.preview_glb_url}
              environmentPreviewUrl={environmentAsset?.preview_glb_url}
              skyboxUrl={skyboxAsset?.original_url}
              sceneConfig={sceneConfig}
              selectedTarget={selectedTarget}
              transformMode={transformMode}
              onObjectTransform={(transform) => {
                updateSceneConfigLocal((prev) => ({ ...prev, object_transform: transform }))
              }}
              onEnvironmentTransform={(transform) => {
                updateSceneConfigLocal((prev) => ({ ...prev, environment_transform: transform }))
              }}
              onCameraTransform={(camera) => {
                updateSceneConfigLocal((prev) => {
                  const prevCamera = prev.camera
                  const isSame =
                    sameVec3(prevCamera.position, camera.position)
                    && sameVec3(prevCamera.target, camera.target)
                    && nearlyEqual(prevCamera.fov_degrees, camera.fov_degrees, 0.01)
                  if (isSame) return prev
                  return { ...prev, camera }
                })
              }}
            />
            <div className="viewport-editor-controls">
              <div className="viewport-controls-group">
                <p className="viewport-controls-title">Редактируем</p>
                <div className="button-row viewport-controls-row">
                  <button className={`target-btn ${selectedTarget === 'object' ? 'active' : ''}`} onClick={() => setSelectedTarget('object')}>Объект</button>
                  <button className={`target-btn ${selectedTarget === 'environment' ? 'active' : ''}`} onClick={() => setSelectedTarget('environment')}>Окружение</button>
                </div>
              </div>
              <div className="viewport-controls-group">
                <p className="viewport-controls-title">Режим</p>
                <div className="button-row viewport-controls-row">
                  <button className={`mode-btn ${transformMode === 'translate' ? 'active' : ''}`} onClick={() => setTransformMode('translate')}>Перемещение</button>
                  <button className={`mode-btn ${transformMode === 'rotate' ? 'active' : ''}`} onClick={() => setTransformMode('rotate')}>Поворот</button>
                  <button className={`mode-btn ${transformMode === 'scale' ? 'active' : ''}`} onClick={() => setTransformMode('scale')}>Масштаб</button>
                </div>
              </div>
            </div>
          </div>

          <section className="parameters-wrap">
            <div className="parameters-toolbar">
              <button type="button" className="secondary-btn" onClick={() => setShowParameters((prev) => !prev)}>
                {showParameters ? 'Скрыть параметры' : 'Показать параметры'}
              </button>
            </div>

            {showParameters ? (
              <>
                <TransformEditor
                  title="Объект"
                  transform={sceneConfig.object_transform}
                  onUpdate={(value) => updateSceneConfigLocal((prev) => ({ ...prev, object_transform: value }))}
                />

                <TransformEditor
                  title="Окружение"
                  transform={sceneConfig.environment_transform}
                  onUpdate={(value) => updateSceneConfigLocal((prev) => ({ ...prev, environment_transform: value }))}
                />

                <section className="panel-block">
                  <h3>Камера</h3>
                  <div className="axis-grid">
                    <AxisInput
                      label="Кам X"
                      value={sceneConfig.camera.position.x}
                      onChange={(x) => updateSceneConfigLocal((prev) => ({
                        ...prev,
                        camera: { ...prev.camera, position: { ...prev.camera.position, x } },
                      }))}
                    />
                    <AxisInput
                      label="Кам Y"
                      value={sceneConfig.camera.position.y}
                      onChange={(y) => updateSceneConfigLocal((prev) => ({
                        ...prev,
                        camera: { ...prev.camera, position: { ...prev.camera.position, y } },
                      }))}
                    />
                    <AxisInput
                      label="Кам Z"
                      value={sceneConfig.camera.position.z}
                      onChange={(z) => updateSceneConfigLocal((prev) => ({
                        ...prev,
                        camera: { ...prev.camera, position: { ...prev.camera.position, z } },
                      }))}
                    />
                    <AxisInput
                      label="Цель X"
                      value={sceneConfig.camera.target.x}
                      onChange={(x) => updateSceneConfigLocal((prev) => ({
                        ...prev,
                        camera: { ...prev.camera, target: { ...prev.camera.target, x } },
                      }))}
                    />
                    <AxisInput
                      label="Цель Y"
                      value={sceneConfig.camera.target.y}
                      onChange={(y) => updateSceneConfigLocal((prev) => ({
                        ...prev,
                        camera: { ...prev.camera, target: { ...prev.camera.target, y } },
                      }))}
                    />
                    <AxisInput
                      label="Цель Z"
                      value={sceneConfig.camera.target.z}
                      onChange={(z) => updateSceneConfigLocal((prev) => ({
                        ...prev,
                        camera: { ...prev.camera, target: { ...prev.camera.target, z } },
                      }))}
                    />
                    <AxisInput
                      label="Угол обзора"
                      value={sceneConfig.camera.fov_degrees}
                      step={1}
                      onChange={(fov_degrees) => updateSceneConfigLocal((prev) => ({
                        ...prev,
                        camera: { ...prev.camera, fov_degrees },
                      }))}
                    />
                  </div>
                </section>
              </>
            ) : null}
          </section>

          {activeTask === 'render' || renderJobId || result ? (
            <div className="result-wrap">
              {result ? (
                <>
                  <img src={result.png_url} alt="Результат рендера" />
                  <a href={result.png_url} target="_blank" rel="noreferrer">Открыть PNG</a>
                </>
              ) : null}
            </div>
          ) : null}
        </main>

        <aside className="sidebar" ref={sidebarRef}>
          <div className="panel-block">
            <h3>Процесс</h3>
            <div className="status-list">
              {workflowRows.map((row) => (
                <StatusRow key={row.label} label={row.label} detail={row.detail} state={row.state} />
              ))}
            </div>
            {error ? <p className="error">{error}</p> : null}
          </div>

          <div className="panel-block">
            <h3>Ассеты</h3>
            <label className="upload-label">
              Выбрать объект из загруженных
              <select
                disabled={busy || assetLibraryLoading}
                value={objectAsset?.asset_id ?? ''}
                onChange={(e) => selectExistingAsset('object', e.target.value)}
              >
                <option value="">Выберите объект</option>
                {assetLibrary.object.map((asset) => (
                  <option key={asset.asset_id} value={asset.asset_id}>
                    {formatAssetOption(asset)}
                  </option>
                ))}
              </select>
            </label>
            <label className="upload-label">
              Объект (.glb/.gltf/.obj/.fbx/.zip)
              <input type="file" accept=".glb,.gltf,.obj,.fbx,.zip" disabled={busy} onChange={(e) => e.target.files?.[0] && handleUpload('object', e.target.files[0])} />
            </label>
            <label className="upload-label">
              Выбрать окружение из загруженных
              <select
                disabled={busy || assetLibraryLoading}
                value={environmentAsset?.asset_id ?? ''}
                onChange={(e) => selectExistingAsset('environment', e.target.value)}
              >
                <option value="">Выберите окружение</option>
                {assetLibrary.environment.map((asset) => (
                  <option key={asset.asset_id} value={asset.asset_id}>
                    {formatAssetOption(asset)}
                  </option>
                ))}
              </select>
            </label>
            <label className="upload-label">
              Окружение (.glb/.gltf/.obj/.fbx/.zip)
              <input type="file" accept=".glb,.gltf,.obj,.fbx,.zip" disabled={busy} onChange={(e) => e.target.files?.[0] && handleUpload('environment', e.target.files[0])} />
            </label>
            <label className="upload-label">
              Выбрать скайбокс из загруженных
              <select
                disabled={busy || assetLibraryLoading}
                value={skyboxAsset?.asset_id ?? ''}
                onChange={(e) => selectExistingAsset('skybox', e.target.value)}
              >
                <option value="">Без скайбокса</option>
                {assetLibrary.skybox.map((asset) => (
                  <option key={asset.asset_id} value={asset.asset_id}>
                    {formatAssetOption(asset)}
                  </option>
                ))}
              </select>
            </label>
            <label className="upload-label">
              Скайбокс (.hdr/.exr, не обязательно)
              <input type="file" accept=".hdr,.exr" disabled={busy} onChange={(e) => e.target.files?.[0] && handleUpload('skybox', e.target.files[0])} />
            </label>
            <div className="action-column">
              <button className="primary-btn" disabled={!canRender} onClick={startRender}>
                {activeTask === 'render' ? <BusyDot /> : null}
                Рендер 1920x1080
              </button>
            </div>
          </div>

        </aside>
      </section>
    </div>
  )
}
