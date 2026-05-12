import React, { Suspense, forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Environment, Grid, OrbitControls, TransformControls, useGLTF } from '@react-three/drei'
import * as THREE from 'three'

function toDeg(rad) {
  return (rad * 180) / Math.PI
}

function toVec3Rounded(vector) {
  return {
    x: Number(vector.x.toFixed(4)),
    y: Number(vector.y.toFixed(4)),
    z: Number(vector.z.toFixed(4)),
  }
}

function getWebGLUnavailableMessage() {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return 'WebGL недоступен в этой среде.'
  }

  const canvas = document.createElement('canvas')
  const webgl2 = canvas.getContext('webgl2')
  if (webgl2) return null

  const webgl1 = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
  if (webgl1) {
    return 'Для текущего 3D-движка нужен WebGL2, но в этом браузере/устройстве доступен только WebGL1.'
  }

  return 'Браузер или GPU не предоставляют рабочий WebGL-контекст.'
}

function GLBModel({
  url,
  transform,
  selected,
  mode,
  onTransformChange,
  onDraggingChange,
  orbitRef,
}) {
  const { scene } = useGLTF(url)
  const root = useMemo(() => scene.clone(true), [scene])
  const groupRef = useRef(null)
  const [groupMounted, setGroupMounted] = useState(false)

  const setGroupNode = useCallback((node) => {
    groupRef.current = node
    setGroupMounted(Boolean(node))
  }, [])

  useEffect(() => {
    if (!groupRef.current) return
    groupRef.current.position.set(transform.position.x, transform.position.y, transform.position.z)
    groupRef.current.rotation.set(
      THREE.MathUtils.degToRad(transform.rotation.x),
      THREE.MathUtils.degToRad(transform.rotation.y),
      THREE.MathUtils.degToRad(transform.rotation.z),
    )
    groupRef.current.scale.set(transform.scale.x, transform.scale.y, transform.scale.z)
  }, [transform])

  const emitTransform = useCallback(() => {
    if (!groupRef.current) return
    onTransformChange({
      position: toVec3Rounded(groupRef.current.position),
      rotation: {
        x: Number(toDeg(groupRef.current.rotation.x).toFixed(3)),
        y: Number(toDeg(groupRef.current.rotation.y).toFixed(3)),
        z: Number(toDeg(groupRef.current.rotation.z).toFixed(3)),
      },
      scale: toVec3Rounded(groupRef.current.scale),
    })
  }, [onTransformChange])

  const node = (
    <group ref={setGroupNode}>
      <primitive object={root} />
    </group>
  )

  if (!selected || !groupMounted) return node

  return (
    <>
      {node}
      <TransformControls
        object={groupRef.current}
        controls={orbitRef?.current || undefined}
        mode={mode}
        onObjectChange={emitTransform}
        onDraggingChanged={(event) => onDraggingChange(Boolean(event?.value))}
      />
    </>
  )
}

class ViewportErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Не удалось загрузить 3D-превью' }
  }

  render() {
    if (this.state.hasError) {
      return <div className="viewport-error">Ошибка 3D-превью: {this.state.message}</div>
    }
    return this.props.children
  }
}

const SceneViewport = forwardRef(function SceneViewport({
  objectPreviewUrl,
  environmentPreviewUrl,
  skyboxUrl,
  sceneConfig,
  selectedTarget,
  transformMode,
  onObjectTransform,
  onEnvironmentTransform,
  onCameraTransform,
}, ref) {
  const orbitRef = useRef(null)
  const [isDraggingTransform, setIsDraggingTransform] = useState(false)
  const webglUnavailableMessage = useMemo(() => getWebGLUnavailableMessage(), [])

  useEffect(() => {
    setIsDraggingTransform(false)
    if (orbitRef.current) {
      orbitRef.current.enabled = true
    }
  }, [selectedTarget, transformMode])

  useEffect(() => {
    const releaseOrbit = () => {
      setIsDraggingTransform(false)
      if (orbitRef.current) {
        orbitRef.current.enabled = true
      }
    }

    window.addEventListener('pointerup', releaseOrbit)
    window.addEventListener('mouseup', releaseOrbit)
    window.addEventListener('blur', releaseOrbit)
    window.addEventListener('mouseleave', releaseOrbit)
    return () => {
      window.removeEventListener('pointerup', releaseOrbit)
      window.removeEventListener('mouseup', releaseOrbit)
      window.removeEventListener('blur', releaseOrbit)
      window.removeEventListener('mouseleave', releaseOrbit)
    }
  }, [])

  function CameraSync({ cameraConfig }) {
    useEffect(() => {
      const controls = orbitRef.current
      if (!controls) return
      controls.object.position.set(
        cameraConfig.position.x,
        cameraConfig.position.y,
        cameraConfig.position.z,
      )
      controls.object.fov = cameraConfig.fov_degrees
      controls.object.updateProjectionMatrix()
      controls.target.set(cameraConfig.target.x, cameraConfig.target.y, cameraConfig.target.z)
      controls.update()
    }, [cameraConfig])

    return null
  }

  const syncCameraToPanel = useCallback(() => {
    const controls = orbitRef.current
    if (!controls) return
    onCameraTransform({
      position: toVec3Rounded(controls.object.position),
      target: toVec3Rounded(controls.target),
      fov_degrees: Number(controls.object.fov.toFixed(2)),
    })
  }, [onCameraTransform])

  const getCameraTransform = useCallback(() => {
    const controls = orbitRef.current
    if (!controls) return null
    return {
      position: toVec3Rounded(controls.object.position),
      target: toVec3Rounded(controls.target),
      fov_degrees: Number(controls.object.fov.toFixed(2)),
    }
  }, [])

  useImperativeHandle(ref, () => ({
    getCameraTransform,
  }), [getCameraTransform])

  const fallbackNode = (
    <div className="viewport-error">
      <p>3D-превью недоступно: {webglUnavailableMessage || 'WebGL не поддерживается.'}</p>
      <p className="viewport-error-hint">
        Попробуйте включить аппаратное ускорение, обновить драйверы GPU или открыть страницу в более новом браузере с WebGL2.
      </p>
    </div>
  )

  const createRenderer = useCallback((rendererConfig) => {
    const contextAttributes = {
      alpha: true,
      antialias: false,
      depth: true,
      failIfMajorPerformanceCaveat: false,
      powerPreference: 'high-performance',
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
      stencil: false,
    }

    const canvas = rendererConfig?.canvas ?? rendererConfig
    if (!canvas || typeof canvas.getContext !== 'function') {
      throw new Error('Canvas недоступен для инициализации WebGL')
    }

    const context = canvas.getContext('webgl2', contextAttributes)
    if (!context) {
      throw new Error('Не удалось инициализировать контекст рендера WebGL2')
    }

    const rendererOptions = rendererConfig?.canvas
      ? rendererConfig
      : { canvas }

    const renderer = new THREE.WebGLRenderer({
      ...rendererOptions,
      context,
    })
    renderer.outputColorSpace = THREE.SRGBColorSpace
    return renderer
  }, [])

  if (webglUnavailableMessage) {
    return fallbackNode
  }

  return (
    <ViewportErrorBoundary>
      <Canvas camera={{ position: [3, -3, 2], fov: 50 }} gl={createRenderer} fallback={fallbackNode}>
        <color attach="background" args={['#0e1117']} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[5, 5, 5]} intensity={1.2} />
        <Grid
          args={[200, 200]}
          cellColor="#3b4252"
          sectionColor="#4c566a"
          infiniteGrid
          fadeDistance={400}
          fadeStrength={1}
        />
        <OrbitControls
          ref={orbitRef}
          makeDefault
          enablePan
          enableRotate
          enableZoom
          enabled={!isDraggingTransform}
          onChange={syncCameraToPanel}
        />
        <CameraSync cameraConfig={sceneConfig.camera} />

        <Suspense fallback={null}>
          {skyboxUrl ? <Environment files={skyboxUrl} background /> : null}

          {environmentPreviewUrl ? (
            <GLBModel
              url={environmentPreviewUrl}
              transform={sceneConfig.environment_transform}
              selected={selectedTarget === 'environment'}
              mode={transformMode}
              onTransformChange={onEnvironmentTransform}
              onDraggingChange={setIsDraggingTransform}
              orbitRef={orbitRef}
            />
          ) : null}

          {objectPreviewUrl ? (
            <GLBModel
              url={objectPreviewUrl}
              transform={sceneConfig.object_transform}
              selected={selectedTarget === 'object'}
              mode={transformMode}
              onTransformChange={onObjectTransform}
              onDraggingChange={setIsDraggingTransform}
              orbitRef={orbitRef}
            />
          ) : null}
        </Suspense>
      </Canvas>
    </ViewportErrorBoundary>
  )
})

export default SceneViewport
