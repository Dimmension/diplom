import axios from 'axios'

const apiBase = import.meta.env.VITE_API_BASE
if (!apiBase) {
  throw new Error('VITE_API_BASE is required')
}

const api = axios.create({
  baseURL: apiBase,
})

export async function uploadAsset(kind, file) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post(`/assets/upload?kind=${kind}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function listAssets(kind, limit = 50) {
  const { data } = await api.get('/assets', {
    params: { kind, limit },
  })
  return data.items
}

export async function createScene(payload) {
  const { data } = await api.post('/scenes', payload)
  return data
}

export async function updateSceneConfig(sceneId, sceneConfig) {
  const { data } = await api.patch(`/scenes/${sceneId}/config`, { scene_config: sceneConfig })
  return data
}

export async function createRender(payload) {
  const { data } = await api.post('/renders', payload)
  return data
}

export async function getRenderStatus(renderJobId) {
  const { data } = await api.get(`/renders/${renderJobId}`)
  return data
}

export async function getRenderResult(renderJobId) {
  const { data } = await api.get(`/renders/${renderJobId}/result`)
  return data
}

export async function createYoloDataset(payload) {
  const { data } = await api.post('/datasets/yolo', payload)
  return data
}

export async function listYoloDatasets(limit = 50, status) {
  const { data } = await api.get('/datasets/yolo', {
    params: { limit, status },
  })
  return data.items
}

export async function getYoloDatasetStatus(datasetJobId) {
  const { data } = await api.get(`/datasets/yolo/${datasetJobId}`)
  return data
}

export async function getYoloDatasetResult(datasetJobId) {
  const { data } = await api.get(`/datasets/yolo/${datasetJobId}/result`)
  return data
}

export async function createLlmEnhanceDataset(payload) {
  const { data } = await api.post('/llm/enhance-dataset', payload)
  return data
}

export async function getLlmEnhanceDatasetStatus(enhanceJobId) {
  const { data } = await api.get(`/llm/enhance-dataset/${enhanceJobId}`)
  return data
}

export async function getLlmEnhanceDatasetResult(enhanceJobId) {
  const { data } = await api.get(`/llm/enhance-dataset/${enhanceJobId}/result`)
  return data
}
