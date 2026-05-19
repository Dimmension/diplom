import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  createYoloDataset: vi.fn(),
  createRender: vi.fn(),
  createScene: vi.fn(),
  getYoloDatasetResult: vi.fn(),
  getYoloDatasetStatus: vi.fn(),
  getRenderResult: vi.fn(),
  getRenderStatus: vi.fn(),
  listAssets: vi.fn(),
  updateSceneConfig: vi.fn(),
  uploadAsset: vi.fn(),
}))

vi.mock('../api/client', () => api)
vi.mock('../components/SceneViewport', () => ({
  default: () => <div data-testid="scene-viewport" />,
}))

import App from '../App'

async function renderWithAssets() {
  api.listAssets.mockImplementation(async (kind) => {
    if (kind === 'object') {
      return [{ asset_id: 101, filename: 'object.glb', created_at: '2026-01-01T00:00:00Z' }]
    }
    if (kind === 'environment') {
      return [{ asset_id: 202, filename: 'environment.glb', created_at: '2026-01-01T00:00:00Z' }]
    }
    return []
  })

  render(<App />)
  await waitFor(() => expect(api.listAssets).toHaveBeenCalledTimes(3))

  fireEvent.change(screen.getByLabelText('Выбрать объект из загруженных'), { target: { value: '101' } })
  fireEvent.change(screen.getByLabelText('Выбрать окружение из загруженных'), { target: { value: '202' } })
}

describe('App polling characterization', () => {
  beforeEach(() => {
    Object.values(api).forEach((fn) => fn.mockReset())
    vi.spyOn(global, 'setInterval').mockImplementation((callback) => {
      if (typeof callback !== 'function') {
        return 1
      }
      queueMicrotask(async () => {
        await callback()
        await callback()
      })
      return 1
    })
    vi.spyOn(global, 'clearInterval').mockImplementation(() => {})
    api.updateSceneConfig.mockResolvedValue({})
    api.createScene.mockResolvedValue({ scene_id: 55 })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('keeps render polling until succeeded and then fetches result', async () => {
    api.createRender.mockResolvedValue({ render_job_id: 9001, status: 'queued' })
    api.getRenderStatus
      .mockResolvedValueOnce({ status: 'running' })
      .mockResolvedValueOnce({ status: 'succeeded' })
    api.getRenderResult.mockResolvedValue({
      png_url: 'https://example.test/final.png',
      mask_url: 'https://example.test/mask.png',
      bbox_url: 'https://example.test/bbox.png',
    })

    await renderWithAssets()

    fireEvent.click(screen.getByRole('button', { name: 'Рендер 1920x1080' }))
    await waitFor(() => expect(api.createRender).toHaveBeenCalledTimes(1))

    await waitFor(() => expect(api.getRenderResult).toHaveBeenCalledWith(9001))
    expect(screen.getByText('Статус: Готово')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Скачать рендер PNG' })).toHaveAttribute('href', 'https://example.test/final.png')
  })

  it('keeps dataset polling until succeeded and then fetches zip result', async () => {
    api.createYoloDataset.mockResolvedValue({ dataset_job_id: 7002, status: 'queued' })
    api.getYoloDatasetStatus
      .mockResolvedValueOnce({ status: 'running', progress: 38 })
      .mockResolvedValueOnce({ status: 'succeeded', progress: 100 })
    api.getYoloDatasetResult.mockResolvedValue({
      zip_url: 'https://example.test/dataset.zip',
      summary: {},
    })

    await renderWithAssets()

    fireEvent.click(screen.getByRole('button', { name: /YOLO датасет/ }))
    await waitFor(() => expect(api.createYoloDataset).toHaveBeenCalledTimes(1))

    await waitFor(() => expect(api.getYoloDatasetResult).toHaveBeenCalledWith(7002))
    expect(screen.getByText('Статус датасета: Готово')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Скачать датасет ZIP' })).toHaveAttribute('href', 'https://example.test/dataset.zip')
  })
})
