import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  createLlmEnhanceDataset: vi.fn(),
  getLlmEnhanceDatasetResult: vi.fn(),
  getLlmEnhanceDatasetStatus: vi.fn(),
  listYoloDatasets: vi.fn(),
}))

vi.mock('../api/client', () => api)

import LlmPage from '../LlmPage'

describe('LlmPage polling characterization', () => {
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
    api.listYoloDatasets.mockResolvedValue([
      {
        dataset_job_id: 77,
        status: 'succeeded',
        progress: 100,
        width: 640,
        height: 640,
        updated_at: '2026-01-01T00:00:00Z',
        preview_image_urls: [],
        preview_pairs: [],
      },
    ])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('polls enhance job until success and then renders sample items', async () => {
    api.createLlmEnhanceDataset.mockResolvedValue({ enhance_job_id: 321, status: 'queued' })
    api.getLlmEnhanceDatasetStatus
      .mockResolvedValueOnce({ status: 'running', progress: 50 })
      .mockResolvedValueOnce({ status: 'succeeded', progress: 100 })
    api.getLlmEnhanceDatasetResult.mockResolvedValue({
      sample_items: [
        {
          image_url: 'https://example.test/original.png',
          bbox_url: 'https://example.test/bbox.png',
          mask_url: 'https://example.test/mask.png',
          enhanced_image_url: 'https://example.test/enhanced.png',
          enhancement_plan: { llm_model: 'gpt-5.4', image_model: 'gpt-image-1.5', edit_prompt: 'prompt' },
        },
      ],
    })

    render(<LlmPage />)
    await waitFor(() => expect(api.listYoloDatasets).toHaveBeenCalledTimes(1))

    fireEvent.click(screen.getByRole('button', { name: 'Запустить улучшение датасета' }))
    await waitFor(() => expect(api.createLlmEnhanceDataset).toHaveBeenCalledWith({ dataset_job_id: 77 }))

    await waitFor(() => expect(api.getLlmEnhanceDatasetResult).toHaveBeenCalledWith(321))
    expect(screen.getByText('Улучшенные примеры')).toBeInTheDocument()
    expect(screen.getByAltText('Enhanced 1')).toHaveAttribute('src', 'https://example.test/enhanced.png')
    expect(screen.getByRole('button', { name: 'Запустить улучшение датасета' })).toBeEnabled()
  })
})
