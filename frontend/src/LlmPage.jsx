import { useEffect, useMemo, useState } from 'react'
import {
  createLlmEnhanceDataset,
  getLlmEnhanceDatasetResult,
  getLlmEnhanceDatasetStatus,
  listYoloDatasets,
} from './api/client'
import { useJobPolling } from './hooks/useJobPolling'
import { getApiErrorMessage, getLlmStatusLabel } from './utils/status'

function formatDate(value) {
  if (!value) return '—'
  return new Date(value).toLocaleString('ru-RU')
}

export default function LlmPage() {
  const [datasets, setDatasets] = useState([])
  const [selectedDatasetId, setSelectedDatasetId] = useState('')
  const [loading, setLoading] = useState(false)
  const [enhanceJobId, setEnhanceJobId] = useState(null)
  const [enhanceStatus, setEnhanceStatus] = useState(null)
  const [enhanceProgress, setEnhanceProgress] = useState(0)
  const [enhanceResultItems, setEnhanceResultItems] = useState([])
  const [enhanceRunning, setEnhanceRunning] = useState(false)
  const [error, setError] = useState('')

  async function reloadDatasets() {
    setLoading(true)
    setError('')
    try {
      const items = await listYoloDatasets(100)
      setDatasets(items)
      if (!items.length) {
        setSelectedDatasetId('')
        return
      }
      setSelectedDatasetId((prev) => {
        if (prev && items.some((item) => String(item.dataset_job_id) === prev)) {
          return prev
        }
        return String(items[0].dataset_job_id)
      })
    } catch (err) {
      setError(getApiErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    reloadDatasets().catch(() => {})
  }, [])

  const selectedDataset = useMemo(
    () => datasets.find((item) => String(item.dataset_job_id) === selectedDatasetId),
    [datasets, selectedDatasetId],
  )

  async function startEnhancement() {
    if (!selectedDataset) return
    setError('')
    setEnhanceResultItems([])
    setEnhanceRunning(true)
    try {
      const created = await createLlmEnhanceDataset({
        dataset_job_id: selectedDataset.dataset_job_id,
      })
      setEnhanceJobId(created.enhance_job_id)
      setEnhanceStatus(created.status)
      setEnhanceProgress(0)
    } catch (err) {
      setError(getApiErrorMessage(err))
      setEnhanceRunning(false)
    }
  }

  useJobPolling({
    jobId: enhanceJobId,
    intervalMs: 3000,
    pollFn: getLlmEnhanceDatasetStatus,
    onPoll: (statusPayload) => {
      setEnhanceStatus(statusPayload.status)
      setEnhanceProgress(statusPayload.progress ?? 0)
    },
    isSuccess: (statusPayload) => statusPayload.status === 'succeeded',
    isFailure: (statusPayload) => statusPayload.status === 'failed',
    onSuccess: async (_statusPayload, currentJobId) => {
      const result = await getLlmEnhanceDatasetResult(currentJobId)
      setEnhanceResultItems(result.sample_items || [])
      setEnhanceRunning(false)
    },
    onFailure: (statusPayload) => {
      setError(statusPayload.error_message || 'Ошибка улучшения датасета')
      setEnhanceRunning(false)
    },
    onError: (err) => {
      setError(getApiErrorMessage(err))
      setEnhanceRunning(false)
    },
  })

  const preview = {
    image_urls: selectedDataset?.preview_image_urls || [],
    pairs: selectedDataset?.preview_pairs || [],
  }

  return (
    <div className="page llm-page">
      <header className="top-nav">
        <a href="/" className="nav-link">Студия</a>
        <a href="/llm" className="nav-link active">LLM</a>
      </header>

      <section className="llm-layout">
        <aside className="panel-block llm-control">
          <h3>Выбор датасета</h3>
          <p>Выберите готовый YOLO датасет для быстрого предпросмотра.</p>

          <label className="upload-label">
            Датасет
            <select
              value={selectedDatasetId}
              disabled={loading || !datasets.length}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
            >
              {!datasets.length ? <option value="">Нет датасетов</option> : null}
              {datasets.map((dataset) => (
                <option key={dataset.dataset_job_id} value={dataset.dataset_job_id}>
                  {`#${dataset.dataset_job_id} · ${getLlmStatusLabel(dataset.status)} · ${dataset.width || '?'}x${dataset.height || '?'}`}
                </option>
              ))}
            </select>
          </label>

          <button type="button" className="secondary-btn" onClick={reloadDatasets} disabled={loading}>
            Обновить список
          </button>
          <button
            type="button"
            className="primary-btn"
            onClick={startEnhancement}
            disabled={!selectedDataset || selectedDataset.status !== 'succeeded' || enhanceRunning}
          >
            {enhanceRunning ? 'Улучшение в процессе...' : 'Запустить улучшение датасета'}
          </button>

          {selectedDataset ? (
            <div className="llm-meta">
              <p><strong>Статус:</strong> {getLlmStatusLabel(selectedDataset.status)}</p>
              <p><strong>Обновлен:</strong> {formatDate(selectedDataset.updated_at)}</p>
              <p><strong>Прогресс:</strong> {selectedDataset.progress}%</p>
            </div>
          ) : null}

          {enhanceJobId ? (
            <div className="llm-meta">
              <p><strong>Enhance job:</strong> #{enhanceJobId}</p>
              <p><strong>Статус:</strong> {getLlmStatusLabel(enhanceStatus)}</p>
              <p><strong>Прогресс:</strong> {enhanceProgress}%</p>
            </div>
          ) : null}

          {error ? <p className="error">{error}</p> : null}
        </aside>

        <main className="panel-block llm-preview">
          {!selectedDataset ? (
            <p>Выберите датасет слева.</p>
          ) : (
            <>
              {selectedDataset.status === 'succeeded' ? (
                preview.pairs.length ? (
                  <div className="llm-pairs-grid">
                    {preview.pairs.map((pair, index) => (
                      <article className="llm-pair-card" key={`${selectedDataset.dataset_job_id}-pair-${index}`}>
                        <h4>{`Пример ${index + 1}`}</h4>
                        <div className="llm-pair-images">
                          <figure>
                            <img src={pair.image_url} alt={`Image ${index + 1}`} />
                            <figcaption>Image</figcaption>
                          </figure>
                          <figure>
                            <img src={pair.bbox_url} alt={`BBox ${index + 1}`} />
                            <figcaption>BBox</figcaption>
                          </figure>
                          <figure>
                            <img src={pair.mask_url} alt={`Mask ${index + 1}`} />
                            <figcaption>Mask</figcaption>
                          </figure>
                        </div>
                      </article>
                    ))}
                  </div>
                ) : preview.image_urls.length ? (
                  <div className="llm-image-grid">
                    {preview.image_urls.map((url, index) => (
                      <img key={`${selectedDataset.dataset_job_id}-${index}`} src={url} alt={`Preview ${index + 1}`} />
                    ))}
                  </div>
                ) : (
                  <p className="dataset-status">Для этого датасета пока нет preview-картинок.</p>
                )
              ) : <p className="dataset-status">Датасет еще генерируется.</p>}

              {enhanceResultItems.length ? (
                <div className="llm-enhanced-block">
                  <h4>Улучшенные примеры</h4>
                  <div className="llm-pairs-grid">
                    {enhanceResultItems.map((item, index) => (
                      <article className="llm-pair-card" key={`enhanced-${index}`}>
                        <h4>{`Enhance ${index + 1}`}</h4>
                        <div className="llm-pair-images llm-pair-images-4">
                          <figure>
                            <img src={item.image_url} alt={`Original ${index + 1}`} />
                            <figcaption>Original</figcaption>
                          </figure>
                          <figure>
                            <img src={item.enhanced_image_url} alt={`Enhanced ${index + 1}`} />
                            <figcaption>Enhanced</figcaption>
                          </figure>
                          <figure>
                            <img src={item.bbox_url} alt={`BBox ${index + 1}`} />
                            <figcaption>BBox</figcaption>
                          </figure>
                          <figure>
                            <img src={item.mask_url} alt={`Mask ${index + 1}`} />
                            <figcaption>Mask</figcaption>
                          </figure>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              ) : null}
            </>
          )}
        </main>
      </section>
    </div>
  )
}
