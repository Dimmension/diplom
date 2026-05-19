const STATUS_LABELS = {
  queued: 'В очереди',
  running: 'Выполняется',
  succeeded: 'Готово',
  failed: 'Ошибка',
}

const DATASET_STATUS_LABELS = {
  queued: 'В очереди',
  running: 'Генерация',
  succeeded: 'Готово',
  failed: 'Ошибка',
}

const LLM_STATUS_LABELS = {
  queued: 'В очереди',
  running: 'Генерируется',
  succeeded: 'Готов',
  failed: 'Ошибка',
}

export function getRenderStatusLabel(status) {
  if (!status) return 'Ожидание'
  return STATUS_LABELS[status] || status
}

export function getDatasetStatusLabel(status) {
  if (!status) return 'Ожидание'
  return DATASET_STATUS_LABELS[status] || status
}

export function getLlmStatusLabel(status) {
  if (!status) return 'Неизвестно'
  return LLM_STATUS_LABELS[status] || status
}

export function getApiErrorMessage(err) {
  return err?.response?.data?.detail || err?.message || 'Неизвестная ошибка'
}
