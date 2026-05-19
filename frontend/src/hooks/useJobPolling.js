import { useEffect, useRef } from 'react'

export function useJobPolling({
  jobId,
  intervalMs,
  pollFn,
  onPoll,
  onSuccess,
  onFailure,
  onError,
  isSuccess,
  isFailure,
}) {
  const pollFnRef = useRef(pollFn)
  const onPollRef = useRef(onPoll)
  const onSuccessRef = useRef(onSuccess)
  const onFailureRef = useRef(onFailure)
  const onErrorRef = useRef(onError)
  const isSuccessRef = useRef(isSuccess)
  const isFailureRef = useRef(isFailure)

  useEffect(() => {
    pollFnRef.current = pollFn
    onPollRef.current = onPoll
    onSuccessRef.current = onSuccess
    onFailureRef.current = onFailure
    onErrorRef.current = onError
    isSuccessRef.current = isSuccess
    isFailureRef.current = isFailure
  }, [pollFn, onPoll, onSuccess, onFailure, onError, isSuccess, isFailure])

  useEffect(() => {
    if (!jobId) return undefined

    let cancelled = false
    const timer = setInterval(async () => {
      try {
        const payload = await pollFnRef.current(jobId)
        if (cancelled) return

        onPollRef.current?.(payload, jobId)

        if (isSuccessRef.current(payload)) {
          await onSuccessRef.current?.(payload, jobId)
          clearInterval(timer)
          return
        }

        if (isFailureRef.current(payload)) {
          onFailureRef.current?.(payload, jobId)
          clearInterval(timer)
        }
      } catch (err) {
        if (cancelled) return
        onErrorRef.current?.(err, jobId)
        clearInterval(timer)
      }
    }, intervalMs)

    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [jobId, intervalMs])
}
