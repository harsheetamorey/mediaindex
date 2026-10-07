import { useEffect, useRef, useState } from 'react'
import { api, ApiError, type ClipPlan, type Job, type Result } from '../api'
import { fmtSpan, fmtTime } from '../lib/time'

type Props = {
  result: Result
  queryContext: Record<string, unknown> | null
  onFindSounds: (start: number) => void
  onFindSimilarMoments: (start: number) => void
}

/** Video preview that seeks to the matched window, plays only that range, and exports a clip. */
export default function VideoMoment({ result, queryContext, onFindSounds, onFindSimilarMoments }: Props) {
  const a = result.asset
  const video = useRef<HTMLVideoElement>(null)
  const stopAt = useRef<number | null>(null)
  const [range, setRange] = useState<[number, number]>([result.start_s ?? 0, result.end_s ?? a.duration ?? 0])
  const [playingRange, setPlayingRange] = useState(false)

  const playRange = (s: number, e: number) => {
    const v = video.current
    if (!v) return
    v.currentTime = s
    stopAt.current = e
    setPlayingRange(true)
    v.play().catch(() => setPlayingRange(false))
  }

  useEffect(() => {
    const v = video.current
    if (!v) return
    const onMeta = () => {
      v.currentTime = result.start_s ?? 0
    }
    const onTime = () => {
      if (stopAt.current != null && v.currentTime >= stopAt.current) {
        v.pause()
        stopAt.current = null
        setPlayingRange(false)
      }
    }
    const onPause = () => setPlayingRange(false)
    v.addEventListener('loadedmetadata', onMeta)
    v.addEventListener('timeupdate', onTime)
    v.addEventListener('pause', onPause)
    return () => {
      v.removeEventListener('loadedmetadata', onMeta)
      v.removeEventListener('timeupdate', onTime)
      v.removeEventListener('pause', onPause)
    }
  }, [result.start_s])

  const moments = [{ start_s: result.start_s, end_s: result.end_s, window_thumbnail_url: result.window_thumbnail_url }, ...(result.other_segments ?? [])]

  return (
    <div className="video-moment">
      <video ref={video} id="mi-video-player" src={a.file_url} controls preload="metadata" />
      <div className="seg-bar" aria-hidden="true">
        {a.duration ? (
          <div className="seg-mark" style={{ left: `${(100 * range[0]) / a.duration}%`, width: `${(100 * (range[1] - range[0])) / a.duration}%` }} />
        ) : null}
      </div>
      <div className="detail-actions">
        <button className="btn primary" onClick={() => playRange(result.start_s ?? 0, result.end_s ?? 0)}>
          {playingRange ? 'Playing…' : `▶ Play matched moment ${fmtSpan(result.start_s, result.end_s)}`}
        </button>
        <button className="btn" onClick={() => onFindSounds(result.start_s ?? 0)} title="Experimental: similarity candidates, not synchronized sound">
          Find sounds for this moment <span className="badge exp">experimental</span>
        </button>
        <button className="btn" onClick={() => onFindSimilarMoments(result.start_s ?? 0)}>Similar moments in other videos</button>
      </div>
      {moments.length > 1 && (
        <div className="moments">
          <span className="muted small">Other matching moments (overlapping windows merged):</span>
          <div className="moment-list">
            {moments.slice(1).map((m) => (
              <button key={String(m.start_s)} className="moment" onClick={() => playRange(m.start_s ?? 0, m.end_s ?? 0)}>
                {m.window_thumbnail_url && <img src={m.window_thumbnail_url} alt="" />}
                <span>{fmtSpan(m.start_s, m.end_s)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      <p className="muted small">
        Moments come from fixed {fmtTime((result.end_s ?? 0) - (result.start_s ?? 0))} index windows sampled at 1 frame per second. They show
        roughly where a match is, not exact event boundaries.
      </p>
      <ClipExport assetId={a.id} range={range} setRange={setRange} duration={a.duration ?? 0} onPreview={playRange} queryContext={queryContext} />
    </div>
  )
}

function ClipExport({
  assetId,
  range,
  setRange,
  duration,
  onPreview,
  queryContext,
}: {
  assetId: string
  range: [number, number]
  setRange: (r: [number, number]) => void
  duration: number
  onPreview: (s: number, e: number) => void
  queryContext: Record<string, unknown> | null
}) {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<'accurate' | 'fast'>('accurate')
  const [dest, setDest] = useState('')
  const [plan, setPlan] = useState<ClipPlan | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return
    const t = setInterval(async () => setJob(await api.job(job.id)), 500)
    return () => clearInterval(t)
  }, [job])

  if (!open)
    return (
      <button className="btn" onClick={() => setOpen(true)}>
        Export clip…
      </button>
    )

  const num = (v: string, fallback: number) => (isFinite(parseFloat(v)) ? parseFloat(v) : fallback)
  const result = job?.result as { clip: string; manifest: string; output_duration_s: number } | null | undefined

  return (
    <section className="export-panel" aria-label="Clip export">
      <h3>Export a clip</h3>
      <div className="clip-row">
        <label>
          Start (s)
          <input type="number" min={0} max={duration} step={0.1} value={range[0]} onChange={(e) => (setRange([num(e.target.value, range[0]), range[1]]), setPlan(null))} />
        </label>
        <label>
          End (s)
          <input type="number" min={0} max={duration} step={0.1} value={range[1]} onChange={(e) => (setRange([range[0], num(e.target.value, range[1])]), setPlan(null))} />
        </label>
        <button className="btn small" onClick={() => onPreview(range[0], range[1])}>▶ Preview range</button>
      </div>
      <fieldset className="clip-mode">
        <label className="check small">
          <input type="radio" name="clip-mode" checked={mode === 'accurate'} onChange={() => (setMode('accurate'), setPlan(null))} />
          Accurate (re-encode, starts exactly at the start time)
        </label>
        <label className="check small">
          <input type="radio" name="clip-mode" checked={mode === 'fast'} onChange={() => (setMode('fast'), setPlan(null))} />
          Fast (stream copy, no quality loss, starts at the nearest earlier keyframe)
        </label>
      </fieldset>
      <div className="query-row">
        <input type="search" aria-label="Clip destination folder" placeholder="/Users/you/Desktop/clips" value={dest} onChange={(e) => (setDest(e.target.value), setPlan(null))} spellCheck={false} />
        <button
          className="btn"
          disabled={!dest.trim()}
          onClick={async () => {
            setError(null)
            setJob(null)
            try {
              setPlan(await api.clipPreview(assetId, range[0], range[1], mode, dest.trim(), queryContext))
            } catch (e) {
              setPlan(null)
              setError(e instanceof ApiError ? e.message : String(e))
            }
          }}
        >
          Preview export
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {plan && !job && (
        <div className="plan">
          <p>
            Will write <code>{plan.dest_name}</code> ({fmtSpan(plan.start_s, plan.end_s)}, {plan.duration_s.toFixed(1)} s) and{' '}
            <code>{plan.manifest_name}</code> into <code>{plan.destination}</code>. Overwrites: {plan.overwrites}.
          </p>
          {plan.mode === 'fast' && plan.effective_start_s < plan.start_s - 0.01 && (
            <p className="warn-text small">
              Fast mode will start at {fmtTime(plan.effective_start_s)} (keyframe), {(plan.start_s - plan.effective_start_s).toFixed(2)} s early.
            </p>
          )}
          <div className="detail-actions">
            <button className="btn primary" onClick={async () => setJob(await api.clipExport(assetId, plan.plan_id))}>
              Export clip
            </button>
            <button className="btn" onClick={() => setPlan(null)}>Cancel</button>
          </div>
        </div>
      )}
      {job && (job.status === 'queued' || job.status === 'running') && (
        <div className="progress">
          <span>Rendering clip…</span>
          <button className="link" onClick={async () => setJob(await api.cancelJob(job.id))}>Stop</button>
        </div>
      )}
      {job && job.status === 'done' && result && (
        <div className="banner ok" role="status">
          Exported <code>{result.clip}</code> ({result.output_duration_s.toFixed(2)} s) with <code>{result.manifest}</code>.
        </div>
      )}
      {job && (job.status === 'failed' || job.status === 'cancelled') && (
        <p className="error">Clip export {job.status}{job.error ? `: ${job.error}` : ''}. No partial file was left behind.</p>
      )}
    </section>
  )
}
