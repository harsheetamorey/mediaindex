import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, type ExportPlan, type Job, type Selection, type SelectionItem } from '../api'

type Props = {
  selectionId: string
  onBack: () => void
  onChanged: () => void
  onUseAsReference: (assetId: string, thumb: string, label: string) => void
}

const STATUS_LABEL: Record<SelectionItem['status'], string> = {
  ok: '',
  missing: 'File missing',
  removed: 'No longer indexed',
  changed: 'File changed since added',
}

function fmtBytes(n: number) {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

function fmtTime(s: number) {
  const m = Math.floor(s / 60)
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`
}

export default function SelectionView({ selectionId, onBack, onChanged, onUseAsReference }: Props) {
  const [data, setData] = useState<(Selection & { items: SelectionItem[] }) | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [exporting, setExporting] = useState(false)

  const load = useCallback(async () => {
    try {
      const d = await api.selection(selectionId)
      setData(d)
      setName(d.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [selectionId])

  useEffect(() => {
    load()
  }, [load])

  if (error) return <div className="banner error">{error}</div>
  if (!data) return <p className="muted">Loading…</p>

  const move = async (i: number, delta: number) => {
    const ids = data.items.map((x) => x.id)
    const j = i + delta
    if (j < 0 || j >= ids.length) return
    ;[ids[i], ids[j]] = [ids[j], ids[i]]
    await api.reorderSelection(selectionId, ids)
    load()
  }

  return (
    <div className="selection-view">
      <div className="sel-head">
        <button className="btn small" onClick={onBack}>← Back to results</button>
        <input
          className="sel-name"
          aria-label="Selection name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={async () => {
            if (name.trim() && name !== data.name) {
              await api.renameSelection(selectionId, name.trim())
              onChanged()
            }
          }}
        />
        <span className="muted small">{data.items.length} items</span>
        <div className="push sel-actions">
          <a className="btn" href={`/api/selections/${selectionId}/manifest`} download>
            Download manifest
          </a>
          <button className="btn primary" onClick={() => setExporting((v) => !v)} disabled={data.items.length === 0}>
            Copy files to folder…
          </button>
        </div>
      </div>

      {exporting && <ExportPanel selectionId={selectionId} onClose={() => setExporting(false)} />}

      {data.items.length === 0 ? (
        <p className="muted empty">This selection is empty. Use “+ Add” on search results to collect items here.</p>
      ) : (
        <ol className="sel-list">
          {data.items.map((it, i) => (
            <li key={it.id} className={`sel-item ${it.status !== 'ok' ? 'problem' : ''}`}>
              <span className="sel-pos">{i + 1}</span>
              {it.thumbnail_url ? <img src={it.thumbnail_url} alt="" /> : <div className="thumb-missing" />}
              <div className="sel-info">
                <div className="sel-path" title={it.snapshot.rel_path}>{it.snapshot.rel_path}</div>
                <div className="muted small">
                  {it.snapshot.library_name ?? 'unknown library'}
                  {it.start_s != null && it.end_s != null ? ` · ${fmtTime(it.start_s)}–${fmtTime(it.end_s)}` : ''}
                  {it.query_context?.text ? ` · found with “${String(it.query_context.text)}”` : ''}
                </div>
                {it.status !== 'ok' && <div className="badge warn">{STATUS_LABEL[it.status]}</div>}
              </div>
              <div className="sel-item-actions">
                <button className="btn small" aria-label="Move up" onClick={() => move(i, -1)} disabled={i === 0}>↑</button>
                <button className="btn small" aria-label="Move down" onClick={() => move(i, 1)} disabled={i === data.items.length - 1}>↓</button>
                {it.status === 'ok' && it.snapshot.media_type === 'image' && (
                  <button className="btn small" onClick={() => onUseAsReference(it.asset_id, it.thumbnail_url ?? '', it.snapshot.rel_path)}>
                    Use as reference
                  </button>
                )}
                <button
                  className="btn small"
                  onClick={async () => {
                    await api.removeFromSelection(selectionId, it.id)
                    load()
                    onChanged()
                  }}
                >
                  Remove
                </button>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function ExportPanel({ selectionId, onClose }: { selectionId: string; onClose: () => void }) {
  const [dest, setDest] = useState('')
  const [includeManifest, setIncludeManifest] = useState(true)
  const [plan, setPlan] = useState<ExportPlan | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return
    const t = setInterval(async () => setJob(await api.job(job.id)), 500)
    return () => clearInterval(t)
  }, [job])

  const preview = async () => {
    setError(null)
    setJob(null)
    try {
      setPlan(await api.exportPreview(selectionId, dest.trim(), includeManifest))
    } catch (e) {
      setPlan(null)
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }

  const result = job?.result as
    | { copied: number; failed: { source: string; error: string }[]; skipped: unknown[]; manifest: string | null; cancelled: boolean }
    | null
    | undefined

  return (
    <section className="export-panel" aria-label="Copy export">
      <h3>Copy selected files to a folder</h3>
      <p className="muted small">
        Copies are made in a folder you choose. Originals are never moved or changed, and existing files are never overwritten:
        name clashes get a numbered name. Use a folder outside your libraries.
      </p>
      <div className="query-row">
        <input
          type="search"
          aria-label="Destination folder"
          placeholder="/Users/you/Desktop/selects"
          value={dest}
          onChange={(e) => {
            setDest(e.target.value)
            setPlan(null)
          }}
          spellCheck={false}
        />
        <button className="btn" onClick={preview} disabled={!dest.trim()}>Preview</button>
      </div>
      <label className="check small">
        <input type="checkbox" checked={includeManifest} onChange={(e) => setIncludeManifest(e.target.checked)} />
        Include a manifest (sources, licence records, how each item was found)
      </label>
      {error && <p className="error" role="alert">{error}</p>}

      {plan && !job && (
        <div className="plan">
          <p>
            Will copy <strong>{plan.files.length}</strong> files ({fmtBytes(plan.total_bytes)}) into <code>{plan.destination}</code>
            {plan.destination_exists ? '' : ' (new folder)'}. Overwrites: {plan.overwrites}.
          </p>
          <ul className="plan-files small">
            {plan.files.map((f) => (
              <li key={f.item_id}>
                <code>{f.dest_name}</code>
                {f.renamed && <span className="badge">renamed to avoid a clash</span>}
              </li>
            ))}
            {plan.manifest_name && <li><code>{plan.manifest_name}</code> <span className="muted">manifest</span></li>}
          </ul>
          {plan.skipped.length > 0 && (
            <p className="warn-text small">
              Skipping {plan.skipped.length}: {plan.skipped.map((s) => `${s.rel_path} (${s.reason})`).join(', ')}
            </p>
          )}
          <div className="detail-actions">
            <button
              className="btn primary"
              disabled={plan.files.length === 0}
              onClick={async () => {
                try {
                  setJob(await api.exportRun(selectionId, plan.plan_id))
                } catch (e) {
                  setError(e instanceof ApiError ? e.message : String(e))
                }
              }}
            >
              Copy {plan.files.length} files
            </button>
            <button className="btn" onClick={onClose}>Cancel</button>
          </div>
        </div>
      )}

      {job && (job.status === 'queued' || job.status === 'running') && (
        <div className="progress">
          <div className="bar"><div className="fill" style={{ width: `${job.total ? (100 * job.done) / job.total : 0}%` }} /></div>
          <span>Copying {job.done}/{job.total}</span>
          <button className="link" onClick={async () => setJob(await api.cancelJob(job.id))}>Stop</button>
        </div>
      )}
      {job && result && job.status !== 'running' && job.status !== 'queued' && (
        <div className={result.failed.length ? 'banner warn' : 'banner ok'} role="status">
          {result.cancelled ? 'Stopped. ' : 'Done. '}Copied {result.copied} files
          {result.manifest ? ` and ${result.manifest}` : ''} to <code>{plan?.destination}</code>.
          {result.failed.length > 0 && ` ${result.failed.length} failed: ${result.failed.map((f) => f.error).join('; ')}`}
        </div>
      )}
      {job && job.status === 'failed' && <p className="error">Export failed: {job.error}</p>}
    </section>
  )
}
