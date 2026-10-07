import { useState } from 'react'
import type { Job, Library } from '../api'

type Props = {
  libraries: Library[]
  selected: Set<string>
  jobs: Record<string, Job>
  onToggle: (id: string) => void
  onAdd: (path: string) => Promise<void>
  onImport: (id: string) => void
  onCancel: (jobId: string) => void
  onRemove: (id: string) => void
  footer?: React.ReactNode
}

export default function Sidebar({ libraries, selected, jobs, onToggle, onAdd, onImport, onCancel, onRemove, footer }: Props) {
  const [adding, setAdding] = useState(false)
  const [path, setPath] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await onAdd(path.trim())
      setPath('')
      setAdding(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="sidebar" aria-label="Libraries">
      <div className="sidebar-head">
        <h2>Libraries</h2>
        <button className="btn small" onClick={() => setAdding((v) => !v)} aria-expanded={adding}>
          {adding ? 'Close' : '+ Add folder'}
        </button>
      </div>

      {adding && (
        <form className="add-form" onSubmit={submit}>
          <label htmlFor="lib-path">Folder path</label>
          <input
            id="lib-path"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/Users/you/Pictures/project"
            autoFocus
            spellCheck={false}
          />
          <p className="hint">
            Browsers can't hand a folder to a local app, so paste the folder's full path. In Finder, select the folder and press
            ⌥⌘C. MediaIndex only reads that folder. Your files are never moved, changed or uploaded.
          </p>
          {error && <p className="error" role="alert">{error}</p>}
          <button className="btn primary" disabled={!path.trim() || busy}>
            {busy ? 'Adding…' : 'Add library'}
          </button>
        </form>
      )}

      {libraries.length === 0 && !adding && (
        <p className="muted pad">No libraries yet. Add a folder of images to get started.</p>
      )}

      <ul className="lib-list">
        {libraries.map((lib) => {
          const job = Object.values(jobs)
            .filter((j) => j.kind === 'import' && j.library_id === lib.id)
            .sort((a, b) => b.created - a.created)[0]
          const active = job && (job.status === 'queued' || job.status === 'running')
          const pct = job && job.total ? Math.round((100 * job.done) / job.total) : 0
          return (
            <li key={lib.id} className={selected.has(lib.id) ? 'lib selected' : 'lib'}>
              <label className="lib-row">
                <input type="checkbox" checked={selected.has(lib.id)} onChange={() => onToggle(lib.id)} />
                <span className="lib-name" title={lib.root_path}>{lib.name}</span>
              </label>
              <div className="lib-meta">
                {lib.indexed_count} / {lib.asset_count} searchable
              </div>
              {active && (
                <div className="progress" aria-live="polite">
                  <div className="bar">
                    <div className="fill" style={{ width: `${pct}%` }} />
                  </div>
                  <span>
                    {job.status === 'queued' ? 'Queued' : `Indexing ${job.done}/${job.total}`}
                  </span>
                  <button className="link" onClick={() => onCancel(job.id)}>Cancel</button>
                </div>
              )}
              {job && !active && job.status !== 'done' && (
                <div className={job.status === 'failed' ? 'error small' : 'muted small'}>
                  Import {job.status}{job.error ? `: ${job.error}` : ''}
                </div>
              )}
              {job && job.status === 'done' && job.result && (
                <ImportSummary result={job.result} />
              )}
              <div className="lib-actions">
                <button className="link" onClick={() => onImport(lib.id)} disabled={!!active}>
                  {lib.asset_count ? 'Re-scan' : 'Import'}
                </button>
                <button
                  className="link danger"
                  onClick={() => {
                    if (confirm(`Remove "${lib.name}" from MediaIndex?\nThis only removes the index. Your files are not deleted.`)) onRemove(lib.id)
                  }}
                >
                  Remove
                </button>
              </div>
            </li>
          )
        })}
      </ul>
      {footer}
    </aside>
  )
}

function ImportSummary({ result }: { result: Record<string, unknown> }) {
  const n = (k: string) => (typeof result[k] === 'number' ? (result[k] as number) : 0)
  const failed = (result.failed as [string, string][] | undefined) ?? []
  return (
    <details className="summary small">
      <summary>
        Last import: {n('new')} new, {n('changed')} changed, {n('unchanged')} unchanged
        {failed.length ? `, ${failed.length} failed` : ''}
        {n('missing_count') ? `, ${n('missing_count')} missing` : ''}
      </summary>
      {failed.length > 0 && (
        <ul>
          {failed.slice(0, 20).map(([f, e]) => (
            <li key={f}><code>{f}</code>: {e}</li>
          ))}
        </ul>
      )}
    </details>
  )
}
