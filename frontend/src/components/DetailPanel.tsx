import { useEffect, useState } from 'react'
import { api, type Result } from '../api'

type Props = {
  result: Result
  onClose: () => void
  onUseAsReference: () => void
  onPrev: () => void
  onNext: () => void
  showScores: boolean
  onAdd: () => void
  added: boolean
}

function fmtBytes(n: number | null) {
  if (!n) return '—'
  const u = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let v = n
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i ? 1 : 0)} ${u[i]}`
}

export default function DetailPanel({ result, onClose, onUseAsReference, onPrev, onNext, showScores, onAdd, added }: Props) {
  const a = result.asset
  const [note, setNote] = useState<string | null>(null)
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft') onPrev()
      else if (e.key === 'ArrowRight') onNext()
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose, onPrev, onNext])

  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-label={a.rel_path} onClick={onClose}>
      <div className="detail" onClick={(e) => e.stopPropagation()}>
        <div className="detail-media">
          <img src={a.file_url} alt={a.rel_path} />
        </div>
        <div className="detail-info">
          <div className="detail-head">
            <h3 title={a.rel_path}>{a.rel_path.split('/').pop()}</h3>
            <button className="btn small" onClick={onClose} aria-label="Close preview">Close</button>
          </div>
          <dl>
            <dt>Path in library</dt>
            <dd><code>{a.rel_path}</code></dd>
            <dt>Dimensions</dt>
            <dd>{a.width && a.height ? `${a.width} × ${a.height}` : '—'}</dd>
            <dt>Size</dt>
            <dd>{fmtBytes(a.size)}</dd>
            <dt>Result rank</dt>
            <dd>#{result.rank}</dd>
            {showScores && (
              <>
                <dt>Raw similarity</dt>
                <dd>{result.similarity.toFixed(4)} <span className="muted">(cosine; not a probability)</span></dd>
              </>
            )}
            {a.source && (
              <>
                <dt>Source</dt>
                <dd>
                  {a.source.dataset ?? 'external'}{a.source.row != null ? `, row ${a.source.row}` : ''}
                  <div className="muted small">{a.source.license_declared}</div>
                  {a.source.note && <div className="muted small">{a.source.note}</div>}
                </dd>
              </>
            )}
          </dl>
          <div className="detail-actions">
            <button className="btn primary" onClick={onUseAsReference}>Use as reference</button>
            <button className="btn" onClick={onAdd} disabled={added}>{added ? 'In selection ✓' : 'Add to selection'}</button>
            <a className="btn" href={a.file_url} target="_blank" rel="noreferrer">Open original</a>
            <button
              className="btn"
              onClick={async () => {
                try {
                  await api.reveal(a.id)
                  setNote('Shown in your file manager.')
                } catch (e) {
                  setNote(e instanceof Error ? e.message : String(e))
                }
              }}
            >
              Reveal in folder
            </button>
            <button
              className="btn"
              onClick={async () => {
                try {
                  const { path } = await api.assetPath(a.id)
                  await navigator.clipboard.writeText(path)
                  setNote('Path copied.')
                } catch (e) {
                  setNote(`Could not copy path: ${e instanceof Error ? e.message : String(e)}`)
                }
              }}
            >
              Copy path
            </button>
          </div>
          {note && <p className="muted small" role="status">{note}</p>}
          <div className="detail-nav">
            <button className="btn small" onClick={onPrev}>← Prev</button>
            <button className="btn small" onClick={onNext}>Next →</button>
          </div>
        </div>
      </div>
    </div>
  )
}
