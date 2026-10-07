import { useEffect, useRef } from 'react'
import type { Result } from '../api'
import { fmtSpan, fmtTime } from '../lib/time'

type Props = {
  results: Result[]
  focused: number
  onFocus: (i: number) => void
  onOpen: (i: number) => void
  showScores: boolean
  onAdd: (i: number) => void
  addedKeys: Set<string>
  onPlay: (i: number) => void
  playingKey: string | null
  offset?: number
  label?: string
}

export const resultKey = (r: Result) => `${r.asset.id}|${r.start_s ?? ''}|${r.end_s ?? ''}`

export default function ResultsGrid(props: Props) {
  const off = props.offset ?? 0
  const results = props.results
  const focused = props.focused - off
  const onFocus = (i: number) => props.onFocus(i + off)
  const onOpen = (i: number) => props.onOpen(i + off)
  const onAdd = (i: number) => props.onAdd(i + off)
  const onPlay = (i: number) => props.onPlay(i + off)
  const { showScores, addedKeys, playingKey } = props
  const grid = useRef<HTMLUListElement>(null)

  useEffect(() => {
    if (focused < 0 || focused >= results.length) return
    const el = grid.current?.querySelectorAll<HTMLElement>('.cell')[focused]
    if (el && el !== document.activeElement && grid.current?.contains(document.activeElement)) el.focus({ preventScroll: false })
  }, [focused, results.length])

  const onKey = (e: React.KeyboardEvent) => {
    if (!grid.current || results.length === 0) return
    const cells = grid.current.querySelectorAll<HTMLElement>('.cell')
    const first = cells[0]?.getBoundingClientRect()
    const perRow = Math.max(1, Array.from(cells).filter((c) => c.getBoundingClientRect().top === first?.top).length)
    const moves: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: perRow, ArrowUp: -perRow }
    if (e.key in moves) {
      e.preventDefault()
      onFocus(Math.min(results.length - 1, Math.max(0, focused + moves[e.key])))
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      onOpen(focused)
    } else if (e.key === 'a' || e.key === '+') {
      e.preventDefault()
      onAdd(focused)
    } else if (e.key === 'p' && results[focused]?.asset.media_type === 'audio') {
      e.preventDefault()
      onPlay(focused)
    }
  }

  return (
    <ul className="grid" ref={grid} onKeyDown={onKey} role="listbox" aria-label={props.label ?? 'Search results'}>
      {results.map((r, i) => (
        <li
          key={r.asset.id + (r.start_s ?? '')}
          className="cell"
          role="option"
          aria-selected={i === focused}
          tabIndex={i === focused || (i === 0 && (focused < 0 || focused >= results.length)) ? 0 : -1}
          onFocus={() => i !== focused && onFocus(i)}
          onClick={() => {
            onFocus(i)
            onOpen(i)
          }}
          title={r.asset.rel_path}
        >
          <img src={r.window_thumbnail_url ?? r.asset.thumbnail_url} alt={r.asset.rel_path} loading="lazy" />
          <button
            className={`cell-add ${addedKeys.has(resultKey(r)) ? 'added' : ''}`}
            tabIndex={-1}
            aria-label={addedKeys.has(resultKey(r)) ? 'In selection' : 'Add to selection'}
            onClick={(e) => {
              e.stopPropagation()
              onAdd(i)
            }}
          >
            {addedKeys.has(resultKey(r)) ? '✓' : '+ Add'}
          </button>
          {r.asset.media_type === 'audio' && (
            <button
              className={`cell-play ${playingKey === resultKey(r) ? 'on' : ''}`}
              tabIndex={-1}
              aria-label={playingKey === resultKey(r) ? 'Stop' : `Play ${fmtSpan(r.start_s, r.end_s)}`}
              onClick={(e) => {
                e.stopPropagation()
                onPlay(i)
              }}
            >
              {playingKey === resultKey(r) ? '■' : '▶'}
            </button>
          )}
          {r.asset.media_type === 'video' && <span className="cell-video" aria-hidden="true">▶ video</span>}
          {r.start_s != null && (
            <span className="cell-span">
              {fmtSpan(r.start_s, r.end_s)}
              {r.asset.duration ? ` of ${fmtTime(r.asset.duration)}` : ''}
            </span>
          )}
          <div className="cell-label">
            <span className="name">{r.asset.rel_path.split('/').pop()}</span>
            {showScores && <span className="score" title="Raw cosine similarity (not a probability)">{r.similarity.toFixed(3)}</span>}
          </div>
        </li>
      ))}
    </ul>
  )
}
