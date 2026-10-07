import { useEffect, useRef } from 'react'
import type { Result } from '../api'

type Props = {
  results: Result[]
  focused: number
  onFocus: (i: number) => void
  onOpen: (i: number) => void
  showScores: boolean
  onAdd: (i: number) => void
  addedKeys: Set<string>
}

export const resultKey = (r: Result) => `${r.asset.id}|${r.start_s ?? ''}|${r.end_s ?? ''}`

export default function ResultsGrid({ results, focused, onFocus, onOpen, showScores, onAdd, addedKeys }: Props) {
  const grid = useRef<HTMLUListElement>(null)

  useEffect(() => {
    const el = grid.current?.querySelectorAll<HTMLElement>('.cell')[focused]
    el?.focus({ preventScroll: false })
  }, [focused])

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
    }
  }

  return (
    <ul className="grid" ref={grid} onKeyDown={onKey} role="listbox" aria-label="Search results">
      {results.map((r, i) => (
        <li
          key={r.asset.id + (r.start_s ?? '')}
          className="cell"
          role="option"
          aria-selected={i === focused}
          tabIndex={i === focused ? 0 : -1}
          onFocus={() => i !== focused && onFocus(i)}
          onClick={() => {
            onFocus(i)
            onOpen(i)
          }}
          title={r.asset.rel_path}
        >
          <img src={r.asset.thumbnail_url} alt={r.asset.rel_path} loading="lazy" />
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
          <div className="cell-label">
            <span className="name">{r.asset.rel_path.split('/').pop()}</span>
            {showScores && <span className="score" title="Raw cosine similarity (not a probability)">{r.similarity.toFixed(3)}</span>}
          </div>
        </li>
      ))}
    </ul>
  )
}
