import { useState } from 'react'
import type { TimelineMonth } from '../api'
import { monthLabel } from '../lib/time'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

type Props = {
  months: TimelineMonth[] // newest first
  undated: number
  active: string | null
  onPick: (month: string) => void
}

/** Browse by date: the years with photos or videos; the chosen year's months, with how many items each. */
export default function Timeline({ months, undated, active, onPick }: Props) {
  const years = [...new Set(months.map((m) => m.month.slice(0, 4)))]
  const [year, setYear] = useState<string | null>(null)
  const shown = year ?? active?.slice(0, 4) ?? years[0]
  if (years.length === 0) return null
  const inYear = months.filter((m) => m.month.startsWith(shown)).reverse() // oldest month first

  return (
    <nav className="timeline small" aria-label="Browse by date">
      <div className="timeline-years">
        <span className="muted">Browse by date</span>
        {years.map((y) => (
          <button key={y} type="button" className={y === shown ? 'chip on' : 'chip'} aria-pressed={y === shown} onClick={() => setYear(y)}>
            {y}
          </button>
        ))}
        {undated > 0 && <span className="muted">· {undated} without a date</span>}
      </div>
      <div className="timeline-months">
        {inYear.map((m) => (
          <button
            key={m.month}
            type="button"
            className={m.month === active ? 'chip on' : 'chip'}
            aria-pressed={m.month === active}
            title={`${monthLabel(m.month)}: ${m.count} ${m.count === 1 ? 'item' : 'items'}`}
            onClick={() => onPick(m.month)}
          >
            {MONTHS[Number(m.month.slice(5)) - 1]} <span className="muted">{m.count}</span>
          </button>
        ))}
      </div>
    </nav>
  )
}
