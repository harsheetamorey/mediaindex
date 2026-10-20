import { useMemo, useState } from 'react'
import type { MapPoint, PlaceOption } from '../api'
import { WORLD_LAND } from '../lib/worldLand'

type Props = {
  points: MapPoint[]
  unplaced: number
  levels: PlaceOption[] // countries, regions and towns, for the list under the map
  active: string | null
  onPick: (place: string) => void
}

const x = (lon: number) => lon + 180
const y = (lat: number) => 90 - lat

/** An offline map of where photos and videos were taken (Natural Earth outline, no map tiles), plus the same places
 * as a list. Click a dot or a name to see what was taken there. */
export default function PlacesMap({ points, unplaced, levels, active, onPick }: Props) {
  const [world, setWorld] = useState(false)
  const [country, setCountry] = useState<string | null>(null)
  const view = useMemo(() => {
    if (world || points.length === 0) return { x: 0, y: 0, w: 360, h: 180 }
    const xs = points.map((p) => x(p.lon))
    const ys = points.map((p) => y(p.lat))
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2
    let w = Math.max(24, (Math.max(...xs) - Math.min(...xs)) * 1.4)
    let h = Math.max(12, (Math.max(...ys) - Math.min(...ys)) * 1.4)
    if (w < h * 2) w = h * 2
    else h = w / 2
    w = Math.min(w, 360)
    h = Math.min(h, 180)
    return { x: Math.min(Math.max(cx - w / 2, 0), 360 - w), y: Math.min(Math.max(cy - h / 2, 0), 180 - h), w, h }
  }, [points, world])
  const most = Math.max(1, ...points.map((p) => p.count))
  const countries = levels.filter((l) => l.depth === 0)
  const shownCountry = country ?? (active ? active.split(', ').pop() ?? null : null)
  const inside = shownCountry ? levels.filter((l) => l.depth > 0 && l.name.endsWith(`, ${shownCountry}`)) : []

  return (
    <section className="places-map small" aria-label="Where photos and videos were taken">
      <svg viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} role="img" aria-label="Map of places">
        <rect x={0} y={0} width={360} height={180} className="map-sea" />
        <path d={WORLD_LAND} className="map-land" />
        {points.map((p) => {
          const r = (view.w / 120) * (0.5 + Math.sqrt(p.count / most))
          return (
            <circle
              key={p.place}
              cx={x(p.lon)}
              cy={y(p.lat)}
              r={r}
              className={active && (p.place === active || p.place.endsWith(`, ${active}`)) ? 'map-dot on' : 'map-dot'}
              strokeWidth={r / 4}
              tabIndex={0}
              role="button"
              aria-label={`${p.place}: ${p.count} ${p.count === 1 ? 'item' : 'items'}`}
              onClick={() => onPick(p.place)}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onPick(p.place))}
            >
              <title>{`${p.place}: ${p.count} ${p.count === 1 ? 'item' : 'items'}`}</title>
            </circle>
          )
        })}
      </svg>
      <div className="timeline-months">
        <button type="button" className="chip" onClick={() => setWorld(!world)}>
          {world ? 'Zoom to my places' : 'Whole world'}
        </button>
        {countries.map((c) => (
          <button
            key={c.name}
            type="button"
            className={c.name === active ? 'chip on' : 'chip'}
            onClick={() => (setCountry(c.name), onPick(c.name))}
          >
            {c.name} <span className="muted">{c.count}</span>
          </button>
        ))}
        {unplaced > 0 && <span className="muted">· {unplaced} without a place</span>}
      </div>
      {inside.length > 0 && (
        <div className="timeline-months">
          {inside.map((l) => (
            <button key={l.name} type="button" className={l.name === active ? 'chip on' : 'chip'} onClick={() => onPick(l.name)}>
              {l.name.split(', ')[0]} <span className="muted">{l.count}</span>
            </button>
          ))}
        </div>
      )}
      <p className="muted">Places are named on this computer from the photos’ GPS: the nearest town. Map: Natural Earth.</p>
    </section>
  )
}
