import { useEffect, useRef, useState } from 'react'
import type { MediaType, PlaceOption, QueryMode, Target, Taken } from '../api'

import { FOCUS_HINT } from '../lib/shortcut'
import { fmtSpan } from '../lib/time'

export type Reference =
  | { kind: 'file'; media: MediaType; file: File; previewUrl: string }
  | { kind: 'asset'; media: MediaType; assetId: string; previewUrl: string; label: string; startS?: number | null; endS?: number | null }

type Props = {
  text: string
  onText: (t: string) => void
  reference: Reference | null
  onReference: (r: Reference | null) => void
  includeIdentical: boolean
  onIncludeIdentical: (v: boolean) => void
  target: Target
  onTarget: (t: Target) => void
  modes: QueryMode[]
  onSearch: () => void
  searching: boolean
  taken: Taken
  onTaken: (t: Taken) => void
  places: PlaceOption[]
}

const IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']
const AUDIO_EXT = ['.wav', '.flac', '.mp3']
const ACCEPT = [...IMAGE_EXT, ...AUDIO_EXT].join(',')

export function mediaOfFile(f: File): MediaType | null {
  const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
  if (IMAGE_EXT.includes(ext)) return 'image'
  if (AUDIO_EXT.includes(ext)) return 'audio'
  return null
}

export default function SearchBar(p: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)

  useEffect(() => () => {
    if (p.reference?.kind === 'file' && p.reference.media === 'image') URL.revokeObjectURL(p.reference.previewUrl)
  }, [p.reference])

  const takeFile = (f: File | undefined) => {
    if (!f) return
    const media = mediaOfFile(f)
    if (!media) {
      setFileError(`Unsupported file type. Use ${[...IMAGE_EXT, ...AUDIO_EXT].join(', ')}.`)
      return
    }
    setFileError(null)
    p.onReference({ kind: 'file', media, file: f, previewUrl: media === 'image' ? URL.createObjectURL(f) : '' })
    if (p.target === 'all') p.onTarget(media)
  }

  const refMedia = p.reference?.media
  const refText = refMedia === 'audio' ? 'sound' : 'image'
    const targetText = { image: 'images', audio: 'sounds', video: 'video moments', all: 'everything' }[p.target]
  const queryKind = p.reference ? `${refMedia}${p.text.trim() ? '+text' : ''}` : 'text'
  const modeInfo = p.modes.find((m) => m.query === queryKind && m.target === (p.target === 'all' ? 'image+audio' : p.target))
  const experimental = modeInfo?.status === 'experimental'
  const mode = p.reference
    ? p.text.trim()
      ? `Reference ${refText} + text → ${targetText}`
      : `Reference ${refText} → ${targetText}`
    : `Text → ${targetText}`
  const canSearch = !!p.reference || !!p.text.trim()

  return (
    <form
      className="searchbar"
      onSubmit={(e) => {
        e.preventDefault()
        if (canSearch) p.onSearch()
      }}
    >
      <div
        className={`dropzone ${dragging ? 'dragging' : ''} ${p.reference ? 'has-ref' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          takeFile(e.dataTransfer.files[0])
        }}
      >
        {p.reference ? (
          <div className="ref">
            {p.reference.media === 'image' ? (
              <img src={p.reference.previewUrl} alt="Reference" />
            ) : (
              <div className="ref-audio">
                {p.reference.kind === 'asset' && p.reference.previewUrl ? <img src={p.reference.previewUrl} alt="" /> : <span className="ref-audio-icon">♪</span>}
                <span className="ref-audio-label">
                  {p.reference.kind === 'file' ? p.reference.file.name : p.reference.label.split('/').pop()}
                  {p.reference.kind === 'asset' && p.reference.startS != null ? ` ${fmtSpan(p.reference.startS, p.reference.endS)}` : ''}
                </span>
              </div>
            )}
            <button type="button" className="ref-clear" aria-label="Remove reference" onClick={() => p.onReference(null)}>
              ×
            </button>
          </div>
        ) : (
          <button type="button" className="ref-pick" onClick={() => fileInput.current?.click()}>
            Drop a reference image or sound
            <span>or click to choose</span>
          </button>
        )}
        <input
          ref={fileInput}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            takeFile(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </div>

      <div className="query">
        <div className="query-top">
          <label htmlFor="q" className="mode">
            {mode}
            {experimental && (
              <span className="badge exp" title="Runs on the shared embedding space, but result quality is unmeasured and uneven">
                experimental
              </span>
            )}
          </label>
          <div className="seg" role="radiogroup" aria-label="Search in">
            {(['image', 'audio', 'video', 'all'] as Target[]).map((t) => (
              <button
                key={t}
                type="button"
                role="radio"
                aria-checked={p.target === t}
                className={p.target === t ? 'on' : ''}
                disabled={!!refMedia && t === 'all'}
                title={refMedia && t === 'all' ? 'Reference searches target one media type at a time' : undefined}
                onClick={() => p.onTarget(t)}
              >
                {{ image: 'Images', audio: 'Sounds', video: 'Videos', all: 'All' }[t]}
              </button>
            ))}
          </div>
        </div>
        <div className="query-row">
          <input
            id="q"
            type="search"
            value={p.text}
            onChange={(e) => p.onText(e.target.value)}
            placeholder={`${
              p.reference?.media === 'audio'
                ? 'Optional: refine, e.g. "in a large hall" (experimental)'
                : p.reference
                  ? 'Optional: refine, e.g. "at night" or "in a forest"'
                  : p.target === 'audio'
                    ? 'Describe a sound, e.g. "glass breaking"…'
                    : p.target === 'video'
                      ? 'Describe a moment, e.g. "a dog in a field"…'
                      : 'Describe what you’re looking for…'
            } (${FOCUS_HINT})`}
            maxLength={2000}
            autoComplete="off"
          />
          <button className="btn primary" disabled={!canSearch || p.searching}>
            {p.searching ? 'Searching…' : 'Search'}
          </button>
        </div>
        {refMedia !== 'audio' && p.target !== 'audio' && (
          <div className="date-row small">
            <span className="muted">Taken</span>
            <label>
              from <input type="date" value={p.taken.from} max={p.taken.to || undefined} onChange={(e) => p.onTaken({ ...p.taken, from: e.target.value })} />
            </label>
            <label>
              to <input type="date" value={p.taken.to} min={p.taken.from || undefined} onChange={(e) => p.onTaken({ ...p.taken, to: e.target.value })} />
            </label>
            {(p.places.length > 0 || p.taken.place) && (
              <label>
                in{' '}
                <select value={p.taken.place} onChange={(e) => p.onTaken({ ...p.taken, place: e.target.value })} aria-label="Place taken">
                  <option value="">anywhere</option>
                  {p.taken.place && !p.places.some((pl) => pl.name === p.taken.place) && <option value={p.taken.place}>{p.taken.place}</option>}
                  {p.places.map((pl) => (
                    <option key={pl.name} value={pl.name}>
                      {'\u00a0\u00a0'.repeat(pl.depth)}
                      {pl.name.split(', ')[0]} ({pl.count})
                    </option>
                  ))}
                </select>
              </label>
            )}
            {(p.taken.from || p.taken.to || p.taken.place) && (
              <button type="button" className="btn small" onClick={() => p.onTaken({ from: '', to: '', place: '' })}>
                Any time or place
              </button>
            )}
            <span className="muted">From the photos’ and videos’ camera data; items without it, and sounds, are left out.</span>
          </div>
        )}
        {fileError && <p className="error small" role="alert">{fileError}</p>}
        {p.reference && (
          <label className="check small">
            <input type="checkbox" checked={p.includeIdentical} onChange={(e) => p.onIncludeIdentical(e.target.checked)} />
            Include the reference and exact copies in results
          </label>
        )}
      </div>
    </form>
  )
}
