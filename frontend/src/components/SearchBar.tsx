import { useEffect, useRef, useState } from 'react'
import type { MediaType } from '../api'
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
  target: MediaType
  onTarget: (t: MediaType) => void
  onSearch: () => void
  searching: boolean
}

const IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.webp']
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
    if (media === 'audio') p.onTarget('audio')
  }

  const refMedia = p.reference?.media
  const refText = refMedia === 'audio' ? 'sound' : 'image'
  const textAllowed = refMedia !== 'audio' // native audio+text refinement is not offered (see docs)
  const mode = p.reference
    ? p.text.trim() && textAllowed
      ? `Reference ${refText} + refinement text`
      : `Similar to reference ${refText}`
    : `Text search · ${p.target === 'image' ? 'images' : 'sounds'}`
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
          <label htmlFor="q" className="mode">{mode}</label>
          <div className="seg" role="radiogroup" aria-label="Search in">
            {(['image', 'audio'] as MediaType[]).map((t) => (
              <button
                key={t}
                type="button"
                role="radio"
                aria-checked={p.target === t}
                className={p.target === t ? 'on' : ''}
                disabled={!!refMedia && refMedia !== t}
                title={refMedia && refMedia !== t ? 'Cross-media search arrives in a later phase' : undefined}
                onClick={() => p.onTarget(t)}
              >
                {t === 'image' ? 'Images' : 'Sounds'}
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
            disabled={!textAllowed}
            placeholder={
              !textAllowed
                ? 'Text refinement is not available for sound references'
                : p.reference
                  ? 'Optional: refine, e.g. "at night" or "in a forest"'
                  : p.target === 'audio'
                    ? 'Describe a sound, e.g. "glass breaking"…'
                    : 'Describe what you’re looking for…'
            }
            maxLength={2000}
            autoComplete="off"
          />
          <button className="btn primary" disabled={!canSearch || p.searching}>
            {p.searching ? 'Searching…' : 'Search'}
          </button>
        </div>
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
