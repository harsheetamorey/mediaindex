import { useEffect, useRef, useState } from 'react'

export type Reference =
  | { kind: 'file'; file: File; previewUrl: string }
  | { kind: 'asset'; assetId: string; previewUrl: string; label: string }

type Props = {
  text: string
  onText: (t: string) => void
  reference: Reference | null
  onReference: (r: Reference | null) => void
  includeIdentical: boolean
  onIncludeIdentical: (v: boolean) => void
  onSearch: () => void
  searching: boolean
}

const ACCEPT = '.jpg,.jpeg,.png,.webp'

export default function SearchBar(p: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => () => {
    if (p.reference?.kind === 'file') URL.revokeObjectURL(p.reference.previewUrl)
  }, [p.reference])

  const takeFile = (f: File | undefined) => {
    if (!f) return
    p.onReference({ kind: 'file', file: f, previewUrl: URL.createObjectURL(f) })
  }

  const mode = p.reference ? (p.text.trim() ? 'Reference image + refinement text' : 'Similar to reference image') : 'Text search'
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
            <img src={p.reference.previewUrl} alt="Reference" />
            <button type="button" className="ref-clear" aria-label="Remove reference image" onClick={() => p.onReference(null)}>
              ×
            </button>
          </div>
        ) : (
          <button type="button" className="ref-pick" onClick={() => fileInput.current?.click()}>
            Drop a reference image
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
        <label htmlFor="q" className="mode">{mode}</label>
        <div className="query-row">
          <input
            id="q"
            type="search"
            value={p.text}
            onChange={(e) => p.onText(e.target.value)}
            placeholder={p.reference ? 'Optional: refine, e.g. "at night" or "in a forest"' : 'Describe what you’re looking for…'}
            maxLength={2000}
            autoComplete="off"
          />
          <button className="btn primary" disabled={!canSearch || p.searching}>
            {p.searching ? 'Searching…' : 'Search'}
          </button>
        </div>
        {p.reference && (
          <label className="check small">
            <input type="checkbox" checked={p.includeIdentical} onChange={(e) => p.onIncludeIdentical(e.target.checked)} />
            Include the reference image and exact copies in results
          </label>
        )}
      </div>
    </form>
  )
}
