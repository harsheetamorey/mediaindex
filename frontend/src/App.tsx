import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, type Job, type Library, type SearchResponse } from './api'
import DetailPanel from './components/DetailPanel'
import ResultsGrid from './components/ResultsGrid'
import SearchBar, { type Reference } from './components/SearchBar'
import Sidebar from './components/Sidebar'

function loadPref(key: string, fallback: boolean) {
  try {
    const v = localStorage.getItem(key)
    return v === null ? fallback : v === '1'
  } catch {
    return fallback
  }
}
function savePref(key: string, v: boolean) {
  try {
    localStorage.setItem(key, v ? '1' : '0')
  } catch {
    /* storage unavailable */
  }
}

export default function App() {
  const [libraries, setLibraries] = useState<Library[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [jobs, setJobs] = useState<Record<string, Job>>({})
  const [backendError, setBackendError] = useState<string | null>(null)

  const [text, setText] = useState('')
  const [reference, setReference] = useState<Reference | null>(null)
  const [includeIdentical, setIncludeIdentical] = useState(false)
  const [searching, setSearching] = useState(false)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [focused, setFocused] = useState(0)
  const [open, setOpen] = useState<number | null>(null)
  const [showScores, setShowScores] = useState(() => loadPref('mi.showScores', false))
  const searchSeq = useRef(0)

  const refreshLibraries = useCallback(async () => {
    try {
      setLibraries(await api.libraries())
      setBackendError(null)
    } catch (e) {
      setBackendError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    refreshLibraries()
    api.jobs().then((js) => setJobs(Object.fromEntries(js.map((j) => [j.id, j])))).catch(() => {})
  }, [refreshLibraries])

  // Poll active jobs.
  const activeIds = useMemo(
    () => Object.values(jobs).filter((j) => j.status === 'queued' || j.status === 'running').map((j) => j.id),
    [jobs],
  )
  useEffect(() => {
    if (activeIds.length === 0) return
    const t = setInterval(async () => {
      const updated = await Promise.all(activeIds.map((id) => api.job(id).catch(() => null)))
      setJobs((prev) => {
        const next = { ...prev }
        for (const j of updated) if (j) next[j.id] = j
        return next
      })
      refreshLibraries()
    }, 1000)
    return () => clearInterval(t)
  }, [activeIds, refreshLibraries])

  const libraryIds = selected.size ? Array.from(selected) : null

  const runSearch = useCallback(async () => {
    const seq = ++searchSeq.current
    setSearching(true)
    setSearchError(null)
    try {
      const r = reference
        ? await api.searchReference(
            reference.kind === 'file' ? { file: reference.file } : { assetId: reference.assetId },
            text,
            libraryIds,
            includeIdentical,
          )
        : await api.searchText(text.trim(), libraryIds)
      if (seq !== searchSeq.current) return
      setResponse(r)
      setFocused(0)
      setOpen(null)
    } catch (e) {
      if (seq !== searchSeq.current) return
      setSearchError(e instanceof ApiError ? e.message : String(e))
    } finally {
      if (seq === searchSeq.current) setSearching(false)
    }
  }, [reference, text, libraryIds, includeIdentical])

  const results = response?.results ?? []
  const openResult = open != null ? results[open] : null
  const st = response?.index_state

  return (
    <div className="app">
      <header className="topbar">
        <h1>MediaIndex</h1>
        <span className="muted small">Local media search · nothing leaves this computer</span>
        <label className="check small push">
          <input
            type="checkbox"
            checked={showScores}
            onChange={(e) => {
              setShowScores(e.target.checked)
              savePref('mi.showScores', e.target.checked)
            }}
          />
          Show technical scores
        </label>
      </header>

      {backendError && (
        <div className="banner error" role="alert">
          Can’t reach the MediaIndex backend ({backendError}). Start it with <code>uv run python -m mediaindex</code>.
        </div>
      )}

      <div className="body">
        <Sidebar
          libraries={libraries}
          selected={selected}
          jobs={jobs}
          onToggle={(id) =>
            setSelected((s) => {
              const n = new Set(s)
              if (n.has(id)) n.delete(id)
              else n.add(id)
              return n
            })
          }
          onAdd={async (path) => {
            const lib = await api.addLibrary(path)
            await refreshLibraries()
            const job = await api.importLibrary(lib.id)
            setJobs((j) => ({ ...j, [job.id]: job }))
          }}
          onImport={async (id) => {
            try {
              const job = await api.importLibrary(id)
              setJobs((j) => ({ ...j, [job.id]: job }))
            } catch (e) {
              alert(e instanceof Error ? e.message : String(e))
            }
          }}
          onCancel={async (jobId) => {
            const job = await api.cancelJob(jobId)
            setJobs((j) => ({ ...j, [job.id]: job }))
          }}
          onRemove={async (id) => {
            await api.removeLibrary(id)
            setSelected((s) => {
              const n = new Set(s)
              n.delete(id)
              return n
            })
            refreshLibraries()
          }}
        />

        <main className="main">
          <SearchBar
            text={text}
            onText={setText}
            reference={reference}
            onReference={setReference}
            includeIdentical={includeIdentical}
            onIncludeIdentical={setIncludeIdentical}
            onSearch={runSearch}
            searching={searching}
          />

          <section className="results" aria-busy={searching}>
            {searchError && <div className="banner error" role="alert">{searchError}</div>}

            {!response && !searchError && (
              <div className="empty">
                {libraries.length === 0 ? (
                  <p>Add a folder of images on the left. MediaIndex indexes it on this computer, then you can search it by description or with an example image.</p>
                ) : (
                  <p>Search {selected.size ? 'the selected libraries' : 'all libraries'} by describing an image, dropping in a reference image, or both.</p>
                )}
              </div>
            )}

            {response && (
              <>
                <div className="results-head small">
                  <span>
                    {results.length} closest matches
                    {selected.size ? ` in ${selected.size} selected ${selected.size === 1 ? 'library' : 'libraries'}` : ''}
                  </span>
                  {showScores && (
                    <span className="muted">
                      {response.mode} · query {response.timing_ms.query_embedding.toFixed(0)} ms · ranking{' '}
                      {response.timing_ms.ranking.toFixed(1)} ms · {response.candidates_searched} candidates
                    </span>
                  )}
                </div>
                {st && st.state === 'partial' && (
                  <div className="banner warn">
                    Only {st.searchable} of {st.total} items are indexed so far. Results cover indexed items only.
                  </div>
                )}
                {st && (st.state === 'empty' || st.state === 'not_indexed') && (
                  <div className="banner warn">Nothing is indexed in this selection yet. Import a folder first.</div>
                )}
                {results.length > 0 && (
                  <p className="muted small">
                    Results are ranked by similarity. The closest matches are always shown, even when nothing truly matches.
                  </p>
                )}
                <ResultsGrid
                  results={results}
                  focused={focused}
                  onFocus={setFocused}
                  onOpen={(i) => setOpen(i)}
                  showScores={showScores}
                />
              </>
            )}
          </section>
        </main>
      </div>

      {openResult && (
        <DetailPanel
          result={openResult}
          showScores={showScores}
          onClose={() => setOpen(null)}
          onPrev={() => setOpen((o) => (o == null ? o : Math.max(0, o - 1)))}
          onNext={() => setOpen((o) => (o == null ? o : Math.min(results.length - 1, o + 1)))}
          onUseAsReference={() => {
            const a = openResult.asset
            setReference({ kind: 'asset', assetId: a.id, previewUrl: a.thumbnail_url, label: a.rel_path })
            setOpen(null)
            window.scrollTo({ top: 0, behavior: 'smooth' })
          }}
        />
      )}
    </div>
  )
}
