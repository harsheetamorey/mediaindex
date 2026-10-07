import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  ApiError,
  type IndexState,
  type Job,
  type Library,
  type MediaType,
  type QueryMode,
  type Result,
  type SearchResponse,
  type Selection,
  type Target,
} from './api'
import { useSegmentPlayer } from './lib/player'
import DetailPanel from './components/DetailPanel'
import ResultsGrid, { resultKey } from './components/ResultsGrid'
import SearchBar, { type Reference } from './components/SearchBar'
import SelectionView from './components/SelectionView'
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
  const [target, setTarget] = useState<Target>('image')
  const [modes, setModes] = useState<QueryMode[]>([])
  const [globalRank, setGlobalRank] = useState(false)
  const player = useSegmentPlayer()
  const [searching, setSearching] = useState(false)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [focused, setFocused] = useState(0)
  const [open, setOpen] = useState<number | null>(null)
  const [showScores, setShowScores] = useState(() => loadPref('mi.showScores', false))
  const searchSeq = useRef(0)

  const [selections, setSelections] = useState<Selection[]>([])
  const [activeSel, setActiveSel] = useState<string | null>(() => {
    try {
      return localStorage.getItem('mi.activeSelection')
    } catch {
      return null
    }
  })
  const [view, setView] = useState<'search' | 'selection'>('search')
  const [addedKeys, setAddedKeys] = useState<Set<string>>(new Set())
  const [toast, setToast] = useState<string | null>(null)

  const refreshSelections = useCallback(async () => {
    try {
      const list = await api.selections()
      setSelections(list)
      setActiveSel((cur) => (cur && list.some((s) => s.id === cur) ? cur : list[0]?.id ?? null))
    } catch {
      /* backend error banner covers this */
    }
  }, [])

  useEffect(() => {
    refreshSelections()
  }, [refreshSelections])

  useEffect(() => {
    try {
      if (activeSel) localStorage.setItem('mi.activeSelection', activeSel)
    } catch {
      /* storage unavailable */
    }
    if (!activeSel) return
    api
      .selection(activeSel)
      .then((d) => setAddedKeys(new Set(d.items.map((i) => `${i.asset_id}|${i.start_s ?? ''}|${i.end_s ?? ''}`))))
      .catch(() => setAddedKeys(new Set()))
  }, [activeSel])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 2500)
    return () => clearTimeout(t)
  }, [toast])

  const addToSelection = useCallback(
    async (r: Result) => {
      try {
        let sid = activeSel
        if (!sid) {
          const created = await api.createSelection('Selection 1')
          sid = created.id
          setActiveSel(sid)
        }
        const ctx = response
          ? { mode: response.mode, ...Object.fromEntries(Object.entries(response.query).filter(([k]) => ['text', 'reference', 'rel_path', 'filename', 'asset_id'].includes(k))), rank: r.rank }
          : null
        const res = await api.addToSelection(sid, r.asset.id, ctx, r.start_s, r.end_s)
        setAddedKeys((s) => new Set(s).add(resultKey(r)))
        const name = selections.find((s) => s.id === sid)?.name ?? 'selection'
        setToast(res.added ? `Added to “${name}”` : `Already in “${name}”`)
        refreshSelections()
      } catch (e) {
        setToast(e instanceof Error ? e.message : String(e))
      }
    },
    [activeSel, response, selections, refreshSelections],
  )

  const refreshLibraries = useCallback(async () => {
    try {
      setLibraries(await api.libraries())
      setBackendError(null)
    } catch (e) {
      setBackendError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    api.capabilities().then((c) => setModes(c.modes)).catch(() => {})
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
      player.stop()
      const single: MediaType = target === 'both' ? (reference?.media ?? 'image') : target
      const r = reference
        ? reference.media === 'audio'
          ? await api.searchAudioReference(
              reference.kind === 'file' ? { file: reference.file } : { assetId: reference.assetId, startS: reference.startS },
              text,
              libraryIds,
              includeIdentical,
              single,
            )
          : await api.searchReference(
              reference.kind === 'file' ? { file: reference.file } : { assetId: reference.assetId },
              text,
              libraryIds,
              includeIdentical,
              single,
            )
        : await api.searchText(text.trim(), libraryIds, target === 'both' ? ['image', 'audio'] : [target], 48, globalRank)
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
  }, [reference, text, libraryIds, includeIdentical, target, player, globalRank])

  const results = useMemo(() => {
    const rs = response?.results ?? []
    if (response?.grouping !== 'by_modality') return rs
    return [...rs.filter((r) => r.asset.media_type === 'image'), ...rs.filter((r) => r.asset.media_type !== 'image')]
  }, [response])
  const openResult = open != null ? results[open] : null
  const rawState = response?.index_state
  const st: IndexState | undefined =
    rawState && 'state' in rawState
      ? (rawState as IndexState)
      : rawState
        ? (Object.values(rawState as Record<string, IndexState>).find((x) => x.state !== 'ready') ??
          Object.values(rawState as Record<string, IndexState>)[0])
        : undefined
  const grouped = response?.grouping === 'by_modality'
  const sections: { label: string; offset: number; items: Result[] }[] = grouped
    ? (['image', 'audio'] as MediaType[])
        .map((m) => ({ m, items: results.filter((r) => (r.asset.media_type === 'image' ? 'image' : 'audio') === m) }))
        .reduce<{ label: string; offset: number; items: Result[] }[]>((acc, g) => {
          const offset = acc.reduce((n, x) => n + x.items.length, 0)
          acc.push({ label: g.m === 'image' ? 'Images' : 'Sounds', offset, items: g.items })
          return acc
        }, [])
    : [{ label: 'Search results', offset: 0, items: results }]

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
        {showScores && (
          <label className="check small" title="Merge images and sounds into one list by raw similarity (not calibrated)">
            <input type="checkbox" checked={globalRank} onChange={(e) => setGlobalRank(e.target.checked)} />
            Mixed ranking (experimental)
          </label>
        )}
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
          footer={
            <section className="selections" aria-label="Selections">
              <div className="sidebar-head">
                <h2>Selections</h2>
                <button
                  className="btn small"
                  onClick={async () => {
                    const s = await api.createSelection(`Selection ${selections.length + 1}`)
                    setActiveSel(s.id)
                    refreshSelections()
                  }}
                >
                  + New
                </button>
              </div>
              {selections.length === 0 && <p className="muted small">Use “+ Add” on results to start a selection.</p>}
              <ul className="lib-list">
                {selections.map((s) => (
                  <li key={s.id} className={s.id === activeSel ? 'lib selected' : 'lib'}>
                    <label className="lib-row">
                      <input type="radio" name="active-selection" checked={s.id === activeSel} onChange={() => setActiveSel(s.id)} />
                      <span className="lib-name">{s.name}</span>
                    </label>
                    <div className="lib-meta">{s.item_count ?? 0} items{s.id === activeSel ? ' · adding here' : ''}</div>
                    <div className="lib-actions">
                      <button
                        className="link"
                        onClick={() => {
                          setActiveSel(s.id)
                          setView('selection')
                        }}
                      >
                        Open
                      </button>
                      <button
                        className="link danger"
                        onClick={async () => {
                          if (!confirm(`Delete selection "${s.name}"? Media files are not affected.`)) return
                          await api.deleteSelection(s.id)
                          if (activeSel === s.id) setView('search')
                          refreshSelections()
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          }
        />

        <main className="main">
          {view === 'selection' && activeSel ? (
            <section className="results">
              <SelectionView
                key={activeSel}
                selectionId={activeSel}
                onBack={() => setView('search')}
                onChanged={refreshSelections}
                onUseAsReference={(assetId, thumb, label) => {
                  setReference({ kind: 'asset', media: 'image', assetId, previewUrl: thumb, label })
                  setView('search')
                }}
              />
            </section>
          ) : (
          <>
          <SearchBar
            text={text}
            onText={setText}
            reference={reference}
            onReference={setReference}
            includeIdentical={includeIdentical}
            onIncludeIdentical={setIncludeIdentical}
            target={target}
            onTarget={setTarget}
            modes={modes}
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
                {response.grouping === 'global (experimental)' && (
                  <div className="banner warn">
                    Experimental mixed ranking: images and sounds are merged by raw similarity, which is not calibrated across media
                    types.
                  </div>
                )}
                {(response.query as { experimental?: boolean }).experimental && (
                  <div className="banner warn">
                    Experimental cross-media search: results are similarity candidates, not synchronized or curated matches.
                  </div>
                )}
                {sections.map((sec) => (
                  <div key={sec.label} className="section">
                    {grouped && (
                      <h2 className="section-title">
                        {sec.label} <span className="muted small">({sec.items.length})</span>
                      </h2>
                    )}
                    {grouped && sec.items.length === 0 && <p className="muted small">No indexed {sec.label.toLowerCase()} in this selection.</p>}
                    <ResultsGrid
                      results={sec.items}
                      offset={sec.offset}
                      label={sec.label}
                      focused={focused}
                      onFocus={setFocused}
                      onOpen={(i) => setOpen(i)}
                      showScores={showScores}
                      onAdd={(i) => addToSelection(results[i])}
                      addedKeys={addedKeys}
                      playingKey={player.playing}
                      onPlay={(i) => {
                        const r = results[i]
                        player.play(resultKey(r), r.asset.file_url, r.start_s, r.end_s)
                      }}
                    />
                  </div>
                ))}
              </>
            )}
          </section>
          </>
          )}
        </main>
      </div>

      {openResult && (
        <DetailPanel
          result={openResult}
          showScores={showScores}
          onAdd={() => addToSelection(openResult)}
          added={addedKeys.has(resultKey(openResult))}
          playing={player.playing === resultKey(openResult)}
          onPlaySegment={(start, end) =>
            player.play(
              start === openResult.start_s ? resultKey(openResult) : `${openResult.asset.id}|${start}|${end}`,
              openResult.asset.file_url,
              start,
              end,
            )
          }
          onClose={() => setOpen(null)}
          onPrev={() => setOpen((o) => (o == null ? o : Math.max(0, o - 1)))}
          onNext={() => setOpen((o) => (o == null ? o : Math.min(results.length - 1, o + 1)))}
          onUseAsReference={() => {
            const a = openResult.asset
            const media: MediaType = a.media_type === 'image' ? 'image' : 'audio'
            setReference({ kind: 'asset', media, assetId: a.id, previewUrl: a.thumbnail_url, label: a.rel_path, startS: openResult.start_s, endS: openResult.end_s })
            if (target === 'both') setTarget(media)
            player.stop()
            setOpen(null)
            window.scrollTo({ top: 0, behavior: 'smooth' })
          }}
        />
      )}
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  )
}
