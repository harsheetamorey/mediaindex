import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, type Asset, type AskStatus, type AskTurn, type DetectedBox, type EditInfo } from '../api'
import { FOCUS_HINT } from '../lib/shortcut'

type Message = {
  role: 'user' | 'assistant'
  text: string
  results?: Asset[]
  boxes?: Record<string, DetectedBox[]>
  maybe?: string[]
  action?: string
  via?: string
  error?: boolean
  edit?: EditInfo | null
  job?: string // an edit still being made
  progress?: string
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** The edited copy, with the original one click away, and a way to keep it. */
function EditView({ edit, source }: { edit: EditInfo; source?: Asset }) {
  const [showOriginal, setShowOriginal] = useState(false)
  const [saved, setSaved] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const save = async () => {
    setError(null)
    setSaving(true)
    try {
      const picked = await api.pickFolder()
      if (picked.cancelled || !picked.path) return
      const r = await api.saveEdit(edit.id, picked.path)
      setSaved(`Saved as “${r.saved}” in ${r.destination}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }
  const comparable = (edit.op === 'remove' || edit.op === 'replace') && source
  const original = showOriginal && comparable
  const tag = original ? 'Original' : { remove: 'Edited copy', replace: 'Edited copy', generate: 'Made-up image', video: 'Video' }[edit.op]
  return (
    <div className={edit.kind === 'video' && edit.height > edit.width ? 'edit-view tall' : 'edit-view'}>
      {edit.kind === 'video' ? (
        <video className="edit-img" src={edit.url} controls playsInline style={{ aspectRatio: `${edit.width} / ${edit.height}` }} />
      ) : (
        <img
          className="edit-img"
          src={original ? source.file_url : edit.url}
          alt={original ? `original: ${source.rel_path}` : edit.op === 'generate' ? `made-up image of ${edit.new}` : `edited copy of ${source?.rel_path ?? 'the photo'}`}
          style={{ aspectRatio: `${edit.width} / ${edit.height}` }}
        />
      )}
      <div className="edit-actions">
        <span className="edit-tag">{tag}</span>
        {comparable && (
          <button className="btn small" onClick={() => setShowOriginal((v) => !v)} aria-pressed={showOriginal}>
            {showOriginal ? 'Show edited' : 'Show original'}
          </button>
        )}
        <button className="btn small primary" onClick={save} disabled={saving}>
          {saving ? 'Choose a folder…' : 'Save a copy…'}
        </button>
      </div>
      {saved && <div className="muted small">{saved}</div>}
      {error && <div className="error small" role="alert">{error}</div>}
    </div>
  )
}

const SUGGESTIONS = ['How many dogs do I have?', 'Show me a city street at night', 'Which of those have cars?', "What's in my library?"]
const SHOWN = 24

export default function AskView({ libraryIds }: { libraryIds: string[] | null }) {
  const [status, setStatus] = useState<AskStatus | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const end = useRef<HTMLDivElement>(null)
  const libs = libraryIds ?? undefined

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.askStatus(libs))
    } catch {
      /* backend error banner covers this */
    }
  }, [libs?.join(',')])

  useEffect(() => {
    refresh()
  }, [refresh])

  const job = status?.detector.job
  useEffect(() => {
    if (!job) return
    const t = setInterval(refresh, 1500)
    return () => clearInterval(t)
  }, [job?.id, refresh])

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  const send = async (message: string, asset?: Asset) => {
    const q = message.trim()
    if (!q || busy) return
    const history: AskTurn[] = messages
      .filter((m) => !m.error)
      .slice(-12)
      .filter((m) => !m.job)
      .map((m) => ({
        role: m.role,
        text: m.text,
        asset_ids: m.results?.map((a) => a.id) ?? [],
        action: m.action,
        edit_id: m.edit?.id ?? null,
      }))
    setMessages((ms) => [...ms, { role: 'user', text: q, results: asset ? [asset] : undefined }])
    setText('')
    setBusy(true)
    try {
      const r = await api.ask(q, history, libs, asset?.id)
      const via = r.described_by === 'chat model' || r.routed_by === 'chat model' || (r.action === 'chat' && r.chat_model.available)
      setMessages((ms) => [
        ...ms,
        {
          role: 'assistant',
          text: r.answer,
          results: r.results,
          boxes: r.boxes,
          maybe: r.maybe_ids,
          action: r.action,
          via: via ? r.chat_model.model : undefined,
          job: r.job?.id,
        },
      ])
      if (r.job) await followEdit(r.job.id)
    } catch (e) {
      setMessages((ms) => [...ms, { role: 'assistant', text: e instanceof ApiError ? e.message : String(e), error: true }])
    } finally {
      setBusy(false)
    }
  }

  // An edit runs as a background job: show its steps, then the finished copy (or why nothing was changed).
  const followEdit = async (jobId: string) => {
    const update = (patch: Partial<Message>) =>
      setMessages((ms) => ms.map((m) => (m.job === jobId ? { ...m, ...patch } : m)))
    for (;;) {
      await sleep(700)
      let j
      try {
        j = await api.job(jobId)
      } catch {
        continue
      }
      if (j.status === 'queued') update({ progress: 'Waiting for other work to finish…' })
      else if (j.status === 'running') update({ progress: j.message || 'Working…' })
      else {
        const res = (j.result ?? {}) as { answer?: string; edit?: EditInfo | null }
        if (j.status === 'done') update({ text: res.answer ?? 'Done.', edit: res.edit ?? null, job: undefined, progress: undefined })
        else update({ text: `The edit didn't work: ${j.error ?? j.status}`, error: true, job: undefined, progress: undefined })
        return
      }
    }
  }

  const cov = status?.detector.coverage
  const unchecked = cov ? cov.photos - cov.checked : 0
  const chat = status?.chat_model

  return (
    <section className="ask">
      <div className="ask-status small">
        <div>
          <strong>Object counts:</strong>{' '}
          {cov ? (
            <>
              {cov.checked} of {cov.photos} photos checked
              {job ? (
                <span className="muted"> · counting… {job.done}/{job.total}</span>
              ) : (
                unchecked > 0 && (
                  <button
                    className="btn small primary"
                    onClick={async () => {
                      await api.askPrepare(libs).catch(() => undefined)
                      refresh()
                    }}
                  >
                    Count objects{cov.checked ? ` in ${unchecked} new` : ''}
                  </button>
                )
              )}
            </>
          ) : (
            '…'
          )}
        </div>
        <div>
          <strong>Chat:</strong>{' '}
          {chat?.available ? (
            <span>{chat.model} on this computer</span>
          ) : (
            <span className="muted" title={chat?.reason}>basic answers only ({chat?.reason ?? 'checking…'})</span>
          )}
        </div>
        {messages.length > 0 && (
          <button className="link" onClick={() => setMessages([])}>
            New chat
          </button>
        )}
      </div>

      {status?.detector.error && !job && (
        <div className="banner error" role="alert">
          Counting objects failed: {status.detector.error}
        </div>
      )}
      <div className="ask-log" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask about your photos in plain words. Everything runs on this computer.</p>
            <div className="ask-suggest">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="btn small" onClick={() => send(s)} disabled={busy}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => {
          const all = m.results ?? []
          const shown = expanded.has(i) ? all : all.slice(0, SHOWN)
          return (
            <div key={i} className={`msg ${m.role}${m.error ? ' error' : ''}`}>
              <div className="bubble">
                {m.text}
                {m.via && <div className="muted small">answered with {m.via}</div>}
                {m.job && <div className="muted small">{m.progress ?? 'Starting…'}</div>}
              </div>
              {m.edit && <EditView edit={m.edit} source={all[0]} />}
              {all.length > 0 && (!m.edit || m.edit.op === 'video') && (
                <div className={all.length === 1 && m.role === 'assistant' ? 'msg-photos single' : 'msg-photos'}>
                  {shown.map((a, n) => (
                    <button
                      key={a.id}
                      className={m.maybe?.includes(a.id) ? 'msg-photo maybe' : 'msg-photo'}
                      style={{ aspectRatio: a.width && a.height ? `${a.width} / ${a.height}` : undefined }}
                      title={`${n + 1}. ${a.rel_path}${m.role === 'assistant' ? ' (click to ask about this photo)' : ''}`}
                      onClick={() => m.role === 'assistant' && send(`What's in photo ${n + 1}?`, a)}
                      disabled={busy || m.role !== 'assistant'}
                    >
                      <img src={a.thumbnail_url} alt={a.rel_path} loading="lazy" />
                      {m.boxes?.[a.id]?.map((b, k) => (
                        <span
                          key={k}
                          className="det-box"
                          title={`${b.label} (${b.score.toFixed(2)})`}
                          data-label={all.length === 1 ? b.label : undefined}
                          style={{
                            left: `${b.box[0] * 100}%`,
                            top: `${b.box[1] * 100}%`,
                            width: `${(b.box[2] - b.box[0]) * 100}%`,
                            height: `${(b.box[3] - b.box[1]) * 100}%`,
                          }}
                        />
                      ))}
                      {m.role === 'assistant' && all.length > 1 && <span className="msg-num">{n + 1}</span>}
                      {m.maybe?.includes(a.id) && <span className="msg-maybe">maybe</span>}
                    </button>
                  ))}
                  {all.length > shown.length && (
                    <button className="btn small" onClick={() => setExpanded((s) => new Set(s).add(i))}>
                      +{all.length - shown.length} more
                    </button>
                  )}
                </div>
              )}
            </div>
          )
        })}
        {busy && <div className="msg assistant"><div className="bubble muted">Thinking…</div></div>}
        <div ref={end} />
      </div>

      <form
        className="ask-input"
        onSubmit={(e) => {
          e.preventDefault()
          send(text)
        }}
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          id="ask-q"
          placeholder={`Ask something, e.g. “how many cats?” or “describe the second one” (${FOCUS_HINT})`}
          aria-label="Ask about your photos"
          maxLength={2000}
          autoFocus
        />
        <button className="btn primary" disabled={busy || !text.trim()}>
          Ask
        </button>
      </form>
    </section>
  )
}
