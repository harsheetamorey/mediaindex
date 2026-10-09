import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, type Asset, type AskStatus, type AskTurn } from '../api'

type Message = {
  role: 'user' | 'assistant'
  text: string
  results?: Asset[]
  action?: string
  via?: string
  error?: boolean
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
      .map((m) => ({ role: m.role, text: m.text, asset_ids: m.results?.map((a) => a.id) ?? [], action: m.action }))
    setMessages((ms) => [...ms, { role: 'user', text: q, results: asset ? [asset] : undefined }])
    setText('')
    setBusy(true)
    try {
      const r = await api.ask(q, history, libs, asset?.id)
      const via = r.described_by === 'chat model' || r.routed_by === 'chat model' || (r.action === 'chat' && r.chat_model.available)
      setMessages((ms) => [
        ...ms,
        { role: 'assistant', text: r.answer, results: r.results, action: r.action, via: via ? r.chat_model.model : undefined },
      ])
    } catch (e) {
      setMessages((ms) => [...ms, { role: 'assistant', text: e instanceof ApiError ? e.message : String(e), error: true }])
    } finally {
      setBusy(false)
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
            <span>{chat.model} on this Mac</span>
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
              </div>
              {all.length > 0 && (
                <div className="msg-photos">
                  {shown.map((a, n) => (
                    <button
                      key={a.id}
                      className="msg-photo"
                      title={`${n + 1}. ${a.rel_path}${m.role === 'assistant' ? ' (click to ask about this photo)' : ''}`}
                      onClick={() => m.role === 'assistant' && send(`What's in photo ${n + 1}?`, a)}
                      disabled={busy || m.role !== 'assistant'}
                    >
                      <img src={a.thumbnail_url} alt={a.rel_path} loading="lazy" />
                      {m.role === 'assistant' && all.length > 1 && <span className="msg-num">{n + 1}</span>}
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
          placeholder="Ask something, e.g. “how many cats?” or “describe the second one”"
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
