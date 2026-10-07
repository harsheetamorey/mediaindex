// Typed client for the local MediaIndex API (same-origin, loopback only).

export type Library = {
  id: string
  name: string
  root_path: string
  asset_count: number
  indexed_count: number
}

export type SourceRecord = {
  dataset?: string | null
  revision?: string | null
  license_declared?: string | null
  source?: string | null
  row?: number | null
  note?: string | null
  inspection_tags?: string | null
} | null

export type Asset = {
  id: string
  library_id: string
  rel_path: string
  media_type: 'image' | 'audio' | 'video'
  size: number | null
  content_hash: string | null
  width: number | null
  height: number | null
  duration: number | null
  status: string
  error: string | null
  source: SourceRecord
  thumbnail_url: string
  file_url: string
}

export type Result = {
  rank: number
  similarity: number
  modality: string
  start_s: number | null
  end_s: number | null
  asset: Asset
}

export type IndexState = {
  state: 'ready' | 'partial' | 'empty' | 'not_indexed' | 'incompatible'
  total: number
  searchable: number
  by_status: Record<string, number>
}

export type SearchResponse = {
  mode: string
  query: Record<string, unknown>
  results: Result[]
  candidates_searched: number
  timing_ms: { query_embedding: number; ranking: number }
  index_state: IndexState
  note: string
}

export type Job = {
  id: string
  kind: string
  library_id: string | null
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted'
  done: number
  total: number
  message: string
  error: string | null
  result: Record<string, unknown> | null
  created: number
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) {
    let detail = r.statusText
    try {
      const body = await r.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(r.status, detail)
  }
  return r.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  libraries: () => request<Library[]>('/api/libraries'),
  addLibrary: (path: string, name?: string) => request<Library>('/api/libraries', json({ path, name })),
  removeLibrary: (id: string) => request<{ deleted: string }>(`/api/libraries/${id}`, { method: 'DELETE' }),
  importLibrary: (id: string) => request<Job>(`/api/libraries/${id}/import`, { method: 'POST' }),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobs: () => request<Job[]>('/api/jobs'),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  asset: (id: string) => request<Asset>(`/api/assets/${id}`),

  searchText: (text: string, libraryIds: string[] | null, limit = 48) =>
    request<SearchResponse>('/api/search/text', json({ text, library_ids: libraryIds, limit })),

  searchReference: (
    ref: { file?: File; assetId?: string },
    text: string,
    libraryIds: string[] | null,
    includeIdentical: boolean,
    limit = 48,
  ) => {
    const form = new FormData()
    if (ref.file) form.append('file', ref.file)
    if (ref.assetId) form.append('asset_id', ref.assetId)
    if (libraryIds?.length) form.append('library_ids', libraryIds.join(','))
    form.append('limit', String(limit))
    form.append('include_identical', String(includeIdentical))
    const trimmed = text.trim()
    if (trimmed) form.append('text', trimmed)
    const url = trimmed ? '/api/search/image-text' : '/api/search/image'
    return request<SearchResponse>(url, { method: 'POST', body: form })
  },
}
