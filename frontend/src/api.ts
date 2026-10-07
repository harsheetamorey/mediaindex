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

export type Segment = { start_s: number | null; end_s: number | null; similarity: number }

export type Result = {
  rank: number
  similarity: number
  modality: string
  start_s: number | null
  end_s: number | null
  asset: Asset
  other_segments?: Segment[]
}

export type MediaType = 'image' | 'audio'


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

export type Selection = { id: string; name: string; item_count?: number; created_at: number; updated_at: number }

export type SelectionItem = {
  id: string
  asset_id: string
  position: number
  start_s: number | null
  end_s: number | null
  status: 'ok' | 'missing' | 'removed' | 'changed'
  thumbnail_url: string | null
  query_context: Record<string, unknown> | null
  snapshot: {
    library_name: string | null
    library_root: string | null
    rel_path: string
    media_type: string
    width: number | null
    height: number | null
    duration: number | null
    source: SourceRecord
  }
}

export type ExportPlan = {
  plan_id: string
  destination: string
  destination_exists: boolean
  files: { item_id: string; dest_name: string; renamed: boolean; bytes: number | null; source: string }[]
  skipped: { item_id: string; rel_path: string; reason: string }[]
  manifest_name: string | null
  total_bytes: number
  overwrites: number
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

  selections: () => request<Selection[]>('/api/selections'),
  createSelection: (name: string) => request<Selection>('/api/selections', json({ name })),
  selection: (id: string) => request<Selection & { items: SelectionItem[] }>(`/api/selections/${id}`),
  renameSelection: (id: string, name: string) =>
    request<Selection>(`/api/selections/${id}`, { ...json({ name }), method: 'PATCH' }),
  deleteSelection: (id: string) => request<{ deleted: string }>(`/api/selections/${id}`, { method: 'DELETE' }),
  addToSelection: (id: string, assetId: string, queryContext: Record<string, unknown> | null, start?: number | null, end?: number | null) =>
    request<{ id: string; added: boolean }>(
      `/api/selections/${id}/items`,
      json({ asset_id: assetId, query_context: queryContext, start_s: start ?? null, end_s: end ?? null }),
    ),
  removeFromSelection: (id: string, itemId: string) =>
    request<{ removed: string }>(`/api/selections/${id}/items/${itemId}`, { method: 'DELETE' }),
  reorderSelection: (id: string, itemIds: string[]) =>
    request<{ ok: boolean }>(`/api/selections/${id}/order`, { ...json({ item_ids: itemIds }), method: 'PUT' }),
  exportPreview: (id: string, destination: string, includeManifest: boolean) =>
    request<ExportPlan>(`/api/selections/${id}/export/preview`, json({ destination, include_manifest: includeManifest })),
  exportRun: (id: string, planId: string) =>
    request<Job>(`/api/selections/${id}/export`, json({ plan_id: planId, confirm: true })),
  assetPath: (id: string) => request<{ path: string }>(`/api/assets/${id}/path`),
  reveal: (id: string) => request<{ revealed: boolean }>(`/api/assets/${id}/reveal`, { method: 'POST' }),

  searchText: (text: string, libraryIds: string[] | null, mediaTypes: MediaType[] = ['image'], limit = 48) =>
    request<SearchResponse>('/api/search/text', json({ text, library_ids: libraryIds, media_types: mediaTypes, limit })),

  searchAudioReference: (
    ref: { file?: File; assetId?: string; startS?: number | null },
    libraryIds: string[] | null,
    includeIdentical: boolean,
    limit = 48,
  ) => {
    const form = new FormData()
    if (ref.file) form.append('file', ref.file)
    if (ref.assetId) form.append('asset_id', ref.assetId)
    if (ref.startS != null) form.append('start_s', String(ref.startS))
    if (libraryIds?.length) form.append('library_ids', libraryIds.join(','))
    form.append('limit', String(limit))
    form.append('include_identical', String(includeIdentical))
    return request<SearchResponse>('/api/search/audio', { method: 'POST', body: form })
  },

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
