// Typed client for the local MediaIndex API (same-origin, loopback only).

export type Library = {
  id: string
  name: string
  root_path: string
  asset_count: number
  indexed_count: number
  watch: number
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

export type Segment = { start_s: number | null; end_s: number | null; similarity: number; window_thumbnail_url?: string }

export type Result = {
  rank: number
  similarity: number
  modality: string
  start_s: number | null
  end_s: number | null
  asset: Asset
  other_segments?: Segment[]
  window_thumbnail_url?: string
  overlapping_windows_merged?: number
}

export type MediaType = 'image' | 'audio' | 'video'
export type Target = MediaType | 'all'

export type ClipPlan = {
  plan_id: string
  start_s: number
  end_s: number
  duration_s: number
  mode: 'accurate' | 'fast'
  effective_start_s: number
  destination: string
  dest_name: string
  manifest_name: string
  overwrites: number
  note: string
}

export type QueryMode = { query: string; target: string; status: string; available: boolean }


export type IndexState = {
  state: 'ready' | 'partial' | 'empty' | 'not_indexed' | 'incompatible'
  total: number
  searchable: number
  by_status: Record<string, number>
}

export type SearchResponse = {
  mode: string
  media_types?: MediaType[]
  grouping?: 'single' | 'by_modality' | 'global (experimental)'
  groups?: Record<string, number>
  query: Record<string, unknown>
  results: Result[]
  candidates_searched: number
  timing_ms: { query_embedding: number; ranking: number }
  index_state: IndexState | Record<string, IndexState>
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
  eta_seconds?: number | null
}

export type AppSettings = { low_priority_indexing: boolean; watch_interval_s: number }

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

export type AskTurn = { role: 'user' | 'assistant'; text: string; asset_ids?: string[]; action?: string }

export type AskResponse = {
  answer: string
  action: 'count' | 'search' | 'refine' | 'describe' | 'explain' | 'chat'
  routed_by: string
  described_by?: string
  asset_ids: string[]
  maybe_ids?: string[]
  results: Asset[]
  boxes: Record<string, DetectedBox[]>
  chat_model: { available: boolean; model?: string; reason?: string }
}

/** Where the detector found an object; box = [x0, y0, x1, y1] as fractions of the photo's width and height. */
export type DetectedBox = { label: string; score: number; box: [number, number, number, number] }

export type AskStatus = {
  detector: { model: string; coverage: { photos: number; checked: number }; job: Job | null }
  chat_model: { available: boolean; model?: string; reason?: string }
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
  askStatus: (libraryIds?: string[]) =>
    request<AskStatus>('/api/ask/status' + (libraryIds?.length ? `?library_ids=${libraryIds.join(',')}` : '')),
  askPrepare: (libraryIds?: string[]) => request<Job>('/api/ask/prepare', json({ library_ids: libraryIds?.length ? libraryIds : null })),
  ask: (message: string, history: AskTurn[], libraryIds?: string[], assetId?: string) =>
    request<AskResponse>(
      '/api/ask',
      json({ message, history, library_ids: libraryIds?.length ? libraryIds : null, asset_id: assetId ?? null }),
    ),
  libraries: () => request<Library[]>('/api/libraries'),
  addLibrary: (path: string, name?: string) => request<Library>('/api/libraries', json({ path, name })),
  setLibraryWatch: (id: string, watch: boolean) =>
    request<Library>(`/api/libraries/${id}`, { ...json({ watch }), method: 'PATCH' }),
  pickFolder: () => request<{ cancelled: boolean; path?: string }>('/api/pick-folder', { method: 'POST' }),
  settings: () => request<AppSettings>('/api/settings'),
  setLowPriority: (on: boolean) =>
    request<AppSettings>('/api/settings', { ...json({ low_priority_indexing: on }), method: 'PATCH' }),
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

  searchText: (text: string, libraryIds: string[] | null, mediaTypes: MediaType[] = ['image'], limit = 48, globalRank = false) =>
    request<SearchResponse>(
      '/api/search/text',
      json({ text, library_ids: libraryIds, media_types: mediaTypes, limit, global_rank: globalRank }),
    ),
  videoWindow: (assetId: string, startS: number, target: MediaType, libraryIds: string[] | null, includeSameVideo = false) =>
    request<SearchResponse>(
      '/api/search/video-window',
      json({ asset_id: assetId, start_s: startS, target, library_ids: libraryIds, include_same_video: includeSameVideo, limit: 48 }),
    ),
  clipPreview: (assetId: string, start: number, end: number, mode: 'accurate' | 'fast', destination: string, queryContext: Record<string, unknown> | null) =>
    request<ClipPlan>(
      `/api/assets/${assetId}/clip/preview`,
      json({ start_s: start, end_s: end, mode, destination, query_context: queryContext }),
    ),
  clipExport: (assetId: string, planId: string) => request<Job>(`/api/assets/${assetId}/clip`, json({ plan_id: planId, confirm: true })),
  capabilities: () => request<{ modes: QueryMode[]; notes: string[] }>('/api/capabilities'),

  searchAudioReference: (
    ref: { file?: File; assetId?: string; startS?: number | null },
    text: string,
    libraryIds: string[] | null,
    includeIdentical: boolean,
    target: MediaType = 'audio',
    limit = 48,
  ) => {
    const form = new FormData()
    if (ref.file) form.append('file', ref.file)
    if (ref.assetId) form.append('asset_id', ref.assetId)
    if (ref.startS != null) form.append('start_s', String(ref.startS))
    if (text.trim()) form.append('text', text.trim())
    form.append('target', target)
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
    target: MediaType = 'image',
    limit = 48,
  ) => {
    const form = new FormData()
    if (ref.file) form.append('file', ref.file)
    if (ref.assetId) form.append('asset_id', ref.assetId)
    form.append('target', target)
    if (libraryIds?.length) form.append('library_ids', libraryIds.join(','))
    form.append('limit', String(limit))
    form.append('include_identical', String(includeIdentical))
    const trimmed = text.trim()
    if (trimmed) form.append('text', trimmed)
    const url = trimmed ? '/api/search/image-text' : '/api/search/image'
    return request<SearchResponse>(url, { method: 'POST', body: form })
  },
}
