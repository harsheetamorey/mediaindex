export function fmtTime(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return '–'
  const m = Math.floor(s / 60)
  const sec = s - m * 60
  return `${m}:${sec.toFixed(sec % 1 ? 1 : 0).padStart(sec % 1 ? 4 : 2, '0')}`
}

export function fmtSpan(start: number | null | undefined, end: number | null | undefined): string {
  return `${fmtTime(start)}–${fmtTime(end)}`
}
