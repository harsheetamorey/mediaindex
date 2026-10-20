export function fmtTime(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return '–'
  const m = Math.floor(s / 60)
  const sec = s - m * 60
  return `${m}:${sec.toFixed(sec % 1 ? 1 : 0).padStart(sec % 1 ? 4 : 2, '0')}`
}

export function fmtSpan(start: number | null | undefined, end: number | null | undefined): string {
  return `${fmtTime(start)}–${fmtTime(end)}`
}

const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

/** "2024-07" -> "July 2024" */
export function monthLabel(month: string): string {
  const [y, m] = month.split('-')
  return `${MONTH_NAMES[Number(m) - 1]} ${y}`
}
