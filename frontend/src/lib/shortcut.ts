/** Keyboard shortcut to jump to the search or message box: ⌘K (Ctrl+K off the Mac), or / when not typing. */

export const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
export const FOCUS_HINT = IS_MAC ? '⌘K' : 'Ctrl+K'

/** True while the user is typing somewhere, so a bare "/" must be left alone. */
export function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false
  if (t.isContentEditable) return true
  if (t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return true
  if (!(t instanceof HTMLInputElement)) return false
  return !['checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'range', 'color'].includes(t.type)
}

/** Which key press asks for the box: ⌘K / Ctrl+K anywhere, "/" only outside text fields. */
export function isFocusShortcut(e: KeyboardEvent): boolean {
  if (e.altKey || e.shiftKey) return false
  if (e.key.toLowerCase() === 'k' && (IS_MAC ? e.metaKey && !e.ctrlKey : e.ctrlKey && !e.metaKey)) return true
  return e.key === '/' && !e.metaKey && !e.ctrlKey && !isTypingTarget(e.target)
}
