/** Which system the browser runs on: for shortcut names, example paths and file-manager hints. */

const ua = typeof navigator !== 'undefined' ? navigator.platform || navigator.userAgent : ''
export const IS_MAC = /Mac|iPhone|iPad/.test(ua)
export const IS_WINDOWS = /Win/.test(ua)

/** An example absolute path under the user's home folder, e.g. "Pictures/project". */
export function examplePath(sub: string): string {
  return IS_WINDOWS ? `C:\\Users\\you\\${sub.replaceAll('/', '\\')}` : `/Users/you/${sub}`
}

export const COPY_PATH_HINT = IS_WINDOWS
  ? 'in File Explorer: hold Shift, right-click it and choose “Copy as path”'
  : IS_MAC
    ? 'in Finder: select it and press ⌥⌘C'
    : 'from your file manager'
