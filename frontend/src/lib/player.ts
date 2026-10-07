import { useCallback, useEffect, useRef, useState } from 'react'

/** One shared audio element that plays a [start, end) segment of a local file and stops at `end`. */
export function useSegmentPlayer() {
  const audio = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState<string | null>(null)
  const [position, setPosition] = useState(0)
  const endRef = useRef<number | null>(null)

  useEffect(() => {
    const a = new Audio()
    a.preload = 'auto'
    a.id = 'mi-segment-player' // kept in the DOM (hidden) so state is inspectable and accessible
    a.hidden = true
    document.body.appendChild(a)
    audio.current = a
    const onTime = () => {
      setPosition(a.currentTime)
      if (endRef.current != null && a.currentTime >= endRef.current) {
        a.pause()
        setPlaying(null)
      }
    }
    const onEnd = () => setPlaying(null)
    a.addEventListener('timeupdate', onTime)
    a.addEventListener('ended', onEnd)
    return () => {
      a.pause()
      a.removeEventListener('timeupdate', onTime)
      a.removeEventListener('ended', onEnd)
      a.remove()
    }
  }, [])

  const stop = useCallback(() => {
    audio.current?.pause()
    setPlaying(null)
  }, [])

  const play = useCallback(async (key: string, url: string, start: number | null, end: number | null) => {
    const a = audio.current
    if (!a) return
    if (playing === key) {
      a.pause()
      setPlaying(null)
      return
    }
    a.pause()
    if (!a.src.endsWith(url)) a.src = url
    endRef.current = end
    const seek = () => {
      a.currentTime = start ?? 0
    }
    if (a.readyState >= 1) seek()
    else await new Promise<void>((res) => a.addEventListener('loadedmetadata', () => (seek(), res()), { once: true }))
    setPlaying(key)
    try {
      await a.play()
    } catch {
      setPlaying(null)
    }
  }, [playing])

  return { play, stop, playing, position }
}
