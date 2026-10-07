import { useEffect, useState } from 'react'

type Health = { status: string; version: string }

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch((e) => setError(String(e)))
  }, [])

  return (
    <main style={{ padding: 24 }}>
      <h1>MediaIndex</h1>
      {health && <p>Backend {health.status} (v{health.version})</p>}
      {error && <p role="alert">Backend unreachable: {error}</p>}
    </main>
  )
}
