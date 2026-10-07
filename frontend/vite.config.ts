import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev server is loopback-only and proxies the API to the local backend.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
})
