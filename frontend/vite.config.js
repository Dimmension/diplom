import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET
if (!apiProxyTarget) {
  throw new Error('VITE_API_PROXY_TARGET is required')
}

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/health': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
})
