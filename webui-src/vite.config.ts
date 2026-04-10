import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: path.resolve(__dirname, '../webui'),
    emptyOutDir: false,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8766',
      '/ws': { target: 'ws://localhost:8766', ws: true },
    },
  },
})
