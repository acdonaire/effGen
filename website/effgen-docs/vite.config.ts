import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Where the built docs are served from. Under a project page that is
  // /<repo>/docs/; the default keeps a root-hosted site working.
  base: process.env.DOCS_BASE_PATH || '/docs/',
  build: {
    outDir: path.resolve(__dirname, '../public/docs'),
    emptyOutDir: true,
  },
})
