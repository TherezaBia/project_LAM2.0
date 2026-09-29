import { defineConfig } from 'vite'
import basicSsl from '@vitejs/plugin-basic-ssl'
export default defineConfig({
  base: './',
  plugins: [basicSsl()],
  server: { host: '0.0.0.0', port: 4173, strictPort: true, allowedHosts: ['terminal.local'], https: true },
  build: { target: 'es2022', chunkSizeWarningLimit: 6000 },
})
