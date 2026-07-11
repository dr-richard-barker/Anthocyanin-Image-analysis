import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative base ('./') so the built site works under a GitHub Pages project
// subpath (https://<user>.github.io/<repo>/) as well as at a domain root.
export default defineConfig({
  base: './',
  server: {
    port: 3000,
    host: '0.0.0.0',
  },
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
});
