import { defineConfig } from 'vite';
const headers={
  'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
  'X-Content-Type-Options':'nosniff',
  'Referrer-Policy':'no-referrer'
};
export default defineConfig({preview:{headers}});
