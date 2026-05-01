# SGLDS Frontend (Vite + React + TypeScript)

Status:
- Connected to the Phase 1 backend API.
- Production build verified with `npm run build`.
- Current pages cover dashboard, search, analyze, gallery, and settings workflows.

Quick start:

```bash
cd frontend
npm install
npm run dev
```

The dev server runs on http://localhost:3000 and proxies to the backend at `/api/v1` (assumes backend on same host). Adjust `src/api/client.ts` if backend runs on a different host/port.

Pages:
- `/` Dashboard page — reads `/api/v1/stats`, `/api/v1/results/stats`, `/api/v1/training/history`.
- `/search` Search page — calls `/api/v1/euclid/search`, `/api/v1/euclid/fetch`, `/api/v1/euclid/images`.
- `/analyze` Analyze page — starts analysis via `/api/v1/analyze/image/{id}`, polls `/api/v1/analyze/runs/{run_id}`, reads history.
- `/gallery` Results gallery — reads `/api/v1/results/gallery` and report/heatmap links.
- `/settings` Config editor — reads and updates `/api/v1/config`.

Environment options:
- `VITE_API_BASE_URL` to override Axios base URL (default `/api/v1`).
- `VITE_BACKEND_ORIGIN` for dev proxy target in `vite.config.ts` (default `http://localhost:8000`).

Next steps:
- Add WebSocket streaming for real-time stage progress.
- Add auth-aware route guards once backend auth/RBAC is implemented.
- Add frontend tests and a CI check for the build.
