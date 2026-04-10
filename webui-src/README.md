# WebUI Source

React + TypeScript + Tailwind CSS control panel.

## Development

```bash
cd webui-src
npm install
npm run dev
```

Dev server proxies `/api` and `/ws` to `localhost:8766` (the Python backend).

## Build

```bash
npm run build
```

Outputs to `../webui/` which is served by the FastAPI backend.
