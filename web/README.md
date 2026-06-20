# firstlook playground (browser)

A no-install web playground: drop a CSV and get the firstlook report — task,
model recommendations, the dashboard, and a data-quality check — running
entirely in your browser via [Pyodide](https://pyodide.org). Nothing is uploaded.

## Run locally

The page loads the *real* firstlook from a wheel hosted next to it, so build the
wheel into this folder first:

```bash
python -m build --wheel
cp dist/*.whl web/
cd web && python -m http.server 8000
```

Open http://localhost:8000 and drop a CSV (or click "try the sample").

## Deploy

`.github/workflows/pages.yml` builds the wheel into `web/` and publishes the
folder to GitHub Pages. One-time setup: Settings → Pages → Source: **GitHub
Actions**, then run the **pages** workflow (Actions tab → Run workflow).

## How it works

- `index.html` — the page (Pyodide + Plotly.js from CDN).
- `app.js` — boots Pyodide, `micropip`-installs the firstlook wheel + plotly,
  reads the dropped CSV, renders the card + Plotly figure.
- `driver.py` — runs inside Pyodide; reuses firstlook's `_card_html` and
  `figure.to_json()`. The core path (detect → recommend → visualize → diagnose)
  needs only pandas/numpy/plotly; scikit-learn loads lazily when you tick
  "fit a leaderboard".

When you bump the package version, update `WHEEL` at the top of `app.js`.
