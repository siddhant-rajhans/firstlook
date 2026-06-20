# Playground smoke test

Pyodide (with pandas/scikit-learn) doesn't run reliably in a headless CI
screenshot, so verify the playground in a real browser:

1. Build the wheel into `web/` and serve:
   ```
   python -m build --wheel && cp dist/*.whl web/
   cd web && python -m http.server 8000
   ```
2. Open http://localhost:8000. The status line should progress:
   `booting → loading data libraries → installing firstlook → ready`.
3. Click **try the sample** (iris). A target dropdown appears (default: `species`).
4. Click **Analyze**. Within a second or two you should see the recommendation
   card (task = classification, models, the data-doctor block) and the dark
   Plotly dashboard.
5. Tick **fit a leaderboard** and Analyze again. After a one-time scikit-learn
   load, the card shows the ranked leaderboard.
6. Drop your own CSV — confirm the target picker populates from its columns.

First load is ~10–20s (Pyodide + libraries download once); later analyses are
near-instant.
