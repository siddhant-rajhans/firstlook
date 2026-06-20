"""In-browser driver for the firstlook playground (runs inside Pyodide).

Reuses the real firstlook: the card HTML and the Plotly figure JSON. Kept tiny
and pure so the same functions work when called from JavaScript via Pyodide.
"""
import io
import json

import pandas as pd

import firstlook as fl
from firstlook.report import _card_html


def columns(csv_text):
    """Return the column names of the uploaded CSV (to populate the target picker)."""
    df = pd.read_csv(io.StringIO(csv_text), nrows=200)
    return json.dumps(list(df.columns))


def run(csv_text, target=None, fit=False, diagnose=True):
    """Run firstlook on the CSV; return JSON with the card HTML and the figure.

    ``fit`` stays False by default so the core path needs only pandas/numpy/plotly
    (no scikit-learn in WASM); the page installs sklearn lazily when fit is on.
    """
    df = pd.read_csv(io.StringIO(csv_text))
    target = target or None
    rep = fl.at(df, target=target, show=False,
                fit=("all" if fit else False), diagnose=diagnose)
    card = _card_html(rep.title, rep.recommendation, rep.shape, rep.target,
                      rep.baseline, rep.leaderboard, rep.narrative, rep.diagnosis)
    return json.dumps({"card": card, "fig": rep.figure.to_json()})
