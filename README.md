<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/siddhant-rajhans/firstlook@main/assets/logo.svg?v=2" alt="firstlook" width="440">
</p>

<p align="center">
  Drop in any dataframe and get the models worth trying, plus the right charts.<br>
  <em>The first look you take at any dataset — done for you.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/firstlook/"><img src="https://img.shields.io/pypi/v/firstlook?style=flat-square&color=00B8D4" alt="PyPI version"></a>
  <a href="https://pypi.org/project/firstlook/"><img src="https://img.shields.io/pypi/pyversions/firstlook?style=flat-square&color=1E88E5" alt="Python versions"></a>
  <a href="https://github.com/siddhant-rajhans/firstlook/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/siddhant-rajhans/firstlook/ci.yml?branch=main&style=flat-square&label=tests" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-MIT-success?style=flat-square" alt="MIT license">
</p>

```python
import firstlook
firstlook.at(df, target="species")
```

One call and you get three things back:

1. **The problem type:** regression, classification, or clustering, inferred from the data.
2. **Models to try:** ranked, with a one-line reason each, plus data-aware warnings (class imbalance, categoricals that need encoding, missing values, too few rows).
3. **The right charts:** a dark, interactive Plotly dashboard that picks the chart per column: bar/histogram for the target, a scatter colored by class (or feature-vs-target for regression), a correlation heatmap, and a closer look.

It works in Jupyter (rich card + interactive charts render inline) and in plain scripts (prints the recommendation; `report.to_html("out.html")` for the visuals).

**See it on real data:** [`examples/`](examples/) runs the full understand → visualize → model arc on three datasets — breast cancer, diabetes, and a messy churn set — with the dashboards rendered.

## Why

Every project starts the same way: load the data, squint at it, remember which chart goes with which column, half-remember the sklearn cheat-sheet. `firstlook` does that opening move for you so you can get to the actual modeling without writing a wall of matplotlib.

## Install

```bash
pip install firstlook
```

From source, for development:

```bash
git clone https://github.com/siddhant-rajhans/firstlook
cd firstlook
pip install -e ".[dev]"
```

## Use

```python
import firstlook
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True).frame
report = firstlook.at(iris, target="target")

report.task          # "classification"
report.start         # "LogisticRegression"
report.models        # [("LogisticRegression", "..."), ...]
report.notes         # ["3 classes", ...]
report.figure        # the Plotly figure (restyle or export it)
report.to_html("iris.html")
```

Need just one piece?

```python
firstlook.detect_task(df, target="price")   # "regression"
firstlook.recommend(df, target="price")     # a Recommendation (task, start, models, notes)
firstlook.visualize(df, target="price")     # a Plotly figure
```

The dark theme is also a registered Plotly template you can use on your own figures:

```python
fig.update_layout(template="firstlook")
```

(`firstlook.at` and `firstlook.play` are the same call.)

## Get a baseline score, too

Pass `fit=True` and `firstlook` trains the recommended model and cross-validates it, so the recommendation comes with a real score attached. Preprocessing (impute, scale, one-hot) is built in, so it fits straight on messy data:

```python
report = firstlook.at(df, target="price", fit=True)
report.baseline      # Baseline(model="LinearRegression", metric="R2", score=0.97, ...)
```

Needs scikit-learn: `pip install "firstlook[fit]"`.

## Or fit a whole leaderboard

`fit="all"` fits and ranks a panel of models — Linear/Logistic, RandomForest, HistGradientBoosting, KNeighbors — through the same pipeline and the same honest metric, by cross-validation:

```python
report = firstlook.at(df, target="churned", fit="all")
report.leaderboard   # ranked entries; report.baseline is the winner
```

On imbalanced data it weights the classes and scores with balanced accuracy, so the ranking is honest rather than flattering.

## Explain my data, with any LLM

`explain=` adds a plain-English read of the *structured* report (task, key features, warnings, scores) — never your raw rows. It's provider-agnostic and the core ships no LLM SDK:

```python
firstlook.at(df, target="species", explain="ollama:llama3")      # local, private
firstlook.at(df, target="species", explain="openai:gpt-4o-mini") # or any cloud
firstlook.at(df, target="species", explain=my_completer)         # or your own (prompt) -> text
firstlook.at(df, target="species", explain=True)                 # auto-detect (env / local Ollama)
```

Local-first by default; keys are read from the environment and never logged. A failed call degrades to no summary rather than raising.

## Data doctor

`diagnose=True` flags the problems that quietly wreck models — target leakage, ID-like and constant columns, near-duplicate rows, heavy skew/outliers — with a fix for each:

```python
report = firstlook.at(df, target="y", diagnose=True)
report.diagnosis     # Diagnosis(findings=[Finding(kind="leakage", severity="high", ...), ...])
```

## See gradient descent

`descent=True` draws the picture every ML course sketches by hand — the 3-D loss surface, the path gradient descent takes down it, and the loss-per-step curve — fit live on your data. It trains a tiny two-parameter model (bias + the feature most tied to the target) so the loss is a real surface: linear regression / MSE for a regression target, logistic regression / log-loss for a binary one.

```python
report = firstlook.at(df, target="price", descent=True)
report.descent                                    # Descent(model="linear regression", final_loss=..., ...)

firstlook.descent(df, target="price")             # or call it on its own
firstlook.descent(df, target="y", feature="age")  # pick which parameter to vary
```

Pure numpy — no scikit-learn needed. The surface is a clean convex bowl because that's the honest shape of a linear/logistic loss; the bumpy landscapes you've seen elsewhere belong to deep nets.

## What it handles today

Tabular regression, classification, and clustering. Image / text / time-series problems are out of scope for now.

## Roadmap

- More chart types (pair plots, missingness maps, target-vs-time).
- A light/lab theme alongside the dark one.
- Image / text / time-series support beyond tabular.

## License

MIT
