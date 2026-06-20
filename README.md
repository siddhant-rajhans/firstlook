# firstlook

Drop in any dataframe — get models to try and the right charts, instantly. The first look you take at any dataset, done for you.

```python
import firstlook
firstlook.at(df, target="species")
```

One call and you get three things back:

1. **The problem type** — regression, classification, or clustering, inferred from the data.
2. **Models to try** — ranked, with a one-line reason each, plus data-aware warnings (class imbalance, categoricals that need encoding, missing values, too few rows).
3. **The right charts** — a dark, interactive Plotly dashboard that picks the chart per column: bar/histogram for the target, a scatter colored by class (or feature-vs-target for regression), a correlation heatmap, and a closer look.

It works in Jupyter (rich card + interactive charts render inline) and in plain scripts (prints the recommendation; `report.to_html("out.html")` for the visuals).

## Why

Every project starts the same way: load the data, squint at it, remember which chart goes with which column, half-remember the sklearn cheat-sheet. `firstlook` does that opening move for you so you can get to the actual modeling — without writing a wall of matplotlib.

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
report.figure        # the Plotly figure — restyle or export it
report.to_html("iris.html")
```

Need just one piece?

```python
firstlook.detect_task(df, target="price")   # "regression"
firstlook.recommend(df, target="price")     # a Recommendation (task, start, models, notes)
firstlook.visualize(df, target="price")     # a Plotly figure
```

The dark theme is also a registered Plotly template — use it on your own figures:

```python
fig.update_layout(template="firstlook")
```

(`firstlook.at` and `firstlook.play` are the same call.)

## What it handles today

Tabular regression, classification, and clustering. Image / text / time-series problems are out of scope for now.

## Roadmap

- A quick baseline fit (train the recommended model, report a score) behind an optional `scikit-learn` extra.
- More chart types (pair plots, missingness maps, target-vs-time).
- A light/lab theme alongside the dark one.

## License

MIT
