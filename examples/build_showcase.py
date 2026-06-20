"""Build (and execute) examples/showcase.ipynb plus the image/HTML gallery.

Run:  python examples/build_showcase.py
Regenerates the executed showcase notebook, the dashboard PNGs in
examples/img/, and the interactive HTML reports in examples/.
"""
import os
import warnings

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = [
    md("# firstlook — the showcase\n\n"
       "From a raw dataframe to **models worth trying**, **the right charts**, and "
       "**a scored baseline** — in one call. Below, the whole arc runs on three real "
       "datasets, next to the boilerplate it replaces."),

    md("## Setup"),
    code("import warnings; warnings.filterwarnings('ignore')\n"
         "import numpy as np, pandas as pd\n"
         "import plotly.io as pio; pio.renderers.default = 'notebook_connected'\n"
         "import firstlook as fl"),

    md("## 1 · Breast cancer — the old way, then one line\n\n"
       "A real diagnostic set: 569 samples, 30 numeric features, malignant vs benign. "
       "Here's the opening you'd normally type by hand."),
    code("from sklearn.datasets import load_breast_cancer\n"
         "bc = load_breast_cancer(as_frame=True).frame\n"
         "bc['target'] = bc['target'].map({0: 'malignant', 1: 'benign'})\n"
         "\n"
         "import matplotlib.pyplot as plt\n"
         "print(bc.shape)\n"
         "print(bc['target'].value_counts())\n"
         "fig, axes = plt.subplots(2, 3, figsize=(11, 5))\n"
         "for ax, col in zip(axes.ravel(), bc.columns[:6]):\n"
         "    ax.hist(bc[col], bins=20); ax.set_title(col, fontsize=8)\n"
         "plt.tight_layout(); plt.show()"),
    md("Six static histograms, no model, no recommendation — and 25 more features still "
       "unlooked-at. You'd still have to decide it's classification, recall which models "
       "fit, scale the features, choose a metric, and write the cross-validation loop.\n\n"
       "The same dataset through firstlook:"),
    code("r = fl.at(bc, target='target', fit=True, title='Breast cancer — diagnosis')"),
    md("One line. It knew the task, ranked the models, drew the four charts that matter, "
       "and returned a cross-validated baseline near 98%. Everything is on the report:"),
    code("r.task, r.start, round(r.baseline.score, 3)"),

    md("## 2 · Diabetes — regression, same call\n\n"
       "Swap to a continuous target (disease progression). The one-liner adapts on its "
       "own: regression models, a feature-vs-target scatter, an R² baseline."),
    code("from sklearn.datasets import load_diabetes\n"
         "db = load_diabetes(as_frame=True).frame\n"
         "r2 = fl.at(db, target='target', fit=True, title='Diabetes — progression')"),

    md("## 3 · Messy data — it catches what bites you\n\n"
       "Real data is rarely clean. This churn set is imbalanced, has a categorical "
       "column, and is missing values — the kind of thing that silently wrecks a naive "
       "pipeline."),
    code("rng = np.random.default_rng(7)\n"
         "n = 1500\n"
         "tenure = rng.integers(1, 72, n)\n"
         "charge = rng.normal(70, 25, n).round(2)\n"
         "plan = rng.choice(['basic', 'plus', 'premium'], n, p=[0.5, 0.3, 0.2])\n"
         "# churn really depends on the features: short tenure, high charge, basic plan\n"
         "logit = -2.2 - 0.03 * tenure + 0.02 * (charge - 70) + np.where(plan == 'basic', 1.2, 0.0)\n"
         "churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)\n"
         "churn = pd.DataFrame({'tenure_months': tenure, 'monthly_charge': charge,\n"
         "                      'plan': plan, 'churned': churned})\n"
         "churn.loc[rng.choice(n, 40, replace=False), 'monthly_charge'] = np.nan  # real-world gaps\n"
         "r3 = fl.at(churn, target='churned', fit=True, title='Customer churn')"),
    md("Read the warnings: it flagged the 85/15 imbalance, the categorical `plan` column "
       "that needs encoding, and the missing values — then scored with **balanced "
       "accuracy**, not the flattering plain accuracy a majority-only guesser would ace. "
       "The impute/scale/one-hot preprocessing happened inside the fit, so it just ran."),

    md("## The pieces, à la carte\n\n"
       "`at()` is the headline, but each stage stands alone."),
    code("print(fl.detect_task(db, target='target'))      # 'regression'\n"
         "print(fl.recommend(churn, target='churned').notes)  # the data-quality warnings\n"
         "fig = fl.visualize(bc, target='target')          # just the figure"),

    md("## Why this is the opening move you want\n\n"
       "From `df` to a scored, charted, model-recommended starting point in one line — no "
       "boilerplate, no half-remembered cheat-sheet, no flattering metric on imbalanced "
       "data. That's the thirty minutes you get back at the start of every project."),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}

print("executing showcase.ipynb ...")
ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": HERE}})
nbf.write(nb, os.path.join(HERE, "showcase.ipynb"))
print("  wrote showcase.ipynb")

# --- gallery: PNG + interactive HTML for each dashboard ---
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import firstlook as fl
from sklearn.datasets import load_breast_cancer, load_diabetes


def _datasets():
    bc = load_breast_cancer(as_frame=True).frame
    bc["target"] = bc["target"].map({0: "malignant", 1: "benign"})
    db = load_diabetes(as_frame=True).frame
    rng = np.random.default_rng(7)
    n = 1500
    tenure = rng.integers(1, 72, n)
    charge = rng.normal(70, 25, n).round(2)
    plan = rng.choice(["basic", "plus", "premium"], n, p=[0.5, 0.3, 0.2])
    logit = -2.2 - 0.03 * tenure + 0.02 * (charge - 70) + np.where(plan == "basic", 1.2, 0.0)
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    churn = pd.DataFrame({"tenure_months": tenure, "monthly_charge": charge,
                          "plan": plan, "churned": churned})
    churn.loc[rng.choice(n, 40, replace=False), "monthly_charge"] = np.nan
    return {
        "breast_cancer": (bc, "target", "Breast cancer — diagnosis"),
        "diabetes": (db, "target", "Diabetes — progression"),
        "churn": (churn, "churned", "Customer churn"),
    }


print("generating gallery (PNG + HTML) ...")
for name, (df, target, title) in _datasets().items():
    rep = fl.at(df, target=target, fit=True, show=False, title=title)
    rep.to_html(os.path.join(HERE, f"{name}.html"))
    rep.figure.write_image(os.path.join(IMG, f"{name}.png"), width=1100, height=640, scale=2)
    print(f"  {name}: {rep.task} | {rep.baseline}")
print("done.")
