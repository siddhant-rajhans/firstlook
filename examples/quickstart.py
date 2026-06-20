"""firstlook quickstart — run me:  python examples/quickstart.py

Detects the task, prints model recommendations, and writes an interactive
dashboard to iris.html. In a Jupyter notebook, `firstlook.at(...)` renders the
card and charts inline instead.
"""
import firstlook
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True).frame
iris["target"] = iris["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

report = firstlook.at(iris, target="target", title="Iris - flower species")
report.to_html("iris.html")
print("\nwrote iris.html - open it for the interactive dashboard")
