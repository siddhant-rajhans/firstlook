"""firstlook — drop in any dataframe, get models to try and the right charts.

    import firstlook
    firstlook.at(df, target="species")

That one call detects the problem type, recommends models (with reasons and
data-aware warnings), and renders a dark, interactive Plotly dashboard. It's
the first look you take at any dataset, done for you.
"""
from . import theme  # noqa: F401  (registers the "firstlook" Plotly template on import)
from .detect import detect_task
from .recommend import recommend, Recommendation
from .visualize import visualize
from .report import play, Report
from .baseline import fit_baseline, Baseline
from .leaderboard import leaderboard, Leaderboard, Entry
from .explain import explain, Narrative, Completer
from .diagnose import diagnose, Diagnosis, Finding
from .theme import THEME

at = play  # headline alias: firstlook.at(df, target=...)

__version__ = "0.4.0"
__all__ = [
    "at", "play", "recommend", "visualize", "detect_task", "fit_baseline",
    "leaderboard", "explain", "diagnose", "Report", "Recommendation", "Baseline",
    "Leaderboard", "Entry", "Narrative", "Completer", "Diagnosis", "Finding",
    "THEME", "__version__",
]
