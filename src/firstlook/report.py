"""``play()`` / ``at()`` — the one call that ties it all together."""
from dataclasses import dataclass, field

from . import theme
from .detect import detect_task
from .recommend import recommend, Recommendation
from .visualize import visualize, _numeric_feats, _rank_features
from .baseline import fit_baseline, Baseline
from .leaderboard import leaderboard as _run_leaderboard
from .explain import explain as _run_explain, Narrative
from .diagnose import diagnose as _run_diagnose, Diagnosis


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _leaderboard_html(lb, accent):
    if lb is None or not getattr(lb, "entries", None):
        return ""
    scores = [e.score for e in lb.entries]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) if hi > lo else 1.0
    rows = ""
    for e in lb.entries:
        w = 6 + 94 * (e.score - lo) / span
        bar_color = accent if e.rank == 1 else "#6a6a86"
        rows += (
            '<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px">'
            f'<span style="color:#8a8aa0;width:12px">{e.rank}</span>'
            f'<span style="font-family:monospace;color:{theme.INK};width:170px">{e.model}</span>'
            '<span style="flex:1;background:#11112a;border-radius:4px;overflow:hidden">'
            f'<span style="display:block;height:8px;width:{w:.0f}%;background:{bar_color}"></span></span>'
            f'<span style="color:{theme.INK};width:52px;text-align:right">{e.score:.3f}</span></div>'
        )
    return (
        '<div style="margin-top:14px"><div style="font-size:12px;letter-spacing:.1em;'
        'text-transform:uppercase;color:#8a8aa0;margin-bottom:6px">leaderboard &middot; '
        f'{lb.metric} ({lb.cv}-fold CV)</div>{rows}</div>'
    )


def _findings_html(diag, accent):
    if diag is None or not getattr(diag, "findings", None):
        return ""
    dot = {"high": theme.PINK, "medium": theme.GOLD, "low": "#8a8aa0"}
    rows = ""
    for f in diag.findings:
        sug = f' <span style="color:#8a8aa0">- {f.suggestion}</span>' if f.suggestion else ""
        rows += ('<div style="margin:4px 0;font-size:13px;color:#cfcfe0">'
                 f'<span style="color:{dot.get(f.severity, "#8a8aa0")}">&#9679;</span> '
                 f'{f.message}{sug}</div>')
    return ('<div style="margin-top:14px"><div style="font-size:12px;letter-spacing:.1em;'
            'text-transform:uppercase;color:#8a8aa0;margin-bottom:6px">data doctor</div>'
            f'{rows}</div>')


def _card_html(title, rec, shape, target, baseline=None, lb=None, narrative=None, diagnosis=None):
    accent = theme.ACCENT.get(rec.task, theme.CYAN)
    diag = _findings_html(diagnosis, accent)
    narr = ""
    if narrative is not None and getattr(narrative, "text", ""):
        safe = (narrative.text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace(chr(10), "<br>"))
        prov = (f' &middot; <span style="color:#6a6a86">{narrative.model}</span>'
                if narrative.model else "")
        narr = (f'<div style="margin:0 0 14px;padding:12px 14px;border-left:3px solid {theme.CYAN};'
                f'background:#0f0f22;border-radius:0 8px 8px 0;font-size:13.5px;color:#cfcfe0">'
                f'<div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;'
                f'color:#8a8aa0;margin-bottom:6px">AI summary{prov}</div>{safe}</div>')
    models = "".join(
        f'<li><span style="color:{theme.CYAN};font-family:monospace;font-weight:600">{m}</span>'
        f' <span style="color:#b9b9cc">{r}</span></li>' for m, r in rec.models)
    notes = "".join(f'<li style="color:#cfcfe0"><span style="color:{accent}">&rarr; </span>{n}</li>'
                    for n in rec.notes) or '<li style="color:#cfcfe0">clean and ready to model.</li>'
    if lb is not None and getattr(lb, "entries", None):
        base = _leaderboard_html(lb, accent)
    elif baseline is not None and baseline.score is not None:
        base = (f'<div style="margin-top:14px;padding:10px 14px;background:#11112a;border-radius:8px;'
                f'font-size:13.5px"><span style="color:#8a8aa0">baseline</span> &nbsp;'
                f'<span style="color:{theme.CYAN};font-family:monospace">{baseline.model}</span> &middot; '
                f'{baseline.metric} <span style="color:{accent};font-weight:700">{baseline.score:.3f}</span> '
                f'<span style="color:#8a8aa0">&plusmn;{baseline.std:.3f} ({baseline.cv}-fold CV)</span></div>')
    elif baseline is not None and baseline.note:
        base = f'<div style="margin-top:14px;font-size:13px;color:#8a8aa0">baseline: {baseline.note}</div>'
    else:
        base = ""
    return f"""<div style="background:{theme.BG};color:{theme.INK};font-family:Inter,system-ui,sans-serif;
 border-radius:12px;padding:18px 20px;max-width:680px">
 <div style="font-size:20px;font-weight:800;letter-spacing:-.02em">{title}</div>
 <div style="margin:6px 0 14px;font-size:13px;color:#8a8aa0">
   <span style="background:{accent};color:#08080f;font-weight:700;font-size:11px;padding:3px 11px;
    border-radius:999px;text-transform:uppercase;letter-spacing:.08em">{rec.task}</span>
   &nbsp; {shape[0]} rows &middot; {shape[1]} cols &middot; target: <code>{target}</code></div>
 {narr}<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div><div style="font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#8a8aa0;
    margin-bottom:8px">models to try</div>
   <ul style="list-style:none;margin:0;padding:0;font-size:13.5px;line-height:1.5">{models}</ul>
   <div style="margin-top:10px;font-size:13px;color:#8a8aa0">start with
    <span style="color:{accent};font-weight:700">{rec.start}</span></div></div>
  <div><div style="font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#8a8aa0;
    margin-bottom:8px">watch out for</div>
   <ul style="list-style:none;margin:0;padding:0;font-size:13.5px;line-height:1.5">{notes}</ul></div>
 </div>{base}{diag}</div>"""


@dataclass
class Report:
    title: str
    task: str
    recommendation: Recommendation
    figure: object
    shape: tuple
    target: str
    baseline: object = None
    leaderboard: object = None
    narrative: object = None
    key_features: list = field(default_factory=list)
    diagnosis: object = None

    @property
    def start(self):
        return self.recommendation.start

    def explain(self, using=None, max_features=5):
        """Narrate this report with any LLM; store and return the Narrative."""
        self.narrative = _run_explain(self, using=using, max_features=max_features)
        return self.narrative

    @property
    def models(self):
        return self.recommendation.models

    @property
    def notes(self):
        return self.recommendation.notes

    def show(self):
        """Render in a notebook (rich card + interactive charts) or print (scripts)."""
        if _in_notebook():
            from IPython.display import HTML, display
            display(HTML(_card_html(self.title, self.recommendation, self.shape,
                                    self.target, self.baseline, self.leaderboard, self.narrative, self.diagnosis)))
            self.figure.show()
        else:
            print(self.title)
            if self.narrative is not None:
                print("\n" + str(self.narrative) + "\n")
            print(self.recommendation)
            if self.leaderboard is not None and getattr(self.leaderboard, "entries", None):
                print("\n" + str(self.leaderboard))
            elif self.baseline is not None:
                print("\n" + str(self.baseline))
            if self.diagnosis is not None and self.diagnosis.findings:
                print("\n" + str(self.diagnosis))
        return self

    def to_html(self, path):
        """Write a standalone dark dashboard (recommendations + charts) to ``path``."""
        chart = self.figure.to_html(full_html=False, include_plotlyjs="cdn",
                                    config={"displayModeBar": False})
        card = _card_html(self.title, self.recommendation, self.shape, self.target,
                          self.baseline, self.leaderboard, self.narrative, self.diagnosis)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{self.title}</title></head>'
                    f'<body style="margin:0;background:{theme.BG}">'
                    f'<div style="max-width:1080px;margin:0 auto;padding:24px">{card}{chart}</div>'
                    f"</body></html>")
        return path

    def _repr_html_(self):
        return _card_html(self.title, self.recommendation, self.shape, self.target,
                          self.baseline, self.leaderboard, self.narrative, self.diagnosis)


def _resolve_fit(fit):
    """Map the ``fit`` argument to None | 'baseline' | 'all'."""
    if fit in (False, None):
        return None
    if fit is True or fit == "baseline":
        return "baseline"
    if fit in ("all", "leaderboard"):
        return "all"
    return None


def _baseline_from_leaderboard(lb):
    """Synthesize a Baseline from the leaderboard winner so .baseline stays populated."""
    if lb is None:
        return None
    if not lb.entries:
        return Baseline(None, None, None, note=lb.note)
    best = lb.best
    return Baseline(model=best.model, metric=lb.metric, score=best.score,
                    std=best.std, cv=lb.cv)


def play(df, target=None, title=None, show=True, fit=False, explain=False, diagnose=False):
    """Inspect ``df``: detect the task, recommend models, and chart the data.

    ``fit=True`` cross-validates the single recommended baseline; ``fit="all"``
    fits and ranks a whole model panel (a leaderboard). Both need scikit-learn
    (``pip install 'firstlook[fit]'``).

    ``explain=`` adds a plain-English narrative from any LLM: ``True`` auto-detects
    one (env vars / local Ollama), or pass a ``"provider:model"`` string, a config
    dict, or your own ``(prompt) -> text`` callable. A failed narration degrades to
    ``None`` rather than raising.

    Returns a :class:`Report` (``.task``, ``.start``, ``.models``, ``.notes``,
    ``.figure``, ``.baseline``, ``.leaderboard``, ``.narrative``, ``.to_html(path)``).
    """
    task = detect_task(df, target)
    rec = recommend(df, target, task)
    fig = visualize(df, target, task)

    num = _numeric_feats(df, target)
    key_features = _rank_features(df, target, num, task)[:5] if num else []

    baseline = lb = None
    mode = _resolve_fit(fit)
    if mode == "baseline":
        baseline = fit_baseline(df, target, task)
    elif mode == "all":
        lb = _run_leaderboard(df, target, task)
        baseline = _baseline_from_leaderboard(lb)

    report = Report(title=title or "your data", task=task, recommendation=rec,
                    figure=fig, shape=df.shape, target=target,
                    baseline=baseline, leaderboard=lb, key_features=key_features)

    if explain not in (False, None):
        using = None if explain is True else explain
        try:
            report.narrative = _run_explain(report, using=using)
        except Exception as e:  # never let narration break the report
            import warnings
            warnings.warn(f"explain= skipped: {e}")

    if diagnose:
        try:
            report.diagnosis = _run_diagnose(df, target, task)
        except Exception as e:  # never let diagnostics break the report
            import warnings
            warnings.warn(f"diagnose= skipped: {e}")

    if show:
        report.show()
    return report
