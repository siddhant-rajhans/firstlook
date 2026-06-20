"""``play()`` — the one call that ties detection, recommendation and charts together."""
from dataclasses import dataclass

from . import theme
from .detect import detect_task
from .recommend import recommend, Recommendation
from .visualize import visualize


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _card_html(title, rec, shape, target):
    accent = theme.ACCENT.get(rec.task, theme.CYAN)
    models = "".join(
        f'<li><span style="color:{theme.CYAN};font-family:monospace;font-weight:600">{m}</span>'
        f' <span style="color:#b9b9cc">{r}</span></li>' for m, r in rec.models)
    notes = "".join(f'<li style="color:#cfcfe0"><span style="color:{accent}">&rarr; </span>{n}</li>'
                    for n in rec.notes) or '<li style="color:#cfcfe0">clean and ready to model.</li>'
    return f"""<div style="background:{theme.BG};color:{theme.INK};font-family:Inter,system-ui,sans-serif;
 border-radius:12px;padding:18px 20px;max-width:680px">
 <div style="font-size:20px;font-weight:800;letter-spacing:-.02em">{title}</div>
 <div style="margin:6px 0 14px;font-size:13px;color:#8a8aa0">
   <span style="background:{accent};color:#08080f;font-weight:700;font-size:11px;padding:3px 11px;
    border-radius:999px;text-transform:uppercase;letter-spacing:.08em">{rec.task}</span>
   &nbsp; {shape[0]} rows &middot; {shape[1]} cols &middot; target: <code>{target}</code></div>
 <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div><div style="font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#8a8aa0;
    margin-bottom:8px">models to try</div>
   <ul style="list-style:none;margin:0;padding:0;font-size:13.5px;line-height:1.5">{models}</ul>
   <div style="margin-top:10px;font-size:13px;color:#8a8aa0">start with
    <span style="color:{accent};font-weight:700">{rec.start}</span></div></div>
  <div><div style="font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#8a8aa0;
    margin-bottom:8px">watch out for</div>
   <ul style="list-style:none;margin:0;padding:0;font-size:13.5px;line-height:1.5">{notes}</ul></div>
 </div></div>"""


@dataclass
class Report:
    title: str
    task: str
    recommendation: Recommendation
    figure: object
    shape: tuple
    target: str

    @property
    def start(self):
        return self.recommendation.start

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
            display(HTML(_card_html(self.title, self.recommendation, self.shape, self.target)))
            self.figure.show()
        else:
            print(self.title)
            print(self.recommendation)
        return self

    def to_html(self, path):
        """Write a standalone dark dashboard (recommendations + charts) to ``path``."""
        chart = self.figure.to_html(full_html=False, include_plotlyjs="cdn",
                                    config={"displayModeBar": False})
        card = _card_html(self.title, self.recommendation, self.shape, self.target)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{self.title}</title></head>'
                    f'<body style="margin:0;background:{theme.BG}">'
                    f'<div style="max-width:1080px;margin:0 auto;padding:24px">{card}{chart}</div>'
                    f"</body></html>")
        return path

    def _repr_html_(self):
        return _card_html(self.title, self.recommendation, self.shape, self.target)


def play(df, target=None, title=None, show=True):
    """Inspect ``df``: detect the task, recommend models, and chart the data.

    Returns a :class:`Report` (``.task``, ``.models``, ``.notes``, ``.figure``,
    ``.to_html(path)``). In a notebook it also displays a card and the charts.
    """
    task = detect_task(df, target)
    rec = recommend(df, target, task)
    fig = visualize(df, target, task)
    report = Report(title=title or "your data", task=task, recommendation=rec,
                    figure=fig, shape=df.shape, target=target)
    if show:
        report.show()
    return report
