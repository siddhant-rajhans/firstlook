"""Brand theme — a registered Plotly template plus the raw palette.

Importing firstlook registers a ``"firstlook"`` Plotly template, so any figure
(yours or ours) can opt into the dark, high-contrast look with
``fig.update_layout(template="firstlook")``.
"""
import plotly.graph_objects as go
import plotly.io as pio

BG = "#080814"
PANEL = "#0c0c1f"
INK = "#e9e9f4"
MUTED = "#9a9ab0"
GRID = "rgba(255,255,255,0.07)"

CYAN = "#00E5FF"
GOLD = "#FFD27F"
PINK = "#FF4081"
PURPLE = "#7C4DFF"
GREEN = "#00E676"
AMBER = "#FFAB00"

PALETTE = [CYAN, PINK, GOLD, PURPLE, GREEN, AMBER]

ACCENT = {"classification": PINK, "regression": GOLD, "clustering": PURPLE}

THEME = {
    "bg": BG, "panel": PANEL, "ink": INK, "muted": MUTED, "grid": GRID,
    "palette": PALETTE, "cyan": CYAN, "gold": GOLD, "pink": PINK,
    "purple": PURPLE, "green": GREEN, "amber": AMBER,
}


def _register():
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=INK, family="Inter, system-ui, -apple-system, sans-serif"),
        colorway=PALETTE,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        title=dict(font=dict(color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["firstlook"] = tpl


_register()
