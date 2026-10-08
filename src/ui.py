CSS = r"""
<style>
:root { --navy:#0b1930; --ink:#10213d; --muted:#667085; --bg:#f5f7fb; --line:#e6eaf0; }
.stApp { background: var(--bg); }
[data-testid="stSidebar"] { background: var(--navy); }
[data-testid="stSidebar"] * { color: #f8fafc; }
.block-container { max-width: 1480px; padding-top: 1.7rem; padding-bottom: 3rem; }
h1,h2,h3 { color:var(--ink); letter-spacing:-0.02em; }
.kicker { color:#2d6cdf; font-weight:700; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.subtle { color:var(--muted); }
.card { background:white; border:1px solid var(--line); border-radius:14px; padding:16px 18px; box-shadow:0 2px 10px rgba(16,33,61,.04); }
.metric-label { color:var(--muted); font-size:.82rem; font-weight:600; }
.metric-value { color:var(--ink); font-size:1.8rem; font-weight:750; line-height:1.15; }
.badge-high { display:inline-block; background:#fee4e2; color:#b42318; border-radius:999px; padding:4px 9px; font-weight:700; font-size:.78rem; }
.badge-medium { display:inline-block; background:#fef0c7; color:#b54708; border-radius:999px; padding:4px 9px; font-weight:700; font-size:.78rem; }
.badge-low { display:inline-block; background:#eaf2ff; color:#175cd3; border-radius:999px; padding:4px 9px; font-weight:700; font-size:.78rem; }
.risk-reason { background:#f8fafc; border-left:4px solid #2d6cdf; padding:10px 12px; margin:8px 0; border-radius:6px; }
hr { border:none; border-top:1px solid var(--line); }
</style>
"""

def money(v):
    try:
        if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
        return f"${v:,.0f}"
    except Exception:
        return "—"

def number(v):
    try:
        if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"{v/1e3:.1f}K"
        return f"{v:,.0f}"
    except Exception:
        return "—"
