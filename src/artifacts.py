"""Assembling generated HTML into one viewable page.

Build mode keeps assembly deterministic: workers write section fragments, this
module wraps them in a shared stylesheet, and the dashboard re-assembles after
every fragment so the preview grows while the run is still going.
"""

from typing import Any

BASE_CSS = """
:root {
  --bg: #0b0f16;
  --surface: #131a26;
  --surface-2: #19222f;
  --line: #27303f;
  --text: #eef2f8;
  --muted: #9fb0c4;
  --accent: #f2b544;
  --accent-ink: #1a1200;
  --radius: 6px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 16px/1.65 "Segoe UI", system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.container { width: min(1040px, calc(100% - 2.5rem)); margin-inline: auto; }
h1, h2, h3 { line-height: 1.15; margin: 0; letter-spacing: -0.015em; }
h1 { font-size: clamp(2rem, 5vw, 3.4rem); }
h2 { font-size: clamp(1.4rem, 3vw, 2rem); }
p { color: var(--muted); }
.hero { padding: clamp(3.5rem, 9vw, 7rem) 0 clamp(2.5rem, 6vw, 4rem); position: relative; overflow: hidden; }
.hero .sub { margin: 1.2rem 0 0; max-width: 60ch; font-size: clamp(1rem, 1.6vw, 1.15rem); }
.hero .actions { display: flex; flex-wrap: wrap; gap: .8rem; margin-top: 2rem; }
.btn {
  display: inline-block; padding: .85rem 1.5rem; border-radius: var(--radius);
  background: var(--accent); color: var(--accent-ink); font-weight: 700;
  text-decoration: none; border: 0;
}
.btn.ghost { background: transparent; color: var(--text); box-shadow: inset 0 0 0 1px var(--line); }
.eyebrow {
  display: inline-block; margin: 0 0 1rem; font-size: 12px; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase; color: var(--accent);
}
section { padding: clamp(2.5rem, 6vw, 4.5rem) 0; }
.features { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-top: 2rem; }
.card {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1.4rem 1.3rem;
}
.card h3 { font-size: 1.05rem; }
.card p { margin: .6rem 0 0; font-size: .95rem; }
.quotes { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin-top: 2rem; }
blockquote {
  margin: 0; padding: 1.4rem 1.3rem; background: var(--surface-2);
  border-left: 3px solid var(--accent); border-radius: var(--radius);
}
blockquote footer { margin-top: .8rem; font-size: .85rem; color: var(--muted); }
.stat { margin-top: 2rem; font-size: clamp(1.6rem, 4vw, 2.4rem); font-weight: 700; color: var(--accent); }
.stat span { color: var(--muted); font-size: 1rem; font-weight: 400; display: block; margin-top: .3rem; }
.cta { text-align: center; background: linear-gradient(180deg, #0e1622, var(--bg)); border-top: 1px solid var(--line); }
.cta p { max-width: 46ch; margin: 1rem auto 1.8rem; }
footer.site { border-top: 1px solid var(--line); padding: 1.6rem 0; font-size: .82rem; color: var(--muted); }
""".strip()

SECTION_GUIDE = """Write one section of a web page as an HTML fragment.

Rules:
- Return only HTML: no <html>, <head>, <body>, no <style>, no markdown fences.
- Use these classes, already styled: container, hero, sub, actions, btn, btn ghost,
  eyebrow, features, card, quotes, blockquote, stat, cta.
- Use <h1> only if this is the hero; otherwise <h2> for the section heading.
- Real copy, specific and finished. No lorem ipsum, no placeholders, no TODOs.
- 40-120 words of prose per section beyond the heading. Keep it tight."""


OVERLAY_CSS = """
.build-section { position: relative; }
.build-section > .agx-outline {
  position: absolute; inset: 6px; border: 2px dashed #f2b544; border-radius: 6px;
  opacity: 0; transition: opacity .3s ease; pointer-events: none;
}
.build-section.agx-active > .agx-outline { opacity: .75; }
.agx-cursor {
  position: fixed; left: 0; top: 0; z-index: 9999; opacity: 0;
  transition: transform .55s cubic-bezier(.4, 0, .2, 1), opacity .3s ease;
  pointer-events: none;
}
.agx-cursor svg { display: block; filter: drop-shadow(0 2px 5px #000a); }
.agx-chip {
  position: absolute; left: 17px; top: 15px; padding: 6px 9px; border-radius: 6px;
  background: #0d1421f2; border: 1px solid #ffffff2e; color: #eef2f7;
  font: 600 11px/1.2 ui-monospace, "Cascadia Mono", Consolas, monospace;
  white-space: nowrap; box-shadow: 0 6px 20px #0007;
}
.agx-dot {
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: #f2b544; margin-right: 6px; vertical-align: middle;
  animation: agx-pulse 1.2s ease-in-out infinite;
}
@keyframes agx-pulse { 0%, 100% { opacity: 1 } 50% { opacity: .25 } }
"""

OVERLAY_JS = """
(function () {
  var cursor = document.createElement('div');
  // Kept for the single-cursor message; every real cursor is made on demand.
  cursor.style.display = 'none';
  document.body.appendChild(cursor);

  function sections() {
    return Array.prototype.slice.call(document.querySelectorAll('.build-section'));
  }

  // One cursor lives on each section a worker is touching, so several workers
  // show up at once instead of a single dot hopping between them.
  var cursors = {};

  function makeCursor(index) {
    var el = document.createElement('div');
    el.className = 'agx-cursor';
    el.innerHTML =
      '<svg width="18" height="18" viewBox="0 0 24 24">' +
      '<path d="M4 2l15 9.5-6.6.9 3.6 7-2.7 1.4-3.6-7L4 17z" fill="#f2b544" stroke="#0b0f16" stroke-width="1"/>' +
      '</svg><span class="agx-chip"></span>';
    document.body.appendChild(el);
    cursors[index] = el;
    return el;
  }

  function place(el, index, label, status, busy) {
    var list = sections();
    var target = list[index];
    if (!target) { el.style.opacity = 0; return; }

    if (!target.querySelector('.agx-outline')) {
      var outline = document.createElement('div');
      outline.className = 'agx-outline';
      target.appendChild(outline);
    }
    target.classList.add('agx-active');

    var rect = target.getBoundingClientRect();
    el.style.opacity = busy ? 1 : 0.55;
    el.style.transform =
      'translate(' + (rect.left + 10 + (index % 3) * 26) + 'px,' + (rect.top + 10 + (index % 2) * 22) + 'px)';
    el.querySelector('.agx-chip').innerHTML =
      '<i class="agx-dot"></i>' + label + (status ? ' \u00b7 ' + status : '');
    if (!busy) el.querySelector('.agx-chip').style.opacity = 0.7;
  }

  function focus(index, label, status) {
    setCursors([{ index: index, label: label, status: status, busy: true }]);
  }

  function setCursors(list) {
    var keep = {};
    (list || []).forEach(function (item) {
      keep[item.index] = true;
      var el = cursors[item.index] || makeCursor(item.index);
      place(el, item.index, item.label || 'worker', item.status || '', item.busy !== false);
    });
    Object.keys(cursors).forEach(function (key) {
      if (!keep[key]) {
        cursors[key].remove();
        delete cursors[key];
        var el = sections()[key];
        if (el) el.classList.remove('agx-active');
      }
    });
    cursor.style.display = 'none';
  }

  function clear() {
    Object.keys(cursors).forEach(function (key) {
      cursors[key].remove();
      delete cursors[key];
    });
    cursor.style.display = 'none';
    sections().forEach(function (el) { el.classList.remove('agx-active'); });
  }

  window.addEventListener('message', function (event) {
    var data = event.data || {};
    if (data.type === 'agx-focus') focus(data.index, data.label, data.status);
    else if (data.type === 'agx-cursors') setCursors(data.cursors);
    else if (data.type === 'agx-clear') clear();
  });
})();
"""


def assemble_page(
    title: str,
    fragments: list[str],
    description: str = "",
    overlay: bool = False,
    theme: dict[str, Any] | None = None,
) -> str:
    """Wrap section fragments in a complete, styled document.

    ``overlay`` adds the agent cursor the dashboard drives with messages, so a
    preview shows which section is being written right now.
    """
    clean = [f.strip() for f in fragments if f and f.strip()]
    if overlay:
        body = "\n\n".join(
            f'<div class="build-section" data-index="{index}">{_strip_wrappers(fragment)}</div>'
            for index, fragment in enumerate(clean)
        )
        extras = f"<style>{OVERLAY_CSS}</style>\n<script>{OVERLAY_JS}</script>"
    else:
        body = "\n\n".join(_strip_wrappers(f) for f in clean)
        extras = ""

    if not body:
        body = '<section class="container"><p>Waiting for the first section…</p></section>'

    meta = f'<meta name="description" content="{_attr(description)}">' if description else ""

    from src.design import theme_css, theme_script

    palette_css = theme_css(theme)
    extra_script = theme_script(theme)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_text(title)}</title>
{meta}
<style>
{BASE_CSS}
{palette_css}
</style>
</head>
<body>
{body}
<footer class="site"><div class="container">Built live by the multi-agent team.</div></footer>
{extras}
{extra_script}
</body>
</html>
"""


def artifact_event(state: Any, title: str) -> str:
    """The current page for a partially finished build."""
    fragments = [
        str(item.get("content", ""))
        for item in sorted(state.subtask_results, key=lambda r: r.get("index", 0))
    ]
    return assemble_page(title, fragments)


def _strip_wrappers(fragment: str) -> str:
    """Remove fences or stray document tags a model may add around a fragment."""
    text = fragment.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    lowered = text.lower()
    for tag in ("</body>", "</html>"):
        if lowered.endswith(tag):
            text = text[: -len(tag)].rstrip()
    if lowered.startswith("<body>"):
        text = text[len("<body>") :].lstrip()
    return text


def _text(value: str) -> str:
    """Escape text bound for HTML content."""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _attr(value: str) -> str:
    """Escape text bound for an HTML attribute."""
    return _text(value).replace('"', "&quot;")
