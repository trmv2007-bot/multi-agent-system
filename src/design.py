"""The design skill: what makes a generated page look considered.

The structurally fixed part is the class palette (container, hero, card, …) so that
sections written independently still compose. Everything a visitor actually sees —
colour, type scale, background, motion — comes from a theme the designer produces
per page. Without a theme every page looks identical, which is exactly what
happens when a build system only varies its words.
"""

import json
import re
from typing import Any

DESIGN_SKILL = """You are designing one page for one subject. Work like a designer, not a decorator.

Rules of the craft:
1. One idea, held. Pick a single visual idea from the subject (a material, a light,
   a rhyme) and let it drive colour, shape and motion. Two ideas read as noise.
2. Contrast before decoration. Most of the page is quiet ground; one thing is loud.
   If everything is emphasised, nothing is.
3. Type carries the page. Set a display size clearly larger than body (3x or more),
   keep body copy at 16-18px with generous line height, and never use more than two
   families.
4. Backgrounds earn their place. A gradient, grain, pattern or illustration should
   say something about the subject — a bakery is warm and floury, an observatory is
   cold and starry. A blank tinted panel is the default, not a decision.
5. Space is the design. Wide margins and consistent vertical rhythm read as
   deliberate; tight even spacing reads as a template.
6. Motion is a courtesy, never a performance. One entrance animation and one hover
   state are plenty. Everything must still work with motion removed.
7. Accessibility is not optional: body text at 4.5:1 contrast against its background,
   visible focus states, meaningful link text.

Anti-patterns to avoid: purple-blue gradient on everything, centred everything,
emoji as iconography, three competing accent colours, cards inside cards, drop
shadows on text, body copy under 15px, "Lorem ipsum"."""

THEME_PROMPT = """You are the designer on this build. Produce the page's visual system.

Use your tools if the subject is unfamiliar: look at how the best work in that
field presents itself before you decide.

Reply with JSON only:
{{
  "feeling": "one line, the idea you are holding",
  "palette": {{"bg": "#..", "surface": "#..", "accent": "#..", "text": "#..", "muted": "#.."}},
  "radius": "6px",
  "display_size": "clamp(2.2rem, 6vw, 4rem)",
  "background_css": "CSS for the page or hero background, e.g. layered gradients",
  "extra_css": "any additional CSS the sections need, using the existing classes",
  "motion_css": "one or two keyframes and transitions, or empty",
  "extras_js": "optional small script for one genuinely useful interaction, or empty"
}}

Rules:
- Colours must be real hex values and must pass 4.5:1 for body text on bg.
- The background must say something about this subject, not be a neutral tint.
- Keep CSS under 60 lines and JS under 25 lines.
- No frameworks, no external URLs, no fonts to download.
- No prose outside the JSON."""

DEFAULT_THEME: dict[str, Any] = {
    "feeling": "quiet and workmanlike",
    "palette": {
        "bg": "#0b0f16",
        "surface": "#131a26",
        "accent": "#f2b544",
        "text": "#eef2f8",
        "muted": "#9fb0c4",
    },
    "radius": "6px",
    "display_size": "clamp(2rem, 5vw, 3.4rem)",
    "background_css": "",
    "extra_css": "",
    "motion_css": "",
    "extras_js": "",
}

COLOUR = re.compile(r"^#[0-9a-fA-F]{3,8}$")

# Used only when the designer does not return usable JSON, so two different
# subjects never produce the same page.
FALLBACK_THEMES: tuple[dict[str, Any], ...] = (
    {
        "feeling": "warm and handmade",
        "palette": {
            "bg": "#14100c",
            "surface": "#1e1712",
            "accent": "#e8a33d",
            "text": "#f7f0e6",
            "muted": "#b4a08a",
        },
        "radius": "10px",
        "display_size": "clamp(2.2rem, 5.6vw, 3.8rem)",
        "background_css": "background: radial-gradient(circle at 15% 0%, #2a1d12, #14100c 65%);",
    },
    {
        "feeling": "cool and exact",
        "palette": {
            "bg": "#070b12",
            "surface": "#0e1520",
            "accent": "#6fb8ff",
            "text": "#e8f0fa",
            "muted": "#8fa2bd",
        },
        "radius": "4px",
        "display_size": "clamp(2rem, 5vw, 3.4rem)",
        "background_css": "background: linear-gradient(160deg, #0b1626, #070b12 70%);",
    },
    {
        "feeling": "green and growing",
        "palette": {
            "bg": "#0a1210",
            "surface": "#101c17",
            "accent": "#63c98a",
            "text": "#eaf6ee",
            "muted": "#93b3a2",
        },
        "radius": "8px",
        "display_size": "clamp(2.1rem, 5.4vw, 3.6rem)",
        "background_css": "background: radial-gradient(circle at 80% 0%, #14261c, #0a1210 60%);",
    },
    {
        "feeling": "editorial and still",
        "palette": {
            "bg": "#f6f3ee",
            "surface": "#fffdf9",
            "accent": "#1f4d3a",
            "text": "#161a17",
            "muted": "#5d6660",
        },
        "radius": "2px",
        "display_size": "clamp(2.4rem, 6vw, 4.2rem)",
        "background_css": "background: #f6f3ee;",
    },
    {
        "feeling": "loud and nightlit",
        "palette": {
            "bg": "#0b0714",
            "surface": "#150e22",
            "accent": "#c86bfa",
            "text": "#f2eafc",
            "muted": "#a795bd",
        },
        "radius": "12px",
        "display_size": "clamp(2.3rem, 6.2vw, 4rem)",
        "background_css": "background: radial-gradient(circle at 70% 20%, #1d1030, #0b0714 65%);",
    },
)


# Subject-matched themes, so a fallback is still relevant rather than random.
SUBJECT_THEMES: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("observatory", "star", "astro", "night sky", "telescope", "space", "planet"),
        {
            "feeling": "cold desert night, stars overhead",
            "palette": {
                "bg": "#050810",
                "surface": "#0b1220",
                "accent": "#7fd4ff",
                "text": "#eaf2ff",
                "muted": "#8fa0bd",
            },
            "radius": "2px",
            "display_size": "clamp(2.4rem, 7vw, 4.4rem)",
            "background_css": "background: radial-gradient(circle at 20% 8%, #14304f 0%, #050810 62%);",
        },
    ),
    (
        ("bakery", "bread", "cake", "pastry", "coffee", "cafe", "roast", "tea"),
        {
            "feeling": "warm, floury, hand-made",
            "palette": {
                "bg": "#140f0a",
                "surface": "#1f1710",
                "accent": "#e6a94b",
                "text": "#f8efe0",
                "muted": "#bda489",
            },
            "radius": "10px",
            "display_size": "clamp(2.3rem, 6vw, 4rem)",
            "background_css": "background: radial-gradient(circle at 15% 0%, #33220f 0%, #140f0a 65%);",
        },
    ),
    (
        ("garden", "plant", "flower", "forest", "farm", "nature", "green", "tea shop"),
        {
            "feeling": "green and growing",
            "palette": {
                "bg": "#08120e",
                "surface": "#0f1e17",
                "accent": "#63c98a",
                "text": "#eaf6ee",
                "muted": "#93b3a2",
            },
            "radius": "8px",
            "display_size": "clamp(2.1rem, 5.4vw, 3.6rem)",
            "background_css": "background: radial-gradient(circle at 80% 0%, #16301f 0%, #08120e 60%);",
        },
    ),
    (
        ("studio", "gallery", "museum", "book", "library", "editorial", "press"),
        {
            "feeling": "editorial and still",
            "palette": {
                "bg": "#f6f3ee",
                "surface": "#fffdf9",
                "accent": "#1f4d3a",
                "text": "#161a17",
                "muted": "#5d6660",
            },
            "radius": "2px",
            "display_size": "clamp(2.4rem, 6vw, 4.2rem)",
            "background_css": "background: #f6f3ee;",
        },
    ),
    (
        ("game", "gaming", "esports", "arcade", "minecraft", "pixel", "vr"),
        {
            "feeling": "loud and nightlit",
            "palette": {
                "bg": "#0b0714",
                "surface": "#150e22",
                "accent": "#c86bfa",
                "text": "#f2eafc",
                "muted": "#a795bd",
            },
            "radius": "12px",
            "display_size": "clamp(2.3rem, 6.2vw, 4rem)",
            "background_css": "background: radial-gradient(circle at 70% 20%, #24103b 0%, #0b0714 65%);",
        },
    ),
    (
        ("gym", "fitness", "training", "sport", "run", "athletic"),
        {
            "feeling": "high contrast, moving",
            "palette": {
                "bg": "#0a0a0c",
                "surface": "#141418",
                "accent": "#ff4f3d",
                "text": "#f7f7f9",
                "muted": "#9d9daa",
            },
            "radius": "4px",
            "display_size": "clamp(2.4rem, 7vw, 4.6rem)",
            "background_css": "background: linear-gradient(165deg, #17171c 0%, #0a0a0c 70%);",
        },
    ),
)


def subject_theme(subject: str) -> dict[str, Any] | None:
    """A theme that matches the subject, when one of the library entries fits."""
    text = str(subject or "").lower()
    for words, theme in SUBJECT_THEMES:
        if any(word in text for word in words):
            merged = dict(DEFAULT_THEME)
            merged.update(theme)
            merged["palette"] = dict(theme["palette"])
            return merged
    return None


def fallback_theme(seed: str) -> dict[str, Any]:
    """A varied theme for when the designer gave nothing usable."""
    index = sum(ord(char) for char in str(seed or "")) % len(FALLBACK_THEMES)
    theme = dict(DEFAULT_THEME)
    theme.update(FALLBACK_THEMES[index])
    theme["palette"] = dict(FALLBACK_THEMES[index]["palette"])
    return theme


def is_default(theme: dict[str, Any] | None) -> bool:
    """Whether a theme is the untouched default (the designer said nothing useful)."""
    if not theme:
        return True
    return bool((theme.get("palette") or {}) == DEFAULT_THEME["palette"])


CAPTURED: dict[str, Any] = {}


def capture_reset() -> None:
    """Clear the theme captured by the designer's tool call."""
    CAPTURED.clear()


def capture_theme(**values: Any) -> dict[str, Any]:
    """Validate and store a theme the designer set through its tool."""
    palette = {
        key: str(values.get(key, DEFAULT_THEME["palette"][key])) for key in DEFAULT_THEME["palette"]
    }
    theme = dict(DEFAULT_THEME)
    theme.update(
        {
            "palette": {
                key: value if COLOUR.match(value) else DEFAULT_THEME["palette"][key]
                for key, value in palette.items()
            },
            "feeling": str(values.get("feeling") or DEFAULT_THEME["feeling"]),
            "radius": str(values.get("radius") or DEFAULT_THEME["radius"]),
            "display_size": str(values.get("display_size") or DEFAULT_THEME["display_size"]),
            "background_css": str(values.get("background_css") or ""),
            "extra_css": str(values.get("extra_css") or ""),
            "motion_css": str(values.get("motion_css") or ""),
            "extras_js": str(values.get("extras_js") or ""),
        }
    )
    CAPTURED.clear()
    CAPTURED.update(theme)
    return theme


def parse_theme(text: str) -> dict[str, Any]:
    """Read the designer's JSON, falling back to the default theme."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return dict(DEFAULT_THEME)

    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(DEFAULT_THEME)

    if not isinstance(raw, dict):
        return dict(DEFAULT_THEME)

    theme = dict(DEFAULT_THEME)
    palette = raw.get("palette")
    if isinstance(palette, dict):
        clean = {
            key: str(value)
            for key, value in palette.items()
            if key in DEFAULT_THEME["palette"] and COLOUR.match(str(value))
        }
        theme["palette"] = {**DEFAULT_THEME["palette"], **clean}

    for key in (
        "feeling",
        "radius",
        "display_size",
        "background_css",
        "extra_css",
        "motion_css",
        "extras_js",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            theme[key] = value.strip()

    return theme


def theme_css(theme: dict[str, Any] | None) -> str:
    """The CSS that makes one page look like itself."""
    if not theme:
        return ""

    palette = theme.get("palette") or DEFAULT_THEME["palette"]
    parts = [
        ":root {",
        f"  --bg: {palette['bg']};",
        f"  --surface: {palette['surface']};",
        f"  --surface-2: {palette['surface']};",
        f"  --accent: {palette['accent']};",
        f"  --text: {palette['text']};",
        f"  --muted: {palette['muted']};",
        f"  --radius: {theme.get('radius', '6px')};",
        "}",
        f"h1 {{ font-size: {theme.get('display_size', DEFAULT_THEME['display_size'])}; }}",
    ]

    background = str(theme.get("background_css") or "").strip()
    if background:
        parts.append("body { " + background + " }")

    for key in ("extra_css", "motion_css"):
        extra = str(theme.get(key) or "").strip()
        if extra:
            parts.append(extra)

    return "\n".join(parts)


def theme_script(theme: dict[str, Any] | None) -> str:
    """The designer's optional interaction, if it gave one."""
    if not theme:
        return ""
    script = str(theme.get("extras_js") or "").strip()
    if not script:
        return ""
    return f"<script>{script}</script>"
