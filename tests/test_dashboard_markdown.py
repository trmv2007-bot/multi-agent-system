"""The dashboard's markdown renderer, exercised as a unit under node.

``src/dashboard/static/markdown.js`` is what turns an agent's answer into the
blocks a person reads. It is JavaScript because the dashboard renders answers in
the browser as they stream in, and it is vendored rather than fetched from a CDN
because the page has to work with no network.

Node loads it directly: the file exports through ``module.exports`` when there is
no ``window``, so the same code the browser runs is the code these assertions
run. The renderer is optional on this machine (a contributor without node skips
rather than fails); the browser job in CI covers the page itself.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

RENDERER = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "static" / "markdown.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is needed to run the markdown renderer")

HARNESS = """
const md = require(process.argv[2]);
const cases = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const out = {};
for (const [name, [fn, text]] of Object.entries(cases)) out[name] = md[fn](text);
process.stdout.write(JSON.stringify(out));
"""

ANSWER = (
    "**Quickest:** the direct route wins.\n\n"
    "| Path | Time | Note |\n"
    "| --- | :---: | --- |\n"
    "| A | 2s | direct |\n"
    "| B | 5s | cheap |\n\n"
    "```bash\n"
    "git status\n"
    "```\n"
)

CASES: dict[str, list[str]] = {
    "bold_table_code": ["render", ANSWER],
    "script_tag": ["render", "<script>alert(1)</script>"],
    "onerror": ["render", "<img src=x onerror=alert(1)>"],
    "javascript_link": ["render", "[click](javascript:alert(1))"],
    "sanitise_script": ["sanitise", "<p>before<script>alert(1)</script>after</p>"],
    "sanitise_onerror": ["sanitise", '<p><img src=x onerror="alert(1)">after</p>'],
    "sanitise_javascript": ["sanitise", '<a href="javascript:alert(1)">click</a>'],
    "unclosed_fence": ["renderBlocks", "Steps so far:\n\n```python\nfor item in queue:"],
}


# An answer streamed in one character at a time must only ever change its last
# block: that is what lets the page replace one node instead of the whole reply.
STREAM_HARNESS = """
const md = require(process.argv[2]);
const answer = require("fs").readFileSync(process.argv[3], "utf8");
let previous = [];
let changed = 0;
let blank = 0;
for (let i = 1; i <= answer.length; i += 1) {
  const text = answer.slice(0, i);
  const parts = md.renderBlocks(text);
  if (text.trim() && !parts.length) blank += 1;
  for (let k = 0; k < Math.min(previous.length, parts.length) - 1; k += 1) {
    if (previous[k] !== parts[k]) { changed += 1; break; }
  }
  previous = parts;
}
process.stdout.write(JSON.stringify({ changed, blank }));
"""


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Every case, rendered once by the renderer the page loads."""
    work = tmp_path_factory.mktemp("markdown")
    script = work / "render.js"
    script.write_text(HARNESS, encoding="utf-8")
    cases = work / "cases.json"
    cases.write_text(json.dumps(CASES), encoding="utf-8")
    done = subprocess.run(
        [NODE, str(script), RENDERER.as_posix(), str(cases)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    loaded: dict[str, Any] = json.loads(done.stdout)
    return loaded


def test_bold_table_and_code_render_as_elements(rendered: dict[str, Any]) -> None:
    """A normal answer becomes markup, and no markdown syntax survives as text."""
    html = rendered["bold_table_code"]
    assert "<strong>Quickest:</strong>" in html
    assert "<table>" in html and "</table>" in html
    assert html.count("</th>") == 3
    assert html.count("</td>") == 6
    assert '<code data-lang="bash">git status</code>' in html
    assert "**" not in html
    assert "| --- |" not in html


def test_a_script_tag_is_text_not_markup(rendered: dict[str, Any]) -> None:
    """Model output cannot open an element the page did not build."""
    html = rendered["script_tag"]
    assert "<script" not in html
    assert "&lt;script&gt;" in html


def test_an_onerror_attribute_is_removed(rendered: dict[str, Any]) -> None:
    """An image tag with a handler never becomes an image tag."""
    html = rendered["onerror"]
    assert "<img" not in html
    assert "&lt;img" in html


def test_a_javascript_link_loses_its_href(rendered: dict[str, Any]) -> None:
    """A `javascript:` target degrades to text, keeping the words a person needs."""
    html = rendered["javascript_link"]
    assert "<a" not in html
    assert "href" not in html
    assert "click" in html


def test_sanitiser_strips_markup_outside_its_allow_list(rendered: dict[str, Any]) -> None:
    """The last line of defence: a script tag and its content do not survive."""
    assert "<script" not in rendered["sanitise_script"]
    assert "before" in rendered["sanitise_script"]
    assert "after" in rendered["sanitise_script"]


def test_sanitiser_keeps_only_safe_attributes(rendered: dict[str, Any]) -> None:
    """`onerror` and a `javascript:` href are dropped; the content stays."""
    assert "onerror" not in rendered["sanitise_onerror"]
    assert "after" in rendered["sanitise_onerror"]
    assert "javascript:" not in rendered["sanitise_javascript"]
    assert "click" in rendered["sanitise_javascript"]


def test_an_unclosed_fence_still_renders_the_text_after_it(rendered: dict[str, Any]) -> None:
    """A half-written fence becomes code to the end, never a message that vanishes."""
    blocks: list[str] = rendered["unclosed_fence"]
    assert blocks[0] == "<p>Steps so far:</p>"
    assert blocks[1].startswith("<pre><code")
    assert "for item in queue:" in blocks[1]
    assert len(blocks) == 2


def test_a_streamed_answer_only_changes_its_last_block(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Every character of the answer renders without touching the blocks above it."""
    work = tmp_path_factory.mktemp("stream")
    script = work / "stream.js"
    script.write_text(STREAM_HARNESS, encoding="utf-8")
    answer = work / "answer.md"
    answer.write_text(ANSWER + "\n> Nothing here left this machine.", encoding="utf-8")
    done = subprocess.run(
        [NODE, str(script), RENDERER.as_posix(), str(answer)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["changed"] == 0
    assert result["blank"] == 0
