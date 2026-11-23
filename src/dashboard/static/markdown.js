/* A small GitHub-flavoured markdown renderer for the dashboard's answers.
 *
 * It is vendored here, not pulled from a CDN: the dashboard has to work with no
 * network, which is the state Rishi's PC is in most of the day.
 *
 * The text it renders comes from a language model, so nothing in it is trusted.
 * Every character of the source is escaped before any tag is produced, tags are
 * emitted only by this file, and the result is then run through `sanitise` in
 * case a future change ever lets markup through. Links are scheme-checked, so a
 * `javascript:` URL renders as plain text.
 *
 * `renderBlocks` returns the answer as a list of top-level HTML strings, one per
 * block, so the page can replace only the block that changed while an answer
 * streams in.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AGXMarkdown = api;
})(typeof globalThis === "undefined" ? this : globalThis, function () {
  "use strict";

  const ESCAPE = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (char) => ESCAPE[char]);
  const unescapeHtml = (text) =>
    String(text)
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/&amp;/g, "&");

  const SAFE_SCHEME = /^(?:https?:|mailto:)/i;
  const SAFE_LOCAL = /^(?:[#/?]|\.{1,2}[\\/])/;

  // A link the model wrote is kept only if it points somewhere harmless.
  function safeHref(raw) {
    const url = String(raw).trim();
    const probe = url.replace(/[\u0000-\u0020\u007f]/g, "").toLowerCase();
    if (!probe) return "";
    if (probe.startsWith("javascript:") || probe.startsWith("vbscript:")) return "";
    if (probe.startsWith("data:")) return "";
    if (SAFE_SCHEME.test(probe) || SAFE_LOCAL.test(probe)) return url;
    return "";
  }

  const HOLD = "\u0001";

  // ── inline ──────────────────────────────────────────────────────
  function inline(source) {
    const held = [];
    const keep = (html) => HOLD + (held.push(html) - 1) + HOLD;

    let text = String(source);
    // Code spans first: their contents are literal and must not see emphasis.
    text = text.replace(/``\s*([\s\S]+?)\s*``/g, (_, code) => keep("<code>" + escapeHtml(code) + "</code>"));
    text = text.replace(/`([^`\n]+)`/g, (_, code) => keep("<code>" + escapeHtml(code) + "</code>"));

    text = escapeHtml(text);

    // Explicit links, then bare URLs. Both are held so the emphasis pass cannot
    // reach the URL or the anchor's own markup.
    text = text.replace(
      /\[([^\]\n]*)\]\(\s*([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\s*\)/g,
      (whole, label, url) => {
        const href = safeHref(unescapeHtml(url));
        if (!href) return escapeHtml(String(label) + " (" + unescapeHtml(String(url)) + ")");
        return keep('<a href="' + escapeHtml(href) + '" rel="noreferrer noopener" target="_blank">' + label + "</a>");
      }
    );

    text = text.replace(/\bhttps?:\/\/[^\s<>"']+/g, (url) => {
      const trimmed = url.replace(/[.,;:!?)\]]+$/, "");
      const tail = url.slice(trimmed.length);
      const href = safeHref(unescapeHtml(trimmed));
      if (!href) return url;
      return keep('<a href="' + escapeHtml(href) + '" rel="noreferrer noopener" target="_blank">' + trimmed + "</a>") + tail;
    });

    text = text.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/(^|[^\w*])\*(?=\S)([^*\n]*?\S)\*(?!\*)/g, "$1<em>$2</em>");
    text = text.replace(/\b__(?=\S)([\s\S]*?\S)__\b/g, "<strong>$1</strong>");
    text = text.replace(/(^|[^\w])_(?=\S)([^_\n]*?\S)_(?!\w)/g, "$1<em>$2</em>");
    text = text.replace(/~~(?=\S)([\s\S]*?\S)~~/g, "<del>$1</del>");
    // A line the model broke on purpose stays broken on screen.
    text = text.replace(/\n/g, "<br>");

    return text.replace(/\u0001(\d+)\u0001/g, (_, index) => held[Number(index)] ?? "");
  }

  // ── blocks ──────────────────────────────────────────────────────
  const FENCE = /^ {0,3}(`{3,}|~{3,})[ \t]*([^`\s]*)[^`]*$/;
  const HEADING = /^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$/;
  const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
  const BULLET = /^( *)([-*+])[ \t]+(.*)$/;
  const NUMBER = /^( *)(\d{1,9})[.)][ \t]+(.*)$/;
  const DELIMITER = /^ {0,3}\|?[ \t]*:?-{1,}:?[ \t]*(?:\|[ \t]*:?-{1,}:?[ \t]*)*\|?[ \t]*$/;

  const isFence = (line) => FENCE.test(line);
  const isRule = (line) => RULE.test(line);
  const isHeading = (line) => HEADING.test(line);
  const isQuote = (line) => /^ {0,3}>/.test(line);
  const isItem = (line) => BULLET.test(line) || NUMBER.test(line);
  const hasPipe = (line) => line.includes("|");

  // A table starts where a row of pipes is followed by its alignment row.
  function isTableStart(lines, index) {
    const head = lines[index];
    const rule = lines[index + 1];
    return hasPipe(head) && head.trim() !== "" && rule !== undefined && DELIMITER.test(rule) && hasPipe(rule);
  }

  function startsBlock(lines, index) {
    const line = lines[index];
    return (
      !line.trim() ||
      isFence(line) ||
      isRule(line) ||
      isHeading(line) ||
      isQuote(line) ||
      isItem(line) ||
      isTableStart(lines, index)
    );
  }

  function splitRow(line) {
    const cells = [];
    let cell = "";
    let inCode = false;
    const body = line.trim().replace(/^\|/, "").replace(/\|$/, "");
    for (let i = 0; i < body.length; i += 1) {
      const char = body[i];
      if (char === "\\" && body[i + 1] === "|") {
        cell += "|";
        i += 1;
      } else if (char === "`") {
        inCode = !inCode;
        cell += char;
      } else if (char === "|" && !inCode) {
        cells.push(cell.trim());
        cell = "";
      } else {
        cell += char;
      }
    }
    cells.push(cell.trim());
    return cells;
  }

  function alignments(rule) {
    return splitRow(rule).map((cell) => {
      const left = cell.startsWith(":");
      const right = cell.endsWith(":");
      if (left && right) return "ta-c";
      if (right) return "ta-r";
      if (left) return "ta-l";
      return "";
    });
  }

  function table(lines, index) {
    const header = splitRow(lines[index]);
    const align = alignments(lines[index + 1]);
    const rows = [];
    let cursor = index + 2;
    while (cursor < lines.length && lines[cursor].trim() !== "" && hasPipe(lines[cursor])) {
      rows.push(splitRow(lines[cursor]));
      cursor += 1;
    }
    // A row still being written keeps the table's shape instead of spilling out.
    const width = header.length;
    const fit = (cells) => cells.slice(0, width).concat(new Array(Math.max(0, width - cells.length)).fill(""));
    const cell = (tag, text, column) => {
      const cls = align[column] ? ' class="' + align[column] + '"' : "";
      return "<" + tag + cls + ">" + inline(text) + "</" + tag + ">";
    };
    let html = "<table><thead><tr>";
    fit(header).forEach((text, column) => (html += cell("th", text, column)));
    html += "</tr></thead>";
    if (rows.length) {
      html += "<tbody>";
      for (const row of rows) {
        html += "<tr>";
        fit(row).forEach((text, column) => (html += cell("td", text, column)));
        html += "</tr>";
      }
      html += "</tbody>";
    }
    html += "</table>";
    return { html, next: cursor };
  }

  function items(lines, index) {
    const entries = [];
    let cursor = index;
    while (cursor < lines.length) {
      const line = lines[cursor];
      const bullet = BULLET.exec(line);
      const number = NUMBER.exec(line);
      if (bullet || number) {
        const match = (bullet || number);
        const entry = { indent: match[1].length, ordered: Boolean(number), text: match[3] };
        // A numbered item after a bullet at the same depth is a new list.
        if (entries.length && entry.indent === 0 && entry.ordered !== entries[entries.length - 1].ordered) {
          break;
        }
        entries.push(entry);
        cursor += 1;
        continue;
      }
      if (!line.trim()) {
        // A blank line only ends the list when nothing list-shaped follows.
        let ahead = cursor + 1;
        while (ahead < lines.length && !lines[ahead].trim()) ahead += 1;
        if (ahead < lines.length && (isItem(lines[ahead]) || /^ {2,}\S/.test(lines[ahead]))) {
          cursor += 1;
          continue;
        }
        break;
      }
      if (/^ {1,}\S/.test(line) && entries.length) {
        entries[entries.length - 1].text += "\n" + line.trim();
        cursor += 1;
        continue;
      }
      break;
    }
    return { html: renderList(buildList(entries)), next: cursor };
  }

  function buildList(entries) {
    const top = { indent: entries[0].indent, ordered: entries[0].ordered, items: [] };
    const stack = [top];
    for (const entry of entries) {
      while (stack.length > 1 && entry.indent < stack[stack.length - 1].indent) stack.pop();
      let list = stack[stack.length - 1];
      if (entry.indent > list.indent && list.items.length) {
        const nested = { indent: entry.indent, ordered: entry.ordered, items: [] };
        list.items[list.items.length - 1].children.push(nested);
        stack.push(nested);
        list = nested;
      }
      list.items.push({ text: entry.text, children: [] });
    }
    return top;
  }

  function renderList(list) {
    const tag = list.ordered ? "ol" : "ul";
    let html = "<" + tag + ">";
    for (const item of list.items) {
      html += "<li>" + inline(item.text);
      for (const child of item.children) html += renderList(child);
      html += "</li>";
    }
    return html + "</" + tag + ">";
  }

  function quote(lines, index) {
    const inner = [];
    let cursor = index;
    while (cursor < lines.length && isQuote(lines[cursor])) {
      inner.push(lines[cursor].replace(/^ {0,3}> ?/, ""));
      cursor += 1;
    }
    return { html: "<blockquote>" + blocks(inner).map((part) => part.html).join("") + "</blockquote>", next: cursor };
  }

  function paragraph(lines, index) {
    const parts = [];
    let cursor = index;
    while (cursor < lines.length && !startsBlock(lines, cursor)) {
      parts.push(lines[cursor].trim());
      cursor += 1;
    }
    return { html: "<p>" + inline(parts.join("\n")) + "</p>", next: cursor };
  }

  function blocks(source) {
    const lines = String(source == null ? "" : source).replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const fence = FENCE.exec(line);
      if (fence) {
        const marker = fence[1];
        const language = fence[2];
        const body = [];
        index += 1;
        // An unclosed fence runs to the end of the text: the rest of a
        // half-written answer is code being typed, never a broken page.
        while (index < lines.length) {
          const closing = new RegExp("^ {0,3}" + marker[0] + "{" + marker.length + ",}[ \\t]*$");
          if (closing.test(lines[index])) {
            index += 1;
            break;
          }
          body.push(lines[index]);
          index += 1;
        }
        const label = language ? ' data-lang="' + escapeHtml(language) + '"' : "";
        out.push({
          type: "code",
          html: "<pre><code" + label + ">" + escapeHtml(body.join("\n")) + "</code></pre>",
        });
        continue;
      }

      const heading = HEADING.exec(line);
      if (heading) {
        const level = heading[1].length;
        out.push({ type: "heading", html: "<h" + level + ">" + inline(heading[2]) + "</h" + level + ">" });
        index += 1;
        continue;
      }

      if (isRule(line)) {
        out.push({ type: "rule", html: "<hr>" });
        index += 1;
        continue;
      }

      if (isQuote(line)) {
        const block = quote(lines, index);
        out.push({ type: "quote", html: block.html });
        index = block.next;
        continue;
      }

      if (isTableStart(lines, index)) {
        const block = table(lines, index);
        out.push({ type: "table", html: block.html });
        index = block.next;
        continue;
      }

      if (isItem(line)) {
        const block = items(lines, index);
        out.push({ type: "list", html: block.html });
        index = block.next;
        continue;
      }

      const block = paragraph(lines, index);
      out.push({ type: "paragraph", html: block.html });
      index = block.next;
    }
    return out;
  }

  // ── sanitiser ───────────────────────────────────────────────────
  // Everything above escapes its input, so this pass should find nothing to do.
  // It exists so that no path can emit markup the dashboard did not build.
  const ALLOWED_TAGS = new Set([
    "a", "b", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "i", "li", "ol", "p", "pre", "s", "strong", "table", "tbody", "td", "th",
    "thead", "tr", "ul",
  ]);
  const DANGEROUS = /<\s*(script|style|iframe|object|embed|svg|math|template|noscript|form|input|link|meta|base)[\s\S]*?<\s*\/\s*\1\s*>/gi;
  const UNPAIRED = /<\s*\/?\s*(script|style|iframe|object|embed|svg|math|template|noscript|form|input|link|meta|base)[^>]*>/gi;
  const TAG = /<\s*(\/?)\s*([a-zA-Z][a-zA-Z0-9]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/g;
  const ATTRIBUTE = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*(?:=\s*("[^"]*"|'[^']*'|[^\s"'=<>`]+))?/g;

  function cleanAttributes(tag, raw) {
    let out = "";
    ATTRIBUTE.lastIndex = 0;
    let match;
    while ((match = ATTRIBUTE.exec(raw))) {
      const name = match[1].toLowerCase();
      const value = (match[2] || "").replace(/^["']|["']$/g, "");
      if (name === "class" || name === "data-lang") {
        out += " " + name + '="' + escapeHtml(value) + '"';
      } else if (tag === "a" && name === "href") {
        const href = safeHref(unescapeHtml(value));
        if (href) out += ' href="' + escapeHtml(href) + '" rel="noreferrer noopener" target="_blank"';
      } else if (tag === "a" && name === "title") {
        out += ' title="' + escapeHtml(value) + '"';
      }
    }
    return out;
  }

  function sanitise(html) {
    let out = String(html);
    out = out.replace(/<!--[\s\S]*?-->/g, "");
    out = out.replace(/<![^>]*>/g, "");
    out = out.replace(DANGEROUS, "");
    out = out.replace(UNPAIRED, "");
    return out.replace(TAG, (whole, closing, name, attrs) => {
      const tag = name.toLowerCase();
      if (!ALLOWED_TAGS.has(tag)) return "";
      if (closing) return "</" + tag + ">";
      return "<" + tag + cleanAttributes(tag, attrs) + ">";
    });
  }

  function render(source) {
    return sanitise(blocks(source).map((part) => part.html).join(""));
  }

  function renderBlocks(source) {
    return blocks(source).map((part) => sanitise(part.html));
  }

  return { render, renderBlocks, sanitise, blocks };
});
