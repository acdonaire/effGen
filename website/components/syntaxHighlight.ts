// Minimal zero-dependency syntax highlighter for Python and Bash code blocks
// on the marketing landing pages. Returns an HTML string with <span class>
// wrappers consumed by the .syntax-code styles in globals.css.

const PY_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await", "break",
  "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
  "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
  "or", "pass", "raise", "return", "try", "while", "with", "yield",
]);

const PY_BUILTINS = new Set([
  "print", "len", "range", "list", "dict", "set", "tuple", "str", "int",
  "float", "bool", "bytes", "open", "isinstance", "type", "super",
]);

const BASH_KEYWORDS = new Set([
  "if", "then", "else", "elif", "fi", "for", "in", "do", "done", "while",
  "case", "esac", "function", "return", "break", "continue", "export", "set",
  "unset", "alias", "local", "readonly", "echo",
]);

const escape = (s: string) =>
  s.replace(/&/g, "&amp;")
   .replace(/</g, "&lt;")
   .replace(/>/g, "&gt;");

const span = (cls: string, text: string) =>
  `<span class="${cls}">${escape(text)}</span>`;

function highlightPython(src: string): string {
  // Token regex: order matters. Long-lived idea: walk through chunks and emit
  // either a styled span or plain text.
  const re =
    /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?''')|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\bf"(?:[^"\\\n]|\\.)*"|\bf'(?:[^'\\\n]|\\.)*')|(\b\d+(?:\.\d+)?\b)|(@\w+)|(\b[A-Z][A-Za-z0-9_]*\b)|(\b[A-Za-z_][A-Za-z0-9_]*\b)|([^\w\s])/g;

  let out = "";
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    if (m.index > lastIndex) {
      out += escape(src.slice(lastIndex, m.index));
    }
    const [token,
      comment, tripleStr, str, fstr, num, decorator, capWord, word, sym] = m;
    if (comment) out += span("syn-comment", token);
    else if (tripleStr) out += span("syn-string", token);
    else if (str) out += span("syn-string", token);
    else if (fstr) out += span("syn-string", token);
    else if (num) out += span("syn-number", token);
    else if (decorator) out += span("syn-decorator", token);
    else if (capWord) out += span("syn-class", token);
    else if (word) {
      if (PY_KEYWORDS.has(word)) out += span("syn-keyword", token);
      else if (PY_BUILTINS.has(word)) out += span("syn-builtin", token);
      else {
        // function call heuristic: lookahead for "("
        const next = src[m.index + token.length];
        if (next === "(") out += span("syn-function", token);
        else out += escape(token);
      }
    } else if (sym) {
      if ("()[]{}".includes(sym)) out += span("syn-punct", token);
      else if (",.;:".includes(sym)) out += span("syn-punct", token);
      else if ("=+-*/%<>!&|^~".includes(sym)) out += span("syn-operator", token);
      else out += escape(token);
    } else {
      out += escape(token);
    }
    lastIndex = m.index + token.length;
  }
  if (lastIndex < src.length) out += escape(src.slice(lastIndex));
  return out;
}

function highlightBash(src: string): string {
  const lines = src.split("\n");
  return lines.map((line) => {
    if (line.trimStart().startsWith("#")) {
      return span("syn-comment", line);
    }
    // Match: leading whitespace, then a sequence of word|string|number|other.
    const re = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\$\w+|\$\{[^}]+\})|(\b\d+(?:\.\d+)?\b)|(--?[A-Za-z][\w-]*)|(\b[A-Za-z_][\w-]*\b)|([|&;><])|([^\w\s])/g;
    let out = "";
    let last = 0;
    let firstToken = true;
    let m: RegExpExecArray | null;
    while ((m = re.exec(line)) !== null) {
      if (m.index > last) out += escape(line.slice(last, m.index));
      const [tok, str, vari, num, flag, word, redir, sym] = m;
      if (str) out += span("syn-string", tok);
      else if (vari) out += span("syn-variable", tok);
      else if (num) out += span("syn-number", tok);
      else if (flag) out += span("syn-flag", tok);
      else if (word) {
        if (BASH_KEYWORDS.has(word)) out += span("syn-keyword", tok);
        else if (firstToken) out += span("syn-function", tok);
        else out += escape(tok);
        firstToken = false;
      } else if (redir) out += span("syn-operator", tok);
      else if (sym) out += escape(tok);
      else out += escape(tok);
      if (!/^\s+$/.test(tok)) firstToken = false;
      last = m.index + tok.length;
    }
    if (last < line.length) out += escape(line.slice(last));
    return out;
  }).join("\n");
}

export function highlightCode(code: string, language: string): string {
  if (language === "python" || language === "py") return highlightPython(code);
  if (language === "bash" || language === "sh" || language === "shell") return highlightBash(code);
  return escape(code);
}
