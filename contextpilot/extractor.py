"""Tree-sitter symbol extraction for Python, JavaScript, TypeScript, and Go.

Extracts structured Symbol objects from source files using tree-sitter AST
parsing. Gracefully handles syntax errors and unsupported languages.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import xxhash

# Language detection by file extension
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
}

# Tree-sitter language loaders — imported lazily per language
_LANGUAGE_CACHE: dict[str, object] = {}

_JS_TS_IMPORT_EXTS = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]


def _find_project_root(start_file: str) -> Path:
    """Best-effort project root detection for local import resolution."""
    current = Path(start_file).resolve().parent
    markers = {".git", "pyproject.toml", ".ctxpilot", "go.mod", "package.json"}

    for candidate in [current, *current.parents]:
        for marker in markers:
            if (candidate / marker).exists():
                return candidate
    return current


def _walk_nodes(node):
    """Yield a node and all descendants depth-first."""
    yield node
    for child in node.children:
        yield from _walk_nodes(child)


def _within_project(path: Path, project_root: Path) -> bool:
    """Return True if path is inside project_root."""
    try:
        path.resolve().relative_to(project_root.resolve())
        return True
    except ValueError:
        return False


def _resolve_module_to_files(module: str, project_root: Path, base_dir: Path, level: int = 0) -> list[Path]:
    """Resolve a Python module path to concrete local files."""
    if not module and level <= 0:
        return []

    # Relative levels: from .foo import ... => level=1 means current package
    # directory, level=2 means parent package, etc.
    anchor = base_dir
    if level > 1:
        for _ in range(level - 1):
            anchor = anchor.parent

    if level > 0:
        target_base = anchor / module.replace(".", "/") if module else anchor
    else:
        target_base = project_root / module.replace(".", "/")

    candidates = [
        target_base.with_suffix(".py"),
        target_base / "__init__.py",
    ]

    resolved: list[Path] = []
    for candidate in candidates:
        c = candidate.resolve()
        if c.exists() and c.is_file() and _within_project(c, project_root):
            resolved.append(c)
    return resolved


def _resolve_js_ts_import(spec: str, file_dir: Path, project_root: Path) -> list[Path]:
    """Resolve a JS/TS import specifier to local files."""
    if not spec:
        return []

    # Only local imports are included for relationship expansion.
    if spec.startswith("."):
        base = (file_dir / spec).resolve()
    elif spec.startswith("/"):
        base = (project_root / spec.lstrip("/")).resolve()
    else:
        return []

    candidates: list[Path] = []

    if base.exists() and base.is_file():
        candidates.append(base)

    if base.suffix:
        if base.exists() and base.is_file():
            candidates.append(base)
    else:
        for ext in _JS_TS_IMPORT_EXTS:
            candidates.append(Path(str(base) + ext))
        for ext in _JS_TS_IMPORT_EXTS:
            candidates.append(base / f"index{ext}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        c = candidate.resolve()
        if c.exists() and c.is_file() and _within_project(c, project_root) and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _resolve_go_import(spec: str, file_dir: Path, project_root: Path) -> list[Path]:
    """Resolve Go import path to local files where possible."""
    if not spec:
        return []

    candidates: list[Path] = []

    if spec.startswith("."):
        base = (file_dir / spec).resolve()
        candidates.append(base)
    elif spec.startswith("/"):
        base = (project_root / spec.lstrip("/")).resolve()
        candidates.append(base)
    else:
        # Attempt module-aware mapping: module/path -> project_root/path
        go_mod = project_root / "go.mod"
        if go_mod.exists():
            try:
                mod_line = go_mod.read_text(encoding="utf-8", errors="replace").splitlines()
                module_name = ""
                for line in mod_line:
                    line = line.strip()
                    if line.startswith("module "):
                        module_name = line.split("module ", 1)[1].strip()
                        break
                if module_name and spec.startswith(f"{module_name}/"):
                    rel = spec[len(module_name) + 1:]
                    candidates.append((project_root / rel).resolve())
            except OSError:
                pass

        # Fallback best-effort local match
        candidates.append((project_root / spec).resolve())

    expanded: list[Path] = []
    for base in candidates:
        if base.exists() and base.is_file():
            expanded.append(base)
        if base.exists() and base.is_dir():
            expanded.extend(sorted(base.glob("*.go")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in expanded:
        c = candidate.resolve()
        if c.exists() and c.is_file() and _within_project(c, project_root) and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _extract_python_import_targets(node_text: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Extract Python import specs from one import node text.

    Returns:
        (modules_to_resolve, modules_with_members)
        - modules_to_resolve: list[(module, level)]
        - modules_with_members: list[(module.member, level)] candidates for from-imports
    """
    modules: list[tuple[str, int]] = []
    members: list[tuple[str, int]] = []
    text = node_text.strip()

    if text.startswith("import "):
        raw = text[len("import "):]
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            module = part.split(" as ", 1)[0].strip()
            if module:
                modules.append((module, 0))
        return modules, members

    if text.startswith("from "):
        m = re.match(r"from\s+([\.]*)([\w\.]+)?\s+import\s+(.+)$", text)
        if not m:
            return modules, members

        dots = m.group(1) or ""
        module = (m.group(2) or "").strip()
        imported = (m.group(3) or "").strip()
        level = len(dots)

        modules.append((module, level))

        imported_names = [i.strip() for i in imported.split(",") if i.strip()]
        for name in imported_names:
            clean = name.split(" as ", 1)[0].strip()
            if clean == "*" or not clean:
                continue
            base = f"{module}.{clean}" if module else clean
            members.append((base, level))

    return modules, members


def _get_language(lang_name: str):
    """Return a tree-sitter Language object, cached after first load."""
    if lang_name in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[lang_name]

    try:
        if lang_name == "python":
            import tree_sitter_python as tsp
            language = tsp.language()
        elif lang_name == "javascript":
            import tree_sitter_javascript as tsjs
            language = tsjs.language()
        elif lang_name == "typescript":
            import tree_sitter_typescript as tsts
            language = tsts.language_typescript()
        elif lang_name == "tsx":
            import tree_sitter_typescript as tsts
            language = tsts.language_tsx()
        elif lang_name == "go":
            import tree_sitter_go as tsgo
            language = tsgo.language()
        else:
            return None
    except ImportError:
        print(
            f"[ctxpilot] Warning: tree-sitter language package for "
            f"'{lang_name}' not installed, skipping.",
            file=sys.stderr,
        )
        return None

    _LANGUAGE_CACHE[lang_name] = language
    return language


@dataclass(frozen=True, slots=True)
class Symbol:
    """A single extracted symbol from source code."""

    file_path: str
    language: str
    name: str
    kind: str  # "function" | "class" | "method" | "arrow_function"
    signature: str
    docstring: str | None
    body_preview: str  # Signature + first 5 non-trivial body lines
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    body_hash: str  # xxhash of the raw body bytes


# ---------------------------------------------------------------------------
# Node type → symbol kind mapping per language
# ---------------------------------------------------------------------------

# Python: function_definition, class_definition
# JavaScript: function_declaration, class_declaration, lexical_declaration (arrow)
# TypeScript/TSX: same as JS + type annotations
# Go: function_declaration, method_declaration, type_declaration

_PYTHON_SYMBOL_NODES = {"function_definition", "class_definition"}
_JS_TS_SYMBOL_NODES = {"function_declaration", "class_declaration"}
_GO_SYMBOL_NODES = {"function_declaration", "method_declaration"}


def _detect_language(file_path: str) -> str | None:
    """Detect language from file extension. Returns None for unsupported."""
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext)


def _node_text(node, source_bytes: bytes) -> str:
    """Extract the text content of a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_body_preview(lines: list[str], max_body_lines: int = 5) -> str:
    """Build body_preview: signature line + first N non-trivial body lines.

    Non-trivial = not blank, not just 'pass', not just type hints.
    """
    if not lines:
        return ""

    # First line is always the signature
    preview_lines = [lines[0]]
    body_count = 0

    for line in lines[1:]:
        stripped = line.strip()
        # Skip trivial lines
        if not stripped:
            continue
        if stripped == "pass":
            continue
        # Skip pure type hint annotations (e.g. "x: int" with no assignment)
        if ":" in stripped and "=" not in stripped and not stripped.startswith(("def ", "class ", "if ", "for ", "while ", "return ", "raise ", "try", "with ", "elif ", "else:", "except", "finally")):
            # Could be a type hint like `x: int` — but also could be dict/slice
            # Only skip if it looks like a simple annotation
            parts = stripped.split(":", 1)
            if len(parts) == 2 and parts[0].strip().isidentifier() and not parts[1].strip().startswith(("{", "[", "(")):
                continue

        preview_lines.append(line)
        body_count += 1
        if body_count >= max_body_lines:
            break

    return "\n".join(preview_lines)


def _compute_body_hash(source_bytes: bytes, start_byte: int, end_byte: int) -> str:
    """Compute xxhash of the raw body bytes for staleness detection."""
    return xxhash.xxh64(source_bytes[start_byte:end_byte]).hexdigest()


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------

def _extract_python_docstring(node, source_bytes: bytes) -> str | None:
    """Extract docstring from a Python function/class node."""
    body = node.child_by_field_name("body")
    if body is None:
        return None

    for child in body.children:
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr and expr.type == "string":
                text = _node_text(expr, source_bytes)
                # Strip triple quotes
                for q in ('"""', "'''"):
                    if text.startswith(q) and text.endswith(q):
                        return text[3:-3].strip()
                return text.strip("\"'").strip()
        elif child.type not in ("comment",):
            break
    return None


def _extract_python_symbols(root, source_bytes: bytes, file_path: str) -> list[Symbol]:
    """Extract symbols from a Python AST."""
    symbols = []
    _walk_python(root.root_node, source_bytes, file_path, symbols, parent_class=None)
    return symbols


def _walk_python(node, source_bytes: bytes, file_path: str, symbols: list[Symbol], parent_class: str | None):
    """Recursively walk Python AST to extract functions, classes, and methods."""
    for child in node.children:
        if child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _node_text(name_node, source_bytes)
            full_name = f"{parent_class}.{name}" if parent_class else name
            kind = "method" if parent_class else "function"

            # Signature: the `def ...(...) -> ...:` line
            sig_line = source_bytes[child.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()

            docstring = _extract_python_docstring(child, source_bytes)

            body_text = _node_text(child, source_bytes)
            body_lines = body_text.split("\n")
            preview = _get_body_preview(body_lines)

            body_hash = _compute_body_hash(source_bytes, child.start_byte, child.end_byte)

            symbols.append(Symbol(
                file_path=file_path,
                language="python",
                name=full_name,
                kind=kind,
                signature=sig_line,
                docstring=docstring,
                body_preview=preview,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                body_hash=body_hash,
            ))

        elif child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            class_name = _node_text(name_node, source_bytes)
            full_class_name = f"{parent_class}.{class_name}" if parent_class else class_name

            sig_line = source_bytes[child.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()
            docstring = _extract_python_docstring(child, source_bytes)
            body_text = _node_text(child, source_bytes)
            body_lines = body_text.split("\n")
            preview = _get_body_preview(body_lines)
            body_hash = _compute_body_hash(source_bytes, child.start_byte, child.end_byte)

            symbols.append(Symbol(
                file_path=file_path,
                language="python",
                name=full_class_name,
                kind="class",
                signature=sig_line,
                docstring=docstring,
                body_preview=preview,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                body_hash=body_hash,
            ))

            # Recurse into class body for methods
            body = child.child_by_field_name("body")
            if body:
                _walk_python(body, source_bytes, file_path, symbols, parent_class=full_class_name)

        elif child.type == "decorated_definition":
            # Decorated functions/classes — recurse to find the actual definition
            _walk_python(child, source_bytes, file_path, symbols, parent_class=parent_class)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript extraction
# ---------------------------------------------------------------------------

def _extract_js_ts_docstring(node, source_bytes: bytes) -> str | None:
    """Extract JSDoc comment preceding a JS/TS node."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = _node_text(prev, source_bytes).strip()
        if text.startswith("/**"):
            # Strip /** ... */
            text = text[3:]
            if text.endswith("*/"):
                text = text[:-2]
            # Clean up leading * on each line
            lines = []
            for line in text.split("\n"):
                line = line.strip().lstrip("* ").strip()
                if line and not line.startswith("@"):
                    lines.append(line)
            return " ".join(lines) if lines else None
    return None


def _extract_js_ts_symbols(root, source_bytes: bytes, file_path: str, lang: str) -> list[Symbol]:
    """Extract symbols from JavaScript/TypeScript AST."""
    symbols = []
    _walk_js_ts(root.root_node, source_bytes, file_path, lang, symbols, parent_class=None)
    return symbols


def _walk_js_ts(node, source_bytes: bytes, file_path: str, lang: str, symbols: list[Symbol], parent_class: str | None):
    """Recursively walk JS/TS AST to extract functions, classes, arrow functions."""
    for child in node.children:
        if child.type in ("function_declaration", "function"):
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _node_text(name_node, source_bytes)
            full_name = f"{parent_class}.{name}" if parent_class else name

            sig_line = source_bytes[child.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()
            docstring = _extract_js_ts_docstring(child, source_bytes)
            body_text = _node_text(child, source_bytes)
            body_lines = body_text.split("\n")
            preview = _get_body_preview(body_lines)
            body_hash = _compute_body_hash(source_bytes, child.start_byte, child.end_byte)

            symbols.append(Symbol(
                file_path=file_path,
                language=lang,
                name=full_name,
                kind="method" if parent_class else "function",
                signature=sig_line,
                docstring=docstring,
                body_preview=preview,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                body_hash=body_hash,
            ))

        elif child.type == "class_declaration":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            class_name = _node_text(name_node, source_bytes)
            full_class_name = f"{parent_class}.{class_name}" if parent_class else class_name

            sig_line = source_bytes[child.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()
            docstring = _extract_js_ts_docstring(child, source_bytes)
            body_text = _node_text(child, source_bytes)
            body_lines = body_text.split("\n")
            preview = _get_body_preview(body_lines)
            body_hash = _compute_body_hash(source_bytes, child.start_byte, child.end_byte)

            symbols.append(Symbol(
                file_path=file_path,
                language=lang,
                name=full_class_name,
                kind="class",
                signature=sig_line,
                docstring=docstring,
                body_preview=preview,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                body_hash=body_hash,
            ))

            # Recurse into class body for methods
            body = child.child_by_field_name("body")
            if body:
                _walk_js_ts(body, source_bytes, file_path, lang, symbols, parent_class=full_class_name)

        elif child.type in ("lexical_declaration", "variable_declaration"):
            # Check for `const foo = (...) => { ... }` arrow functions
            _extract_arrow_functions(child, source_bytes, file_path, lang, symbols, parent_class)

        elif child.type == "export_statement":
            # export function, export class, export const
            _walk_js_ts(child, source_bytes, file_path, lang, symbols, parent_class=parent_class)

        elif child.type == "method_definition":
            # Class methods — inside class body
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _node_text(name_node, source_bytes)
            full_name = f"{parent_class}.{name}" if parent_class else name

            sig_line = source_bytes[child.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()
            docstring = _extract_js_ts_docstring(child, source_bytes)
            body_text = _node_text(child, source_bytes)
            body_lines = body_text.split("\n")
            preview = _get_body_preview(body_lines)
            body_hash = _compute_body_hash(source_bytes, child.start_byte, child.end_byte)

            symbols.append(Symbol(
                file_path=file_path,
                language=lang,
                name=full_name,
                kind="method",
                signature=sig_line,
                docstring=docstring,
                body_preview=preview,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                body_hash=body_hash,
            ))

        elif child.type == "class_body":
            # Recurse into class bodies
            _walk_js_ts(child, source_bytes, file_path, lang, symbols, parent_class=parent_class)


def _extract_arrow_functions(node, source_bytes: bytes, file_path: str, lang: str, symbols: list[Symbol], parent_class: str | None):
    """Extract arrow functions from const/let variable declarations."""
    for child in node.children:
        if child.type != "variable_declarator":
            continue

        name_node = child.child_by_field_name("name")
        value_node = child.child_by_field_name("value")

        if name_node is None or value_node is None:
            continue

        # Check if the value is an arrow function
        actual_value = value_node
        # Handle TypeScript type assertions: `const x = expr as Type`
        if actual_value.type == "as_expression":
            actual_value = actual_value.children[0] if actual_value.children else actual_value

        if actual_value.type != "arrow_function":
            continue

        name = _node_text(name_node, source_bytes)
        full_name = f"{parent_class}.{name}" if parent_class else name

        # Build signature from the whole declaration line
        sig_line = source_bytes[node.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()

        docstring = _extract_js_ts_docstring(node, source_bytes)
        body_text = _node_text(node, source_bytes)
        body_lines = body_text.split("\n")
        preview = _get_body_preview(body_lines)
        body_hash = _compute_body_hash(source_bytes, node.start_byte, node.end_byte)

        symbols.append(Symbol(
            file_path=file_path,
            language=lang,
            name=full_name,
            kind="arrow_function",
            signature=sig_line,
            docstring=docstring,
            body_preview=preview,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            body_hash=body_hash,
        ))


# ---------------------------------------------------------------------------
# Go extraction
# ---------------------------------------------------------------------------

def _extract_go_docstring(node, source_bytes: bytes) -> str | None:
    """Extract Go doc comment preceding a function/method."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = _node_text(prev, source_bytes).strip()
        if text.startswith("//"):
            return text[2:].strip()
    return None


def _extract_go_symbols(root, source_bytes: bytes, file_path: str) -> list[Symbol]:
    """Extract symbols from Go AST."""
    symbols = []

    for child in root.root_node.children:
        if child.type == "function_declaration":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _node_text(name_node, source_bytes)
            sig_line = source_bytes[child.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()
            docstring = _extract_go_docstring(child, source_bytes)
            body_text = _node_text(child, source_bytes)
            body_lines = body_text.split("\n")
            preview = _get_body_preview(body_lines)
            body_hash = _compute_body_hash(source_bytes, child.start_byte, child.end_byte)

            symbols.append(Symbol(
                file_path=file_path,
                language="go",
                name=name,
                kind="function",
                signature=sig_line,
                docstring=docstring,
                body_preview=preview,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                body_hash=body_hash,
            ))

        elif child.type == "method_declaration":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue

            # Get receiver type for full name
            receiver = child.child_by_field_name("receiver")
            receiver_type = ""
            if receiver:
                # Walk receiver to find type identifier
                for rchild in receiver.children:
                    if rchild.type == "parameter_list":
                        for param in rchild.children:
                            if param.type == "parameter_declaration":
                                type_node = param.child_by_field_name("type")
                                if type_node:
                                    receiver_type = _node_text(type_node, source_bytes).strip("*")

            name = _node_text(name_node, source_bytes)
            full_name = f"{receiver_type}.{name}" if receiver_type else name

            sig_line = source_bytes[child.start_byte:].decode("utf-8", errors="replace").split("\n")[0].strip()
            docstring = _extract_go_docstring(child, source_bytes)
            body_text = _node_text(child, source_bytes)
            body_lines = body_text.split("\n")
            preview = _get_body_preview(body_lines)
            body_hash = _compute_body_hash(source_bytes, child.start_byte, child.end_byte)

            symbols.append(Symbol(
                file_path=file_path,
                language="go",
                name=full_name,
                kind="method",
                signature=sig_line,
                docstring=docstring,
                body_preview=preview,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                body_hash=body_hash,
            ))

    return symbols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_symbols(file_path: str) -> list[Symbol]:
    """Extract all symbols from a source file.

    Supports Python, JavaScript, TypeScript, and Go.
    Returns an empty list for unsupported languages or parse errors.
    """
    import tree_sitter as ts

    lang_name = _detect_language(file_path)
    if lang_name is None:
        return []

    path = Path(file_path)
    if not path.exists():
        print(f"[ctxpilot] Warning: file not found: {file_path}", file=sys.stderr)
        return []

    try:
        source_bytes = path.read_bytes()
    except OSError as e:
        print(f"[ctxpilot] Warning: could not read {file_path}: {e}", file=sys.stderr)
        return []

    language = _get_language(lang_name)
    if language is None:
        return []

    try:
        parser = ts.Parser(ts.Language(language))
        tree = parser.parse(source_bytes)
    except Exception as e:
        print(
            f"[ctxpilot] Warning: tree-sitter parse failed for {file_path}: {e}",
            file=sys.stderr,
        )
        return []

    # Dispatch to language-specific extractor
    normalized_lang = lang_name if lang_name != "tsx" else "typescript"

    if lang_name == "python":
        return _extract_python_symbols(tree, source_bytes, file_path)
    elif lang_name in ("javascript", "typescript", "tsx"):
        return _extract_js_ts_symbols(tree, source_bytes, file_path, normalized_lang)
    elif lang_name == "go":
        return _extract_go_symbols(tree, source_bytes, file_path)
    else:
        return []


def extract_imports(file_path: str) -> list[str]:
    """Extract one-hop local import edges for a source file.

    Returns absolute paths for imported files that exist under the project root.
    Non-local/stdlib/third-party imports are skipped. Parse failures return []
    and never raise.
    """
    import tree_sitter as ts

    lang_name = _detect_language(file_path)
    if lang_name is None:
        return []

    path = Path(file_path)
    if not path.exists():
        return []

    try:
        source_bytes = path.read_bytes()
    except OSError:
        return []

    language = _get_language(lang_name)
    if language is None:
        return []

    try:
        parser = ts.Parser(ts.Language(language))
        tree = parser.parse(source_bytes)
    except Exception:
        return []

    project_root = _find_project_root(file_path)
    file_dir = path.resolve().parent

    imports: list[Path] = []
    seen: set[Path] = set()

    try:
        for node in _walk_nodes(tree.root_node):
            ntype = node.type
            text = _node_text(node, source_bytes)

            if lang_name == "python" and ntype in {"import_statement", "import_from_statement"}:
                modules, members = _extract_python_import_targets(text)

                for module, level in modules:
                    for resolved in _resolve_module_to_files(module, project_root, file_dir, level):
                        if resolved not in seen:
                            imports.append(resolved)
                            seen.add(resolved)

                # from x import y may refer to module x.y or a symbol in x.
                for member, level in members:
                    for resolved in _resolve_module_to_files(member, project_root, file_dir, level):
                        if resolved not in seen:
                            imports.append(resolved)
                            seen.add(resolved)

            elif lang_name in {"javascript", "typescript", "tsx"}:
                if ntype == "import_declaration":
                    match = re.search(r"from\s+[\"']([^\"']+)[\"']", text)
                    if not match:
                        match = re.search(r"import\s+[\"']([^\"']+)[\"']", text)
                    if match:
                        spec = match.group(1)
                        for resolved in _resolve_js_ts_import(spec, file_dir, project_root):
                            if resolved not in seen:
                                imports.append(resolved)
                                seen.add(resolved)

                elif ntype == "call_expression" and "require(" in text:
                    match = re.search(r"require\(\s*[\"']([^\"']+)[\"']\s*\)", text)
                    if match:
                        spec = match.group(1)
                        for resolved in _resolve_js_ts_import(spec, file_dir, project_root):
                            if resolved not in seen:
                                imports.append(resolved)
                                seen.add(resolved)

            elif lang_name == "go" and ntype == "import_declaration":
                specs = re.findall(r'\"([^\"]+)\"', text)
                for spec in specs:
                    for resolved in _resolve_go_import(spec, file_dir, project_root):
                        if resolved not in seen:
                            imports.append(resolved)
                            seen.add(resolved)
    except Exception:
        return []

    return [str(p) for p in imports]


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m contextpilot.extractor <file_path> [file_path ...]", file=sys.stderr)
        sys.exit(1)

    all_symbols = []
    for fpath in sys.argv[1:]:
        symbols = extract_symbols(fpath)
        all_symbols.extend(symbols)

    # Print as JSON to stdout
    output = [asdict(s) for s in all_symbols]
    print(json.dumps(output, indent=2))
