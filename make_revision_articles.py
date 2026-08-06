from __future__ import annotations

import difflib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
OLD = PAPER / "main_before_actions_20260805.tex"
if not OLD.exists():
    OLD = PAPER / "validation_logs" / "main_before_actions_20260805.tex"
NEW = PAPER / "main.tex"
BLUE_OUT = PAPER / "article_v1_0_blue_changes.tex"
RED_BLUE_OUT = PAPER / "article_v1_0_red_blue_changes.tex"


REVISION_MACROS = r"""
\definecolor{revisionblue}{RGB}{0,0,180}
\definecolor{revisionred}{RGB}{180,0,0}
\newcommand{\RevisionAddBegin}{\begingroup\color{revisionblue}}
\newcommand{\RevisionAddEnd}{\endgroup}
\newenvironment{RevisionDeletion}{%
  \par\begingroup\color{revisionred}\footnotesize\ttfamily
  \setlength{\parindent}{0pt}\setlength{\parskip}{1pt}\obeyspaces
}{\par\endgroup}
"""


BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")
DANGEROUS_ENVS = {
    "abstract",
    "algorithm",
    "algorithm*",
    "algorithmic",
    "array",
    "equation",
    "figure",
    "figure*",
    "split",
    "tabular",
    "tabular*",
    "table",
    "table*",
    "tikzpicture",
}
NO_COLOR_ENVS = {"abstract"}
INLINE_COLOR_ENVS = {"keywords"}
STRUCTURAL_PREFIXES = (
    r"\begin{",
    r"\end{",
    r"\label{",
    r"\bibliography",
    r"\bibliographystyle",
    r"\Require",
    r"\Ensure",
    r"\State",
    r"\If",
    r"\EndIf",
    r"\For",
    r"\EndFor",
)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def inject_macros(lines: list[str]) -> list[str]:
    out: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and line.strip() == r"\begin{document}":
            out.append(REVISION_MACROS + "\n")
            inserted = True
        out.append(line)
    if not inserted:
        raise RuntimeError(r"Could not find \begin{document}")
    return out


def split_at_begin_document(lines: list[str]) -> tuple[list[str], list[str]]:
    for i, line in enumerate(lines):
        if line.strip() == r"\begin{document}":
            return lines[: i + 1], lines[i + 1 :]
    raise RuntimeError(r"Could not find \begin{document}")


def update_env_stack(stack: list[str], line: str) -> None:
    for env in BEGIN_RE.findall(line):
        stack.append(env)
    for env in END_RE.findall(line):
        if env in stack:
            stack.pop(len(stack) - 1 - stack[::-1].index(env))


def strip_latex_comment(line: str) -> str:
    escaped = False
    for idx, char in enumerate(line):
        if char == "\\":
            escaped = not escaped
            continue
        if char == "%" and not escaped:
            return line[:idx]
        escaped = False
    return line


def brace_delta(line: str) -> int:
    text = strip_latex_comment(line)
    delta = 0
    escaped = False
    for char in text:
        if char == "\\":
            escaped = not escaped
            continue
        if char == "{" and not escaped:
            delta += 1
        elif char == "}" and not escaped:
            delta -= 1
        escaped = False
    return delta


def chunk_delta(lines: list[str]) -> int:
    return sum(brace_delta(line) for line in lines)


def in_dangerous_env(stack: list[str]) -> bool:
    return any(env in DANGEROUS_ENVS for env in stack)


def escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text.rstrip("\n"))


def deletion_block(lines: list[str], header: str | None = None) -> list[str]:
    out = [r"\begin{RevisionDeletion}" + "\n"]
    if header:
        out.append(escape_latex_text(header) + r"\par" + "\n")
    for line in lines:
        if line.strip():
            out.append(escape_latex_text(line) + r"\par" + "\n")
        else:
            out.append(r"\par" + "\n")
    out.append(r"\end{RevisionDeletion}" + "\n")
    return out


def can_inline_color(stack: list[str]) -> bool:
    return any(env in INLINE_COLOR_ENVS for env in stack)


def must_skip_color(stack: list[str] | None) -> bool:
    return stack is not None and any(env in NO_COLOR_ENVS for env in stack)


def blue_block(
    lines: list[str],
    stack: list[str] | None = None,
    brace_depth: int = 0,
) -> list[str]:
    if not lines:
        return []
    if must_skip_color(stack):
        return lines
    if any(line.lstrip().startswith(STRUCTURAL_PREFIXES) for line in lines):
        return lines
    if stack is not None and can_inline_color(stack):
        out: list[str] = []
        for line in lines:
            stripped = line.rstrip("\n")
            if stripped:
                out.append(r"\textcolor{revisionblue}{" + stripped + "}\n")
            else:
                out.append(line)
        return out
    if brace_depth != 0 or chunk_delta(lines) != 0:
        return lines
    return [r"\RevisionAddBegin" + "\n", *lines, r"\RevisionAddEnd" + "\n"]


def append_deletions_before_bibliography(lines: list[str], deletions: list[list[str]]) -> list[str]:
    if not deletions:
        return lines
    appendix: list[str] = [
        "\n",
        r"\clearpage" + "\n",
        r"\section*{Deleted Material Shown in Red}" + "\n",
        "The following red blocks collect deleted manuscript material whose "
        "inline placement would otherwise break LaTeX structural environments.\n",
        "\n",
    ]
    for idx, block in enumerate(deletions, start=1):
        appendix.extend(deletion_block(block, f"Deleted block {idx}"))
        appendix.append("\n")

    for i, line in enumerate(lines):
        if line.startswith(r"\bibliographystyle"):
            return lines[:i] + appendix + lines[i:]
    for i, line in enumerate(lines):
        if line.strip() == r"\end{document}":
            return lines[:i] + appendix + lines[i:]
    return lines + appendix


def make_blue_only(old_body: list[str], new_body: list[str], preamble: list[str]) -> list[str]:
    out = inject_macros(preamble)
    matcher = difflib.SequenceMatcher(a=old_body, b=new_body, autojunk=False)
    env_stack: list[str] = []
    brace_depth = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(new_body[j1:j2])
            for line in new_body[j1:j2]:
                update_env_stack(env_stack, line)
                brace_depth += brace_delta(line)
        elif tag in {"insert", "replace"}:
            out.extend(blue_block(new_body[j1:j2], env_stack, brace_depth))
            for line in new_body[j1:j2]:
                update_env_stack(env_stack, line)
                brace_depth += brace_delta(line)
        elif tag == "delete":
            continue
    return out


def make_red_blue(old_body: list[str], new_body: list[str], preamble: list[str]) -> list[str]:
    out = inject_macros(preamble)
    matcher = difflib.SequenceMatcher(a=old_body, b=new_body, autojunk=False)
    env_stack: list[str] = []
    brace_depth = 0
    after_maketitle = False
    deferred_deletions: list[list[str]] = []

    def note_new_lines(lines: list[str]) -> None:
        nonlocal after_maketitle, brace_depth
        for line in lines:
            if r"\maketitle" in line:
                after_maketitle = True
            update_env_stack(env_stack, line)
            brace_depth += brace_delta(line)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_chunk = old_body[i1:i2]
        new_chunk = new_body[j1:j2]
        if tag == "equal":
            out.extend(new_chunk)
            note_new_lines(new_chunk)
        elif tag == "insert":
            block = blue_block(new_chunk, env_stack, brace_depth)
            out.extend(block)
            note_new_lines(new_chunk)
        elif tag == "delete":
            deferred_deletions.append(old_chunk)
        elif tag == "replace":
            deferred_deletions.append(old_chunk)
            block = blue_block(new_chunk, env_stack, brace_depth)
            out.extend(block)
            note_new_lines(new_chunk)

    return append_deletions_before_bibliography(out, deferred_deletions)


def main() -> None:
    old_lines = read_lines(OLD)
    new_lines = read_lines(NEW)
    new_preamble, new_body = split_at_begin_document(new_lines)
    _, old_body = split_at_begin_document(old_lines)

    BLUE_OUT.write_text("".join(make_blue_only(old_body, new_body, new_preamble)), encoding="utf-8")
    RED_BLUE_OUT.write_text("".join(make_red_blue(old_body, new_body, new_preamble)), encoding="utf-8")

    print(f"Wrote {BLUE_OUT}")
    print(f"Wrote {RED_BLUE_OUT}")


if __name__ == "__main__":
    main()
