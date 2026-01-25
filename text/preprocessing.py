from __future__ import annotations
from typing import List
import re

# Students sometimes write name/title/body on separate lines or Shift+Enter lines
# without terminal punctuation. If we later join everything into one block, the sentence
# splitter may treat those lines as part of the first sentence.
# These helpers enforce sentence boundaries at LINE ENDS before flattening.

_END_PUNCT = {".", "!", "?"}
_TRAILING_CLOSERS = {'"', "`", "'", ")", "]", "}", "“", "‘", "«", "˙", "ˆ", "¨", "´"}

def _ends_with_terminal_punct(s: str) -> bool:
    t = (s or "").rstrip()
    if not t:
        return False
    
    # Strip trailing closers (quotes/brackets) to check the true last punct
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1].rstrip()
        print(t)

    return bool(t) and (t[-1] in _END_PUNCT)

def add_periods_to_line_ends(paragraphs: List[str]) -> List[str]:
    """
    Ensure each LINE ends with terminal punctuation, so sentence splitting
    respects name/title/body boundaries when paragraphs are flattened.
    
    Applies to:
        - paragraph boundaries (because we later join paragraphs)
        - explicit line breaks inside a paragraph ('\\n' from Shift + Enter)
    
    Rules:
        - Empty lines preserved
        - If a non-empty line does not end with . ! ? (allowing trailing closers),
            append a ".".
    """
    out: List[str] = []

    for p in paragraphs:
        txt = (p or "").replace("\r\n", "\n")
        lines = txt.split("\n")
        
        fixed_lines: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                fixed_lines.append("") # Keep empty line
                continue
            if _ends_with_terminal_punct(s):
                fixed_lines.append(s)
            else:
                fixed_lines.append(s + ".")
        out.append("\n".join(fixed_lines))
    print(out)
    return out

def flatten_paragraphs_to_single(paragraphs: List[str]) -> str:
    """
    Flattens paragraphs into a single paragraph string.
    Call add_periods_to_line_ends(...) FIRST if you want line-end
    boundaries to be preserved as sentence boundaries after flattening.
    """   
    parts: List[str] = []

    for p in paragraphs:
        s = (p or "").replace("\r\n", "\n").replace("\n", " ").strip()
        if s:
            parts.append(s)

    merged = " ".join(parts).strip()
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged         

# Runs a quick test
# def main():
#     sens = ["Hello there.\r\nWhat is your name?\n\n\nMy name is Daniel", "Hi there ", "Hi there `", "Hello bum`", "Oopsy daisy", 
#             "What is that? ", "What is that ?...`", '"Who are you?"']

#     for p in sens:
#         print(p)
    
#     out = add_periods_to_line_ends(sens)
#     print(flatten_paragraphs_to_single(out))


# if __name__ == "__main__":
#     main()