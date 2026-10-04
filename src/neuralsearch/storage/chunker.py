"""
Hierarchical, structure-aware document chunker that respects markdown sections,
headings, code blocks, and preserves context hierarchies.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class Chunk:
    """A semantic chunk of a document."""
    chunk_index: int
    content: str
    headings: List[str] = field(default_factory=list)
    start_line: int = 1
    end_line: int = 1
    token_count: int = 0


class MarkdownChunker:
    """
    Splits Markdown and structured text into semantic chunks along heading boundaries
    and paragraph breaks while keeping code blocks intact.
    """

    HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def __init__(self, max_tokens: int = 300, min_tokens: int = 30, overlap_tokens: int = 40):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _approx_token_count(text: str) -> int:
        """Approximate word/token count using whitespace splitting."""
        return len(text.split())

    def chunk_document(self, text: str, document_title: Optional[str] = None) -> List[Chunk]:
        """
        Chunks text while preserving heading hierarchy context.
        """
        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        sections: List[dict] = []
        current_headings: List[str] = [document_title] if document_title else []
        current_lines: List[str] = []
        start_line = 1

        in_code_block = False

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Track fenced code blocks to prevent splitting them
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            # Heading detected outside code block
            heading_match = self.HEADING_REGEX.match(stripped) if not in_code_block else None

            if heading_match and current_lines:
                # Flush previous section
                section_text = "".join(current_lines).strip()
                if section_text:
                    sections.append({
                        "text": section_text,
                        "headings": list(current_headings),
                        "start_line": start_line,
                        "end_line": line_num - 1,
                    })
                current_lines = [line]
                start_line = line_num

                # Update heading stack
                heading_level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # Trim deeper headings and append current
                target_len = heading_level - 1 if document_title else heading_level
                current_headings = current_headings[:max(0, target_len)]
                current_headings.append(heading_text)
            else:
                current_lines.append(line)

        # Flush final section
        if current_lines:
            section_text = "".join(current_lines).strip()
            if section_text:
                sections.append({
                    "text": section_text,
                    "headings": list(current_headings),
                    "start_line": start_line,
                    "end_line": len(lines),
                })

        # Break overly large sections into chunks with overlap
        chunks: List[Chunk] = []
        chunk_idx = 0

        for sec in sections:
            sec_text = sec["text"]
            sec_headings = sec["headings"]
            heading_prefix = " > ".join(sec_headings) + "\n\n" if sec_headings else ""

            words = sec_text.split()
            if len(words) <= self.max_tokens:
                content = (heading_prefix + sec_text).strip()
                chunks.append(
                    Chunk(
                        chunk_index=chunk_idx,
                        content=content,
                        headings=sec_headings,
                        start_line=sec["start_line"],
                        end_line=sec["end_line"],
                        token_count=len(words),
                    )
                )
                chunk_idx += 1
            else:
                # Sliding window chunking
                step = self.max_tokens - self.overlap_tokens
                for i in range(0, len(words), step):
                    window = words[i : i + self.max_tokens]
                    if len(window) < self.min_tokens and i > 0:
                        break  # Avoid trailing tiny fragments

                    chunk_body = " ".join(window)
                    content = (heading_prefix + chunk_body).strip()
                    chunks.append(
                        Chunk(
                            chunk_index=chunk_idx,
                            content=content,
                            headings=sec_headings,
                            start_line=sec["start_line"],
                            end_line=sec["end_line"],
                            token_count=len(window),
                        )
                    )
                    chunk_idx += 1

        return chunks
