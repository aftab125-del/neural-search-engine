# UI/UX Design System: Private Hybrid Neural Search Engine

**Project**: NeuralSearch  
**Path**: `D:\neural-search-engine\DESIGN.md`  
**Aesthetic Direction**: Light Editorial Minimalism (Warm Alabaster & Burnt Terracotta)  

---

## 1. Design Philosophy

The interface follows the principles of **quiet technical luxury**:
1. **Content First**: Every pixel serves readability and search velocity. No distracting gradients, floating shapes, or unnecessary decorative clutter.
2. **Warmth Over Cold Sterile Grays**: Rather than standard clinical gray/white dark-mode templates, the canvas uses a soft, tactile warm alabaster tone that prevents eye fatigue.
3. **Engine Transparency**: Recruiters and engineers can see *how* the search engine works in real-time through tasteful micro-typography and latency indicators.

---

## 2. Color System & Semantic Tokens

```css
:root {
  /* Canvas & Background Surfaces */
  --bg-canvas: #FCF2E5;         /* Warm Alabaster Cream (Main Page Background) */
  --bg-surface: #FFFFFF;        /* Crisp Floating White (Cards, Inputs, Modals) */
  --bg-surface-subtle: #FAF5EE; /* Soft Tinted Surface (Metadata Badges, Previews) */
  --bg-surface-muted: #F3EBDD;  /* Deeper Cream (Divider Bars, Active Elements) */

  /* Borders & Dividers */
  --border-subtle: #E5DBCA;     /* Standard Card & Container Border */
  --border-strong: #D5C9B0;     /* Focused & Hovered Container Border */

  /* Typography */
  --text-primary: #3A3131;      /* Deep Rich Espresso (Headings, Primary Text) */
  --text-body: #524646;         /* Charcoal Espresso (Body Copy, Snippet Text) */
  --text-muted: #7A7465;        /* Sage Khaki (Timestamps, Secondary Metadata) */
  --text-subtle: #A8A492;       /* Warm Slate Muted (Placeholder, Inactive Tags) */

  /* Brand Accent: Burnt Terracotta */
  --accent-primary: #EC5B38;    /* Vibrant Burnt Terracotta (Action Buttons, Focus) */
  --accent-hover: #D74B2A;      /* Deep Terracotta Hover State */
  --accent-subtle: rgba(236, 91, 56, 0.12); /* Match Highlight Background */
  --accent-border: rgba(236, 91, 56, 0.28); /* Highlight Border */

  /* Telemetry & Badges */
  --badge-fast: #16A34A;        /* Green for <20ms latency */
  --badge-warm: #EC5B38;        /* Terracotta for 20-50ms latency */
}
```

---

## 3. Typography Hierarchy

| Style | Font Family | Size | Weight | Line Height | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hero Title** | Plus Jakarta Sans / Inter | 24px (1.5rem) | 700 Bold | 1.2 | Header Title & Brand Mark |
| **Card Heading** | Plus Jakarta Sans / Inter | 16px (1.0rem) | 700 Bold | 1.3 | Document Title in Search Results |
| **Snippet Body** | Plus Jakarta Sans / Inter | 14px (0.875rem) | 400 Regular | 1.6 | Extracted Document Snippets |
| **Micro Labels** | JetBrains Mono | 11px (0.6875rem) | 500 Medium | 1.0 | BM25/Vector Scores, Latency Pills |
| **Metadata Tag** | JetBrains Mono | 12px (0.75rem) | 600 SemiBold | 1.0 | Rank #1, Document Path, Mode |

---

## 4. Component Specifications

### 4.1 Search Input Bar
- **Dimensions**: Full container width, minimum height `56px`, `16px` horizontal padding.
- **Surface**: Pure White (`#FFFFFF`) with a `2px` hairline border (`--border-subtle`).
- **Focus State**: Border transitions to `--accent-primary` (`#EC5B38`) with a subtle 4px blur ring (`rgba(236, 91, 56, 0.15)`).
- **Embedded Badges**: Right-aligned real-time latency pill (e.g. `24ms`) and keyboard shortcut indicator (`ESC`).

### 4.2 Search Result Card
- **Surface**: White (`#FFFFFF`) on Warm Alabaster Canvas (`#FCF2E5`).
- **Border**: `1px solid var(--border-subtle)`, rounded with `16px` radius.
- **Hover State**: Elevates by `-2px` with a subtle box shadow `0 8px 24px -4px rgba(82, 70, 70, 0.08)` and border shifting toward `--accent-primary` (`#EC5B38`).
- **Highlighting**: Matched keyword spans receive a soft warm highlight:
  ```css
  mark.highlight {
    background-color: rgba(236, 91, 56, 0.14);
    color: #D14421;
    font-weight: 600;
    padding: 1px 4px;
    border-radius: 4px;
  }
  ```

### 4.3 Stage Latency Waterfall Bar
- Positioned directly beneath the search input.
- Displays discrete execution times for:
  `BM25 Lexical (4ms)` • `ONNX Bi-Encoder (11ms)` • `Cross-Encoder (9ms)` $\rightarrow$ **Total: 24ms**
- Styled using monospace micro-typography to emphasize low-level systems engineering.

### 4.4 "Explainability" Slide-Over Drawer
- Clicking any result opens a right-side drawer showing:
  - Exact token match matrix for BM25 (query terms vs document term counts).
  - Cosine distance angle and vector similarity confidence.
  - Cross-encoder reranking delta (e.g., *Ranked #4 in BM25, promoted to #1 by Cross-Encoder*).
  - Full document content viewer with syntax highlighting for code files.

---

## 5. Keyboard Navigation Map

- `/` or `Ctrl + K`: Instantly focus search input from anywhere.
- `ArrowDown` / `j`: Move selection to next search result.
- `ArrowUp` / `k`: Move selection to previous search result.
- `Enter`: Open full preview or external document file.
- `e`: Toggle Explainability Drawer for the currently selected result.
- `Escape`: Clear query / Close drawer / Unfocus input.
