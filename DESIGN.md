# UI/UX Design System: Private, Ad-Free Neural Web Search Engine

**Project**: NeuralSearch  
**Path**: `D:\neural-search-engine\DESIGN.md`  
**Aesthetic Direction**: Light Editorial Minimalism (Warm Alabaster & Burnt Terracotta)  

---

## 1. Web Search Design Philosophy

The interface follows the principles of **privacy, clarity, and zero commercial noise**:
1. **Ad-Free Serenity**: Zero banner ads, zero sponsored promotions, zero flashing widgets. Just pure, organic internet knowledge.
2. **Warm Alabaster Canvas (`#FCF2E5`)**: Avoids harsh blinding pure white or clinical grays, creating a calm reading environment.
3. **Transparent Privacy Indicators**: Highlights to the user that their query was anonymized and displays how many trackers were stripped.

---

## 2. Color System & Semantic Tokens

```css
:root {
  /* Canvas & Background Surfaces */
  --bg-canvas: #FCF2E5;         /* Warm Alabaster Cream (Page Canvas) */
  --bg-surface: #FFFFFF;        /* Pure White (Search Bar, Result Cards) */
  --bg-surface-subtle: #FAF5EE; /* Soft Tinted Cream (Badges, Buttons) */

  /* Borders & Dividers */
  --border-subtle: #E5DBCA;     /* Standard Container Border */
  --border-strong: #D5C9B0;     /* Active / Focus Border */

  /* Typography */
  --text-primary: #3A3131;      /* Deep Espresso (Headings, Result Titles) */
  --text-body: #524646;         /* Charcoal Espresso (Snippet Text) */
  --text-muted: #7A7465;        /* Sage Khaki (Domains, Paths, Metadata) */
  --text-link: #2563EB;         /* Subtle Blue / Terracotta for Web URLs */

  /* Brand Accent: Burnt Terracotta */
  --accent-primary: #EC5B38;    /* Action Buttons, Focus Rings, Active Tabs */
  --accent-subtle: rgba(236, 91, 56, 0.12); /* Keyword Highlight Background */
  --privacy-green: #15803D;     /* Privacy Shield Pill */
}
```

---

## 3. Web Search Component Specifications

### 3.1 Hero Search Bar & Category Tabs
- Category navigation:
  - **All Web**: General search across the entire internet.
  - **Tech & Code**: Prioritizes GitHub, StackOverflow, MDN, documentation, and papers.
  - **News & Articles**: Focuses on current publications and editorial pieces.
- Prominent search input with auto-focus, real-time debouncing, and search keyboard shortcut (`/`).

### 3.2 Privacy & Security Shield Bar
- Displays privacy audit status for the active query:
  - `🛡️ Private Session (No cookies, no IP logging)`
  - `🚫 4 Sponsored Ads Blocked`
  - `⚡ 12 Tracking Parameters Stripped`

### 3.3 Web Result Card
- **Domain Row**: Site favicon + clean domain breadcrumb (`en.wikipedia.org > wiki > Machine_learning`).
- **Title**: Clean clickable link pointing directly to the external website (no tracking redirection hops).
- **Snippet**: Extracted web summary with `<mark class="highlight">` spans for matched search terms.
- **Badges**:
  - `Organic Web Result`
  - `Direct Link (No tracking hop)`
  - `Inspect Rerank Score` button to open explainability breakdown.

### 3.4 Direct Answer & Wikipedia Instant Box
- For fact-based queries (e.g. `"what is quantum computing"` or `"define neural network"`), an elevated knowledge summary card appears at the top of the search results with a direct link to the canonical source.
