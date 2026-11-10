# UI/UX Design System: Private, Multi-Modal Neural Web Search Engine

**Project**: NeuralSearch  
**Path**: `D:\neural-search-engine\DESIGN.md`  
**Aesthetic Direction**: Editorial Ocean & Parchment (Merino, Rock Blue, Venice Blue)  

---

## 1. Web Search Design Philosophy

The interface follows the principles of **privacy, clarity, and boutique editorial typography**:
1. **Ad-Free Serenity**: Zero banner ads, zero sponsored promotions, zero tracking cookies. Just pure, organic internet knowledge.
2. **Merino Parchment Canvas (`#F5EEDD`)**: A warm, natural cream that eliminates clinical glare and provides an editorial aesthetic.
3. **Venice Blue Focal Strength (`#16587B`)**: Deep oceanic blue that anchors brand identity, links, primary actions, and selected tabs.
4. **Rock Blue Soft Accents (`#84B3CE`)**: Gentle mist blue providing focus rings, subtle badges, and card hover highlights.
5. **Multi-Modal Ergonomics**: Coverage of all primary search verticals (All, Images, Videos, News, Shopping, Local Files) and Visual Photo Search.
6. **Transparent Privacy Indicators**: Live telemetry on trackers stripped, ad domains blocked, and on-device neural reranker confidence.

---

## 2. Color System & Semantic Tokens

```css
:root {
  /* Core Palette */
  --color-merino: #F5EEDD;         /* Warm Parchment Cream (Page Canvas) */
  --color-rock-blue: #84B3CE;       /* Soft Oceanic Sky Blue (Rings, Hover, Badges) */
  --color-venice-blue: #16587B;     /* Deep Oceanic Navy (Logo, Primary Buttons, Links) */
  --color-deep-abyss: #0D384F;      /* High-Contrast Body Text (WCAG AAA 9.8:1) */
  --color-slate-mist: #527488;      /* Muted Secondary Text & Telemetry */

  /* Canvas & Background Surfaces */
  --bg-canvas: #F5EEDD;
  --bg-surface: #FFFFFF;
  --bg-surface-subtle: #FAF6ED;

  /* Borders & Focus */
  --border-subtle: #E2D9C5;
  --border-strong: #84B3CE;
  --border-focus: #16587B;
  --focus-glow: rgba(132, 179, 206, 0.30);

  /* Typography */
  --text-primary: #0D384F;
  --text-body: #0D384F;
  --text-muted: #527488;
  --text-link: #16587B;

  /* Brand Accent */
  --accent-primary: #16587B;
  --accent-subtle: rgba(132, 179, 206, 0.28);
  --privacy-green: #15803D;
}
```

---

## 3. Typography Hierarchy

* **Brand Wordmark & Display Headers**: `Fraunces`, Georgia, serif (600–800 weight)
* **Interface & Body Text**: `Plus Jakarta Sans`, sans-serif (400–700 weight)
* **Code & Telemetry**: `JetBrains Mono`, monospace

---

## 4. Component Specifications

### 4.1 Hero Search Bar with Google Lens Camera & Autocomplete
* **Search Input Shell**: White surface with `#DCD0B6` border. On focus, transitions to `#16587B` border with a 4px soft ring of `rgba(132, 179, 206, 0.30)`.
* **Photo Search Button (📷)**: Direct trigger for visual reverse image search with `#16587B` camera icon.
* **Live Autocomplete Dropdown**: Query suggestions with `#FAF6ED` hover state and keyboard navigation.

### 4.2 Category Navigation Tabs
* **🌐 All Web**: Cross-Encoder neural reranking and Wikipedia instant answers.
* **🖼️ Images**: High-resolution image grid with dimension chips, hover zoom, and lightbox preview.
* **🎬 Videos**: Video list with platform badge, duration indicator, and video thumbnails.
* **📰 News**: Breaking news cards with source publication, timestamps, and article snippets.
* **🛍️ Shopping**: E-commerce product cards with price tags (`$XX.XX`) and direct store links.
* **📁 Local Docs**: Offline search across local code and documentation files.

### 4.3 Photo Search Modal
* Drag-and-drop file upload zone for `.jpg`, `.png`, `.webp`.
* Image URL input for searching public web images.
* On-device visual analysis panel (dimensions, aspect ratio, perceptual hash).
* Visually similar image matching grid + direct 1-click external privacy reverse search links (Google Lens, Bing Visual, TinEye).

### 4.4 Full-Resolution Image Lightbox
* Responsive dark overlay lightbox for inspecting full-resolution image assets with direct links to source domains.

### 4.5 Privacy Shield & Telemetry Bar
* Real-time privacy metrics:
  - `🛡️ PRIVACY SHIELD: Active`
  - `X Trackers Stripped`
  - `Y Ads Blocked`
  - `Zero Cookies / IP Logging`
