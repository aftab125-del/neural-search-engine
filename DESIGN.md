# UI/UX Design System: Private, Multi-Modal Neural Web Search Engine

**Project**: NeuralSearch  
**Path**: `D:\neural-search-engine\DESIGN.md`  
**Aesthetic Direction**: Light Editorial Minimalism (Warm Alabaster & Burnt Terracotta)  

---

## 1. Web Search Design Philosophy

The interface follows the principles of **privacy, clarity, and zero commercial noise**:
1. **Ad-Free Serenity**: Zero banner ads, zero sponsored promotions, zero flashing widgets. Just pure, organic internet knowledge.
2. **Warm Alabaster Canvas (`#FCF2E5`)**: Avoids harsh blinding pure white or clinical grays, creating a calm reading environment.
3. **Multi-Modal Ergonomics**: Comprehensive coverage of all primary Google search verticals (All, Images, Videos, News, Shopping, Local Files) and Visual Photo Search.
4. **Transparent Privacy Indicators**: Highlights to the user that their query was anonymized and displays live metrics on trackers purged and ads blocked.

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

## 3. Component Specifications

### 3.1 Hero Search Bar with Google Lens Camera & Autocomplete
* **Search Input Bar**: Prominent rounded bar with auto-focus, real-time debouncing, and keyboard shortcut (`/`).
* **Photo Search Button (📷)**: Direct trigger for visual reverse image search.
* **Live Autocomplete Dropdown**: Instant query suggestions (<50ms) as the user types with keyboard arrow navigation.

### 3.2 Category Navigation Tabs
* **🌐 All Web**: General search across the entire internet with Cross-Encoder neural reranking and Wikipedia instant answers.
* **🖼️ Images**: High-resolution masonry image grid with dimensions tags, hover zoom, and lightbox preview.
* **🎬 Videos**: Rich video list with platform badge (YouTube, Web), duration indicator, and video thumbnails.
* **📰 News**: Breaking news cards with source publication, timestamps ("2 hours ago"), and article snippets.
* **🛍️ Shopping**: E-commerce product cards with price tags (`$XX.XX`), store badges (Amazon, Best Buy, Walmart, eBay), and direct links.
* **📁 Local Docs**: Offline search across local code and documentation files.

### 3.3 Photo Search (Google Lens Equivalent) Modal
* Drag-and-drop file upload zone for `.jpg`, `.png`, `.webp`.
* Image URL input for searching public web images.
* On-device computer vision inspection panel (dimensions, aspect ratio, perceptual hash).
* Visually similar image matching grid + direct 1-click external privacy reverse search links (Google Lens, Bing Visual, TinEye).

### 3.4 Full-Resolution Image Lightbox
* Responsive dark overlay lightbox for inspecting full-resolution image assets with direct links to source domains.

### 3.5 Privacy Shield & Telemetry Bar
* Real-time privacy metrics:
  - `🛡️ PRIVACY SHIELD: Active`
  - `X Trackers Stripped`
  - `Y Ads Blocked`
  - `Zero Cookies / IP Logging`
