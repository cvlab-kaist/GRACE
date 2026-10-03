# GRACE project page

Static site, no build step. Open `index.html` locally or push to GitHub Pages.

## Files
- `index.html` main page, `gallery.html` gallery page
- `static/js/data.js` every video path and prompt on both pages (edit this)
- `static/css/style.css` colors and layout (tokens at the top: `--hero`, `--accent`, ...)
- `static/images/` paper figures (Fig. 2, 3, 4) and logos

## Adding videos
1. Put mp4 files under `static/videos/` (H.264, muted; 3–8 MB each keeps the gallery fast).
2. Edit the paths in `static/js/data.js`. Missing files show a grey box, so you can fill them in gradually.
3. In the gallery, `vsWan` is the main Wan2.1 vs GRACE set, `moreBaselines` adds DC-Gen, `samples` is GRACE only.
   Delete the `fillExamples` block at the bottom of `data.js` once your own entries are in.

Tip: `ffmpeg -i in.mp4 -c:v libx264 -crf 26 -preset slow -an -movflags +faststart out.mp4`

## Before publishing
- Paper / arXiv links: `href="#"` in `index.html` (search for `class="pill"`).
- Code: currently shown as "coming soon"; replace the `<span class="pill" aria-disabled...>` with an `<a class="pill" href="...">` when the repo is public.
- BibTeX: replace `arXiv:[id]`.
- Author homepages: wrap names in `<a href="...">` inside `.authors`.

## Deploy on GitHub Pages
Push this folder to a repo, then Settings > Pages > Deploy from branch (main, root).
