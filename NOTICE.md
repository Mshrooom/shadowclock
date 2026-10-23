# Data notices

## Code

MIT, see `LICENSE`. This covers everything under `pipeline/`, `scripts/`, `web/src/`, `tests/` and `notebooks/`.

## OpenStreetMap-derived data (ODbL)

The twelve city scenes are built from building footprints and height tags from **OpenStreetMap**.

> Contains information from OpenStreetMap, (c) OpenStreetMap contributors, available under the
> Open Database Licence (ODbL) 1.0. https://www.openstreetmap.org/copyright

The following files are a *derived database* of that data and carry the same licence and attribution (ODbL 1.0):

- `data/heightmaps/*.npz` (rasterised building heights, 0.5 m cells)
- `docs/assets/cities.js` (the same heights, compressed for the web viewer)
- `data/samples/*.jpg` (renders of those heights)
- `results/pairs_real.jsonl`, `results/real_per_city.*` (derived statistics)

If you redistribute or build on these files, keep this notice and share the derived data under ODbL.
This is a plain-language summary, not legal advice. Read the licence at https://opendatacommons.org/licenses/odbl/1-0/.

## Basemaps in `docs/map.html`

Tiles are loaded from the internet at view time and are not stored in this repository:
CARTO Positron (data (c) OpenStreetMap contributors, style (c) CARTO) and Esri World Imagery
(Tiles (c) Esri, Maxar, Earthstar Geographics). Check their terms before using the map commercially.

## Third-party software

- three.js (MIT), bundled into `docs/assets/world.js`. Source: https://github.com/mrdoob/three.js
- Leaflet, folium and leafmap (BSD/MIT) are loaded by `docs/map.html` from public CDNs.
- Blender (GPL) is used only as an external renderer through its `bpy` module. No Blender code is included.
- pvlib (BSD-3) is a test-only dependency used to cross-check the solar position code.
- The NOAA solar position equations are public. See https://gml.noaa.gov/grad/solcalc/calcdetails.html.

## Models

No model weights are included. `pipeline/run_vlm_local.py` downloads open models from Hugging Face
(for example Qwen2.5-VL, LLaVA-OneVision) under their own licences.
