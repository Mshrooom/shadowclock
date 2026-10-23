#!/usr/bin/env python3
"""
make_map.py - interactive results map (opengeos/leafmap) of the 12 OSM-derived scenes.

One circle per city, coloured by the pixel baseline's R1 ordering accuracy (one-hue ramp, 50% = chance .. 100%), sized by
the share of the 256 m field that is built up. Click a city: result with 95% interval, scene statistics, three sample
renders and a link into the 4D viewer (world.html) at that city. Basemaps: light vector map and satellite imagery.

    python scripts/make_map.py --results results --samples data/samples --out docs/map.html
"""
import argparse
import base64
import json
import os

import folium
import leafmap.foliumap as leafmap
import numpy as np

NAMES = {"newyork": "New York", "buenosaires": "Buenos Aires", "capetown": "Cape Town"}
ZOOM_EXTENT_M = 256.0


def nice(c):
    return NAMES.get(c, c.capitalize())


def ramp(acc):
    """one-hue blue ramp for 50..100 % (lighter = closer to chance)."""
    t = float(np.clip((acc - 50.0) / 50.0, 0, 1))
    lo, hi = np.array([197, 219, 246]), np.array([12, 68, 140])
    r, g, b = (lo + (hi - lo) * t).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def b64(path):
    return "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode("ascii")


def popup_html(c, thumbs, viewer_href):
    imgs = "".join(
        f'<figure style="margin:0;text-align:center"><img src="{b64(t["path"])}" width="112" height="112" '
        f'style="border-radius:4px;display:block"><figcaption style="font-size:10.5px;color:#555">{int(t["hour"]):02d}:00 · sun {t["el"]:.0f}°</figcaption></figure>'
        for t in thumbs)
    acc = (f'<b>{c["R1_acc"]:.0f}%</b> [{c["R1_ci_lo"]:.0f}-{c["R1_ci_hi"]:.0f}] on {c["R1_n"]} distinct pairs'
           f' (R2: {c["R2_acc"]:.0f}%)')
    return f"""
<div style="font:13px/1.45 system-ui,sans-serif;width:360px">
  <div style="font-size:15px;font-weight:600">{nice(c["city"])}</div>
  <div style="color:#555;margin-bottom:6px">{c["lat"]:.3f}, {c["lon"]:.3f} · {"northern" if c["lat"] >= 0 else "southern"} hemisphere</div>
  <div>Pixel baseline, which image is earlier (R1): {acc}</div>
  <div style="color:#555">{c["n_buildings"]} buildings, {100 * c["built_fraction"]:.0f}% built up, heights from OSM for
  {100 * c["explicit_height_fraction"]:.0f}% (rest assumed), tallest {c["max_height_m"]:.0f} m</div>
  <div style="display:flex;gap:6px;margin:8px 0">{imgs}</div>
  <a href="{viewer_href}" target="_blank" style="font-weight:600">Open in the 4D viewer &rarr;</a>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--out", default="docs/map.html")
    ap.add_argument("--viewer", default="world.html", help="viewer URL relative to the map page")
    args = ap.parse_args()

    cities = json.load(open(os.path.join(args.results, "real_per_city.json")))["cities"]
    index = json.load(open(os.path.join(args.samples, "index.json")))

    m = leafmap.Map(center=[10, 10], zoom=2, draw_control=False, measure_control=False, fullscreen_control=True,
                    layers_control=False, search_control=False, scale_control=True)
    folium.TileLayer("cartodbpositron", name="Light map (CARTO / OpenStreetMap)", control=True).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri, Maxar, Earthstar Geographics", name="Satellite (Esri World Imagery)", control=True).add_to(m)

    fg = folium.FeatureGroup(name="Cities (baseline R1 accuracy)", show=True)
    for c in cities:
        thumbs = [{"path": os.path.join(args.samples, t["file"]), "hour": t["hour"], "el": t["sun_el_deg"]}
                  for t in index if t["city"] == c["city"]]
        href = f'{args.viewer}#city={c["city"]}&date=2027-06-21&t=12&view=orbit'
        radius = 8 + 22 * c["built_fraction"]
        folium.CircleMarker(
            location=[c["lat"], c["lon"]], radius=radius, color="#0b0b0b", weight=1.2, fill=True,
            fill_color=ramp(c["R1_acc"]), fill_opacity=0.92,
            tooltip=f'{nice(c["city"])}: {c["R1_acc"]:.0f}% (R1)',
            popup=folium.Popup(folium.Html(popup_html(c, thumbs, href), script=True), max_width=400),
        ).add_to(fg)
        folium.map.Marker(
            [c["lat"], c["lon"]],
            icon=folium.DivIcon(html=f'<div style="font:600 11px system-ui;color:#0b0b0b;text-shadow:0 0 3px #fff,0 0 3px #fff;'
                                     f'white-space:nowrap;margin:{int(radius) + 2}px 0 0 -20px">{nice(c["city"])}</div>'),
        ).add_to(fg)
    fg.add_to(m)
    lats = [c["lat"] for c in cities]
    lons = [c["lon"] for c in cities]
    m.fit_bounds([[min(lats) - 6, min(lons) - 12], [max(lats) + 6, max(lons) + 12]])
    folium.LayerControl(collapsed=True).add_to(m)

    m.add_colormap(cmap="Blues", vmin=50, vmax=100, label="Pixel baseline accuracy, R1 (50 = chance)")
    legend = """
<div style="position:fixed;left:12px;bottom:34px;z-index:9999;background:rgba(255,255,255,.93);padding:10px 12px;border-radius:8px;
 font:12px/1.4 system-ui,sans-serif;max-width:290px;box-shadow:0 1px 4px rgba(0,0,0,.3)">
 <b>shadowclock: 12 OSM-derived scenes</b><br>Colour = pixel-baseline accuracy at ordering two renders of the same place and date
 (R1). Size = built-up share of the 256 m field. About 30 distinct pairs per city, so intervals are wide.<br>
 <span style="color:#555">Simulated scenes, not imagery. Footprints &copy; OpenStreetMap contributors (ODbL).</span></div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(args.out)  # folium save (leafmap's to_html(title=...) trips over folium markers)
    html = open(args.out, encoding="utf-8").read()
    html = html.replace("<head>", '<head>\n<title>shadowclock results map</title>', 1) if "<title>" not in html else html
    open(args.out, "w", encoding="utf-8").write(html)
    print("map ->", args.out, f"({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
