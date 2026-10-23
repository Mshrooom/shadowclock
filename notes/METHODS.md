# Methods

## Sun position and time
Sun azimuth and elevation come from the NOAA solar-position equations (`pipeline/solar.py`), cross-checked against pvlib
(tests allow 0.1 degrees for elevation above 10 degrees; the equations are less accurate near the horizon, which the
benchmark avoids). `web/src/solar.js` is a line-for-line port; `tests/test_solar.py` runs both on the same inputs and
requires agreement.

"Time" throughout is **local mean solar time** at the scene's longitude. In `metadata.jsonl` the fields `utc_hour` and
`utc_minute` hold the local mean time, not UTC. True UTC = `utc_hour` - longitude / 15. The names were kept for
backward compatibility.

## Geometry
Images are north-up, straight-down, orthographic. East is to the right. A shadow points away from the sun, so the shadow
bearing is the sun azimuth plus 180 degrees. Scenes are flat ground with flat-roofed prisms (OSM footprints extruded to
their height) and no baked-in lighting.

## Scenes
- **Procedural box towns** (`generate_dataset.py`): random boxes, used for the first model runs.
- **OSM-derived scenes** (`generate_real_scenes.py`): building footprints and `height` / `building:levels` tags from the
  Overpass API for twelve 256 m blocks. Where a tag is missing, type-based defaults are used; the explicit-height share is
  stored per scene (26% in Melbourne up to 83% in Sydney). The result is simulated imagery, not a photograph.

## Pairs
Two images of the same scene, latitude and date with a time gap of 1, 2 or 4 hours, both with sun elevation between 20 and
70 degrees (near-zenith sun leaves almost no shadow). Each pair also appears with A and B swapped. Accuracy and
intervals use distinct pairs only (Wilson 95%).

## Baseline
`pipeline/baseline.py` uses only the pixels plus the metadata a model gets in each regime. It segments ground, shadow and roof,
estimates each building's shadow axis (matched filter on the roof-plus-shadow footprint) with a roof-edge fallback, and takes a
robust circular mean as the shadow azimuth. Sun azimuth is shadow azimuth plus 180 degrees. R0 assumes the northern
hemisphere; R1 orders by the rotation direction of the azimuth; R2 inverts the solar ephemeris to a clock time per image.
It does not read the renderer's true sun direction. It is an existence proof that the information is in the pixels, not a model
of what a vision-language model should do.

## Viewer
Height maps (0.5 m cells, uint16 decimetres, zlib) become a merged-run prism mesh in three.js. A directional light with a
4096 px shadow map casts the shadows. In the benchmark view the brightness is flat (E0 / sin(elevation)) so brightness is not
a clue to the time. The wall shading uses normals from a blurred footprint mask to avoid stair-step stripes on diagonal
walls; the shadows themselves use the true geometry. Viewer shadows were compared with exact analytic shadow masks.
