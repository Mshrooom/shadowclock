#!/usr/bin/env python3
"""
generate_real_scenes.py - REAL urban geometry, rendered with the same sun-sweep pipeline.

For each real city block it
  1. downloads OpenStreetMap building footprints (+ height / levels tags) with the Overpass API (cached on disk),
  2. extrudes them in Blender (one merged mesh, flat ground, no baked-in photo texture so there are no fake shadows),
  3. places the sun from the solar ephemeris at the block's REAL latitude/longitude (times are local mean solar time),
  4. renders north-up nadir orthographic images and writes the same files as generate_dataset.py
     (images/, metadata.jsonl, scenes/scene_XXX_height.npy, optional shadow_gt/), so make_pairs.py, baseline.py,
     run_vlm_local.py and evaluate_hypotheses.py work unchanged.

Run (needs internet for the first download; afterwards add --offline):
    python generate_real_scenes.py --out data_real --cities default --samples 32 --denoise --device gpu
or inside Blender:  blender -b -P generate_real_scenes.py -- --out data_real ...

Honest scope: this is real building footprints and (where OSM has them) real heights, rendered synthetically. It is NOT a
photograph. Many OSM buildings carry no height tag; those get type-based defaults and the fraction is recorded in
scenes/scene_XXX.json. The ground is flat (no terrain) and has no roads, trees or vehicles.
Data (c) OpenStreetMap contributors, ODbL - keep the attribution and check the licence terms before redistributing.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
from solar import solar_position, sun_vector_enu  # noqa: E402

# approximate city-centre coordinates (deg). Check that the block has buildings in OSM before relying on a city. [VERIFY]
CITIES = {
    "delhi": (28.6315, 77.2167), "london": (51.5155, -0.0922), "newyork": (40.7549, -73.9840),
    "tokyo": (35.6896, 139.6917), "madrid": (40.4168, -3.7038), "chicago": (41.8819, -87.6278),
    "sydney": (-33.8688, 151.2093), "melbourne": (-37.8136, 144.9631), "santiago": (-33.4489, -70.6693),
    "buenosaires": (-34.6037, -58.3816), "capetown": (-33.9249, 18.4241), "johannesburg": (-26.2041, 28.0473),
}
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
TYPE_HEIGHT = {"house": 7.0, "detached": 7.0, "semidetached_house": 7.0, "terrace": 8.0, "residential": 12.0,
               "apartments": 18.0, "dormitory": 15.0, "commercial": 20.0, "office": 25.0, "retail": 10.0,
               "industrial": 9.0, "warehouse": 9.0, "garage": 3.5, "garages": 3.5, "shed": 3.0, "roof": 4.0,
               "hotel": 22.0, "hospital": 18.0, "school": 10.0, "church": 14.0, "public": 14.0}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--cities", default="default",
                   help="'default' (6 northern + 6 southern cities) or a comma list of names from the built-in table, "
                        "or name:lat:lon entries for your own blocks, e.g. 'delhi,myblock:28.61:77.21'")
    p.add_argument("--scene-start", type=int, default=100, help="first scene id (100+ avoids clashing with the synthetic set)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--gsd", type=float, default=0.5)
    p.add_argument("--pad-m", type=float, default=150.0, help="extra buildings around the field so their shadows fall into the image")
    p.add_argument("--dates", default="2027-03-20,2027-06-21,2027-12-21")
    p.add_argument("--step-min", type=int, default=60)
    p.add_argument("--min-el", type=float, default=20.0)
    p.add_argument("--default-height", type=float, default=10.0)
    p.add_argument("--min-buildings", type=int, default=15, help="skip a block with fewer usable buildings")
    p.add_argument("--osm-cache", default=None, help="folder for the cached Overpass answers (default: <out>/osm_cache)")
    p.add_argument("--offline", action="store_true", help="never touch the network: use cached files only")
    p.add_argument("--user-agent", default="shadowclock-benchmark/1.0 (academic research; OSM building footprints)",
                   help="Overpass rejects generic Python user agents (HTTP 406); put your own project name/e-mail here if asked")
    p.add_argument("--overpass-url", default=None, help="use this Overpass endpoint first (e.g. a mirror)")
    p.add_argument("--check-network", action="store_true", help="only test the Overpass endpoints and print what they answer")
    p.add_argument("--print-queries", action="store_true",
                   help="print the Overpass query and cache path for each city (manual download via overpass-turbo.eu) and exit")
    p.add_argument("--download-only", action="store_true",
                   help="only download/cache the OSM data and print per-block statistics (no rendering): use it to inspect the blocks first")
    p.add_argument("--engine", default="cycles", choices=["cycles", "eevee"])
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--denoiser", default="oidn", choices=["oidn", "optix"])
    p.add_argument("--threads", type=int, default=0, help="CPU render threads (0 = all); try 2-4 on out-of-memory errors")
    p.add_argument("--ambient", type=float, default=0.6)
    p.add_argument("--horiz-irradiance", type=float, default=3.0)
    p.add_argument("--physical-brightness", action="store_true")
    p.add_argument("--shadow-gt", action="store_true", help="also write analytic shadow masks (slow on dense blocks)")
    p.add_argument("--render-retries", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)



# ----------------------------------------------------------------------------- renderer helpers (self-contained)
ROOF_COLORS = [  # linear RGB, deliberately non-green so roofs and ground separate by hue
    (0.45, 0.45, 0.47), (0.55, 0.42, 0.30), (0.50, 0.20, 0.12),
    (0.30, 0.34, 0.42), (0.62, 0.60, 0.55), (0.40, 0.30, 0.28),
]


def make_diffuse_material(name, rgb):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True  # deprecated (always on) in Blender 5.x, still needed in 3.x/4.x
    except Exception:  # noqa: BLE001
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def setup_cycles_device(device):
    if device != "gpu":
        bpy.context.scene.cycles.device = "CPU"
        return
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for fn in ("refresh_devices", "get_devices"):
            if hasattr(prefs, fn):
                getattr(prefs, fn)()
                break
        for dev_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = dev_type
            except TypeError:
                continue
            for fn in ("refresh_devices", "get_devices"):
                if hasattr(prefs, fn):
                    getattr(prefs, fn)()
                    break
            gpus = [d for d in prefs.devices if d.type == dev_type]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type == dev_type)
                bpy.context.scene.cycles.device = "GPU"
                print(f"[real] Cycles GPU backend: {dev_type} ({', '.join(d.name for d in gpus)})")
                return
    except Exception as exc:  # noqa: BLE001
        print(f"[real] GPU setup failed ({exc})")
    print("[real] WARNING: no usable GPU found, falling back to CPU")
    bpy.context.scene.cycles.device = "CPU"


def pick_engine(scene, args):
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = bool(args.denoise)
        if args.denoise:
            try:
                scene.cycles.denoiser = "OPTIX" if args.denoiser == "optix" else "OPENIMAGEDENOISE"
            except TypeError:
                print("[real] WARNING: requested denoiser not available, using OpenImageDenoise")
                scene.cycles.denoiser = "OPENIMAGEDENOISE"
        if args.threads:
            scene.render.threads_mode = "FIXED"
            scene.render.threads = int(args.threads)
        setup_cycles_device(args.device)
    else:
        names = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        eevee = [n for n in names if "EEVEE" in n]
        if not eevee:
            raise RuntimeError(f"no EEVEE engine in this Blender build: {names}")
        scene.render.engine = sorted(eevee)[-1]


def build_environment(ground_rgb, args):
    """Empty scene with flat ground, uniform sky, a small-angle sun and a north-up nadir orthographic camera."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    col = bpy.data.collections.new("shadowclock")
    scene.collection.children.link(col)
    half = 400.0
    gmesh = bpy.data.meshes.new("ground")
    gmesh.from_pydata([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)], [], [(0, 1, 2, 3)])
    gmesh.update()
    gmesh.materials.append(make_diffuse_material("ground", ground_rgb))
    gobj = bpy.data.objects.new("ground", gmesh)
    col.objects.link(gobj)

    world = bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:  # noqa: BLE001
        pass
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.85, 0.92, 1.0, 1.0)
    bg.inputs["Strength"].default_value = args.ambient

    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.angle = math.radians(0.53)
    sun_obj = bpy.data.objects.new("sun", sun_data)
    col.objects.link(sun_obj)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = args.size * args.gsd
    cam_data.clip_start, cam_data.clip_end = 1.0, 3000.0
    cam_obj = bpy.data.objects.new("cam", cam_data)
    cam_obj.location = (0.0, 0.0, 1000.0)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)  # looks down -Z: image up = +Y (north), image right = +X (east)
    col.objects.link(cam_obj)
    scene.camera = cam_obj

    r = scene.render
    r.resolution_x = r.resolution_y = args.size
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    pick_engine(scene, args)
    return scene, sun_obj, col


def aim_sun(sun_obj, az_deg, el_deg, args):
    to_sun = Vector(sun_vector_enu(az_deg, el_deg))
    sun_obj.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()  # light travels along local -Z
    if args.physical_brightness:
        energy = args.horiz_irradiance
    else:
        energy = args.horiz_irradiance / math.sin(math.radians(el_deg))  # keeps horizontal-surface brightness constant
    sun_obj.data.energy = energy


def image_ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def render_checked(scene, sun_obj, az, el, img_abs, args):
    """Render one frame and verify the PNG exists; retry on failure."""
    for attempt in range(1 + args.render_retries):
        aim_sun(sun_obj, az, el, args)
        scene.render.filepath = img_abs.replace("\\", "/")
        status = None
        try:
            status = bpy.ops.render.render(write_still=True)
        except Exception as exc:  # noqa: BLE001 - out-of-memory, driver reset, ...
            print(f"[real] WARNING render raised {type(exc).__name__}: {exc}", flush=True)
            if "out of memory" in str(exc).lower() and attempt == 0:
                print("[real] HINT: out of memory. Close other programs, or use --threads 2, or drop --denoise and raise "
                      "--samples (e.g. 96), or render on a machine with more memory (e.g. Kaggle).", flush=True)
            time.sleep(2.0)
        if image_ok(img_abs):
            return True
        print(f"[real] WARNING no image written (status={status}) for {os.path.basename(img_abs)}, "
              f"attempt {attempt + 1}/{1 + args.render_retries}", flush=True)
    return False


def analytic_shadow_mask(H, gsd, az_deg, el_deg):
    """Exact cast-shadow mask of a height field for a directional sun (ray marching toward the sun)."""
    n = H.shape[0]
    az, el = math.radians(az_deg), math.radians(el_deg)
    sx, sy, t = math.sin(az), math.cos(az), math.tan(el)
    rows, cols = np.mgrid[0:n, 0:n]
    shadow = np.zeros((n, n), dtype=bool)
    d_max = float(H.max()) / t + gsd
    d = gsd
    while d <= d_max:
        rr = np.rint(rows - sy * d / gsd).astype(np.int64)
        cc = np.rint(cols + sx * d / gsd).astype(np.int64)
        ok = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
        hh = np.zeros_like(H)
        hh[ok] = H[rr[ok], cc[ok]]
        shadow |= hh > (H + d * t + 1e-3)
        d += gsd
    return shadow


# ----------------------------------------------------------------------------- OpenStreetMap
def overpass_query(lat, lon, half_m):
    dlat = half_m / 110574.0
    dlon = half_m / (111320.0 * math.cos(math.radians(lat)))
    bbox = f"{lat - dlat:.6f},{lon - dlon:.6f},{lat + dlat:.6f},{lon + dlon:.6f}"
    return f'[out:json][timeout:90];way["building"]({bbox});out geom;'


def endpoints(args):
    return ([args.overpass_url] if args.overpass_url else []) + [u for u in OVERPASS_URLS if u != args.overpass_url]


def check_network(args):
    ok_any = False
    for url in endpoints(args):
        status_url = url.replace("/interpreter", "/status")
        req = urllib.request.Request(status_url, headers={"User-Agent": args.user_agent, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"[net] OK   {status_url}  HTTP {resp.status}: {resp.read().decode(errors='replace')[:160]!r}")
                ok_any = True
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:200]
            print(f"[net] FAIL {status_url}  HTTP {exc.code} {exc.reason}: {body!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"[net] FAIL {status_url}  {type(exc).__name__}: {exc}")
    print("[net] at least one endpoint answered" if ok_any else
          "[net] no endpoint answered: try another network/VPN, --overpass-url <mirror>, or the manual route (--print-queries)")
    return ok_any


def fetch_osm(name, lat, lon, half_m, cache_dir, offline, args=None):
    path = os.path.join(cache_dir, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if offline:
        raise FileNotFoundError(f"--offline but no cached file {path}")
    os.makedirs(cache_dir, exist_ok=True)
    query = overpass_query(lat, lon, half_m)
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    ua = args.user_agent if args else "shadowclock-benchmark/1.0 (academic research)"
    for attempt in range(4):
        for url in (endpoints(args) if args else OVERPASS_URLS):
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": ua, "Accept": "*/*"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    osm = json.loads(resp.read().decode())
                with open(path, "w") as f:
                    json.dump(osm, f)
                print(f"[real] {name}: downloaded {len(osm.get('elements', []))} OSM elements", flush=True)
                return osm
            except urllib.error.HTTPError as exc:
                last = exc
                body = exc.read().decode(errors="replace")[:150]
                print(f"[real] {name}: {url} -> HTTP {exc.code} {exc.reason} {body!r}; retrying", flush=True)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                print(f"[real] {name}: {url} failed ({type(exc).__name__}: {exc}); retrying", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Overpass download failed for {name}: {last}")


def parse_height(tags, default):
    """Return (height_m, source) with source in {'height', 'levels', 'type', 'default'}."""
    h = tags.get("height") or tags.get("building:height")
    if h:
        m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(m|ft|')?", str(h).replace(",", "."))
        if m:
            val = float(m.group(1))
            if m.group(2) in ("ft", "'"):
                val *= 0.3048
            if 2.0 <= val <= 400.0:
                return val, "height"
    lv = tags.get("building:levels")
    if lv:
        try:
            n = float(str(lv).replace(",", "."))
            if 1 <= n <= 120:
                return n * 3.2 + 1.0, "levels"
        except ValueError:
            pass
    t = tags.get("building")
    if t in TYPE_HEIGHT:
        return TYPE_HEIGHT[t], "type"
    return default, "default"


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def is_simple(poly):
    n = len(poly)
    if n > 80:
        return True  # too long to test cheaply; accept
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return False
    return True


def parse_buildings(osm, lat0, lon0, default_h, limit_m):
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110574.0
    buildings, skipped = [], 0
    for el in osm.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        pts = [((g["lon"] - lon0) * kx, (g["lat"] - lat0) * ky) for g in el["geometry"]]
        if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-6 and abs(pts[0][1] - pts[-1][1]) < 1e-6:
            pts = pts[:-1]
        clean = [pts[0]] if pts else []
        for p in pts[1:]:
            if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 0.05:
                clean.append(p)
        if len(clean) < 3:
            skipped += 1
            continue
        xs, ys = [p[0] for p in clean], [p[1] for p in clean]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        if abs(cx) > limit_m or abs(cy) > limit_m:
            continue
        area = 0.5 * sum(clean[i][0] * clean[(i + 1) % len(clean)][1] - clean[(i + 1) % len(clean)][0] * clean[i][1]
                         for i in range(len(clean)))
        if abs(area) < 8.0 or not is_simple(clean):
            skipped += 1
            continue
        if area < 0:
            clean.reverse()  # counter-clockwise => outward wall normals, upward roof normal
        tags = el.get("tags", {})
        h, src = parse_height(tags, default_h)
        buildings.append(dict(poly=clean, h=float(h), h_source=src, osm_id=el.get("id"), btype=tags.get("building")))
    return buildings, skipped


def raster_heights(buildings, n_px, gsd):
    """Exact height map (metres) on an n_px x n_px grid centred on the block, row 0 = north edge. Even-odd polygon fill."""
    c = (np.arange(n_px) + 0.5 - n_px / 2.0) * gsd
    r = (n_px / 2.0 - np.arange(n_px) - 0.5) * gsd
    H = np.zeros((n_px, n_px), dtype=np.float32)
    for b in buildings:
        poly = np.asarray(b["poly"])
        xmin, ymin = poly.min(0)
        xmax, ymax = poly.max(0)
        cols = np.nonzero((c >= xmin) & (c <= xmax))[0]
        rows = np.nonzero((r >= ymin) & (r <= ymax))[0]
        if cols.size == 0 or rows.size == 0:
            continue
        X, Y = np.meshgrid(c[cols], r[rows])
        inside = np.zeros(X.shape, dtype=bool)
        x1, y1 = poly[:, 0], poly[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        for xa, ya, xb, yb in zip(x1, y1, x2, y2):
            inside ^= ((ya > Y) != (yb > Y)) & (X < (xb - xa) * (Y - ya) / (yb - ya + 1e-12) + xa)
        sub = H[np.ix_(rows, cols)]
        sub[inside] = np.maximum(sub[inside], b["h"])
        H[np.ix_(rows, cols)] = sub
    return H


# ----------------------------------------------------------------------------- Blender geometry
def add_city_mesh(buildings, collection):
    verts, faces, mat_idx = [], [], []
    for b in buildings:
        poly, n, base = b["poly"], len(b["poly"]), len(verts)
        verts += [(x, y, 0.0) for x, y in poly] + [(x, y, b["h"]) for x, y in poly]
        k = (b["osm_id"] or 0) % len(ROOF_COLORS)
        faces.append([base + n + i for i in range(n)])  # roof (counter-clockwise => normal up)
        mat_idx.append(k)
        for i in range(n):
            j = (i + 1) % n
            faces.append([base + i, base + j, base + n + j, base + n + i])  # wall
            mat_idx.append(k)
    mesh = bpy.data.meshes.new("city")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for k, col in enumerate(ROOF_COLORS):
        mesh.materials.append(make_diffuse_material(f"roof_{k}", col))
    mesh.polygons.foreach_set("material_index", mat_idx)
    mesh.update()
    obj = bpy.data.objects.new("city", mesh)
    collection.objects.link(obj)
    return obj


def time_grid_local(lat, lon, date_str, step_min, min_el):
    """Local MEAN solar clock times (minutes after local mean midnight) with the sun above min_el."""
    y, m, d = (int(x) for x in date_str.split("-"))
    minutes = np.arange(0, 24 * 60, step_min)
    az, el = solar_position(lat, lon, y, m, d, minutes / 60.0 - lon / 15.0)  # UTC hour = local mean hour - lon/15
    return [(int(mm), float(a), float(e)) for mm, a, e in zip(minutes, az, el) if e >= min_el]


def resolve_cities(spec):
    if spec == "default":
        names = ["delhi", "london", "newyork", "tokyo", "madrid", "chicago",
                 "sydney", "melbourne", "santiago", "buenosaires", "capetown", "johannesburg"]
        return [(n, *CITIES[n]) for n in names]
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 3:
            out.append((parts[0], float(parts[1]), float(parts[2])))
        elif item in CITIES:
            out.append((item, *CITIES[item]))
        else:
            raise SystemExit(f"unknown city '{item}': use a built-in name or name:lat:lon")
    return out


def main():
    args = parse_args()
    args.out = os.path.abspath(args.out)
    if args.check_network:
        sys.exit(0 if check_network(args) else 1)
    cache_dir = os.path.abspath(args.osm_cache) if args.osm_cache else os.path.join(args.out, "osm_cache")
    half_field = args.size * args.gsd / 2.0
    if half_field + args.pad_m > 380.0:
        raise SystemExit("size*gsd/2 + pad-m must be <= 380 m (the ground plane is 800 m wide)")
    if args.print_queries:
        for name, lat, lon in resolve_cities(args.cities):
            print(f"=== {name}: paste into https://overpass-turbo.eu , Run, Export -> raw OSM data (JSON), save as\n    "
                  f"{os.path.join(cache_dir, name + '.json')}\n{overpass_query(lat, lon, half_field + args.pad_m)}\n")
        return
    pad_px = int(round(args.pad_m / args.gsd))
    n_pad = args.size + 2 * pad_px
    for sub in ("images", "scenes") + (("shadow_gt",) if args.shadow_gt else ()):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    dates = [x.strip() for x in args.dates.split(",") if x.strip()]
    meta_path = os.path.join(args.out, "metadata.jsonl")
    cities = resolve_cities(args.cities)
    lo, hi = args.scene_start, args.scene_start + len(cities)
    all_rows = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            all_rows = [r for r in (json.loads(l) for l in f) if not (lo <= r["scene"] < hi)]

    failed, consecutive, t0 = [], 0, time.time()
    for i, (name, lat, lon) in enumerate(cities):
        s = args.scene_start + i
        osm = fetch_osm(name, lat, lon, half_field + args.pad_m, cache_dir, args.offline, args)
        buildings, skipped = parse_buildings(osm, lat, lon, args.default_height, half_field + args.pad_m)
        if len(buildings) < args.min_buildings:
            print(f"[real] {name}: only {len(buildings)} usable buildings (skipped {skipped}); block dropped", flush=True)
            continue
        src_count = {k: sum(b["h_source"] == k for b in buildings) for k in ("height", "levels", "type", "default")}
        if args.download_only:
            print(f"[real] {name}: {len(buildings)} usable buildings (skipped {skipped} invalid/tiny polygons), "
                  f"explicit heights {100 * (src_count['height'] + src_count['levels']) / len(buildings):.0f}% "
                  f"(height tag {src_count['height']}, levels tag {src_count['levels']}, by type {src_count['type']}, default {src_count['default']}), "
                  f"tallest {max(b['h'] for b in buildings):.0f} m", flush=True)
            continue
        H_pad = raster_heights(buildings, n_pad, args.gsd)
        H = H_pad[pad_px:pad_px + args.size, pad_px:pad_px + args.size].copy()
        np.save(os.path.join(args.out, "scenes", f"scene_{s:03d}_height.npy"), H)
        info = dict(scene=s, city=name, lat=lat, lon=lon, gsd=args.gsd, size=args.size, n_buildings=len(buildings),
                    skipped_polygons=skipped, height_sources=src_count,
                    explicit_height_fraction=(src_count["height"] + src_count["levels"]) / len(buildings),
                    max_height_m=float(max(b["h"] for b in buildings)),
                    built_fraction_of_field=float((H > 0).mean()), attribution="OpenStreetMap contributors (ODbL)")
        with open(os.path.join(args.out, "scenes", f"scene_{s:03d}.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(f"[real] scene {s} {name}: {len(buildings)} buildings, explicit heights {100 * info['explicit_height_fraction']:.0f}%, "
              f"max {info['max_height_m']:.0f} m, built-up {100 * info['built_fraction_of_field']:.0f}% of the field", flush=True)

        scene, sun_obj, col = build_environment((0.10, 0.30, 0.06), args)
        add_city_mesh(buildings, col)

        rows = []
        for date_str in dates:
            for minute, az, el in time_grid_local(lat, lon, date_str, args.step_min, args.min_el):
                stem = f"s{s:03d}_{name}_{date_str}_{minute // 60:02d}{minute % 60:02d}"
                img_rel = "images/" + stem + ".png"
                img_abs = os.path.join(args.out, "images", stem + ".png")
                sh_rel = ("shadow_gt/" + stem + ".npz") if args.shadow_gt else None
                if args.overwrite or not image_ok(img_abs):
                    if render_checked(scene, sun_obj, az, el, img_abs, args):
                        consecutive = 0
                    else:
                        failed.append(img_rel)
                        consecutive += 1
                        if consecutive >= 3:
                            print(f"[real] ABORT: 3 renders in a row produced no file (expected {img_abs}). "
                                  "Look at the 'Saved:' line Blender printed above.", flush=True)
                            sys.exit(3)
                        continue
                if sh_rel:
                    sh_abs = os.path.join(args.out, "shadow_gt", stem + ".npz")
                    if args.overwrite or not os.path.exists(sh_abs):
                        full = analytic_shadow_mask(H_pad, args.gsd, az, el)  # padded canvas: shadows of outside buildings count
                        np.savez_compressed(sh_abs, shadow=full[pad_px:pad_px + args.size, pad_px:pad_px + args.size])
                rows.append(dict(id=stem, scene=s, city=name, image=img_rel, shadow_gt=sh_rel, lat=lat, lon=lon, date=date_str,
                                 utc_minute=minute, utc_hour=minute / 60.0, time_basis="local_mean_solar",
                                 sun_az_deg=az, sun_el_deg=el, gsd_m=args.gsd, size_px=args.size))
        all_rows.extend(rows)
        all_rows.sort(key=lambda r: (r["scene"], r["date"], r["utc_minute"]))
        with open(meta_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[real] scene {s} {name}: {len(rows)} images listed ({len(failed)} failed so far, {time.time() - t0:.0f}s elapsed)", flush=True)
    if args.download_only:
        print("[real] download-only run finished (nothing rendered). Check the numbers above, then run again without --download-only.")
        return
    print(f"[real] done. metadata: {meta_path} ({len(all_rows)} rows)")
    if failed:
        print(f"[real] ERROR: {len(failed)} images could not be rendered; re-run the same command to retry only those.")
        sys.exit(2)


if __name__ == "__main__":
    main()
