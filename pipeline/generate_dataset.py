#!/usr/bin/env python3
"""
generate_dataset.py - Blender generator for the "shadow clock" benchmark.

Renders north-up, nadir, orthographic overhead views of procedural building
scenes under a physically placed sun (NOAA solar ephemeris, solar.py), and
saves exact ground truth (sun azimuth/elevation, UTC time, height map and an
analytic shadow mask).

Run inside Blender:
    blender -b -P generate_dataset.py -- --out data --n-scenes 20
or with the pip "bpy" module (Python 3.11 for bpy 4.x/5.0):
    python generate_dataset.py --out data --n-scenes 20

Only numpy is needed besides Blender's own modules (numpy ships with Blender).

Image geometry convention (checked by check_shadows.py):
    image up = north (+Y), image right = east (+X), camera looks straight down.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import bpy
from mathutils import Vector

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pragma: no cover - some embedded interpreters
    HERE = os.getcwd()
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from solar import solar_position, sun_vector_enu  # noqa: E402
from geom import analytic_shadow_mask, rasterize_height_polys  # noqa: E402


# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n-scenes", type=int, default=20)
    p.add_argument("--scene-start", type=int, default=0, help="first scene index (for sharding across runs)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--size", type=int, default=512, help="image size in pixels (square)")
    p.add_argument("--gsd", type=float, default=0.5, help="ground sample distance, metres/pixel")
    p.add_argument("--lats", default="50,30,-30,-50", help="comma separated latitudes (deg, south negative)")
    p.add_argument("--lon", type=float, default=0.0, help="longitude (deg east); 0 => UTC == local mean time")
    p.add_argument("--dates", default="2027-03-20,2027-06-21,2027-12-21",
                   help="comma separated YYYY-MM-DD (3 dates so both hemispheres get a summer date; sun below --min-el is skipped)")
    p.add_argument("--step-min", type=int, default=30, help="time step in minutes")
    p.add_argument("--min-el", type=float, default=20.0, help="skip sun elevations below this (deg)")
    p.add_argument("--engine", default="cycles", choices=["cycles", "eevee"])
    p.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel")
    p.add_argument("--denoise", action="store_true", help="enable OpenImageDenoise in Cycles")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--denoiser", default="oidn", choices=["oidn", "optix"],
                   help="denoiser when --denoise is set: oidn (CPU/any) or optix (NVIDIA RTX with --device gpu) [optix untested]")
    p.add_argument("--threads", type=int, default=0,
                   help="CPU render threads (0 = all). Use 2-4 if you see 'out of memory' errors on a laptop")
    p.add_argument("--ambient", type=float, default=0.6, help="uniform sky (ambient) strength")
    p.add_argument("--horiz-irradiance", type=float, default=3.0,
                   help="sun strength on a horizontal surface; sun strength = this / sin(elevation)")
    p.add_argument("--physical-brightness", action="store_true",
                   help="use constant sun strength so image brightness changes with elevation "
                        "(default keeps horizontal-surface brightness constant so brightness is not a cue)")
    p.add_argument("--no-shadow-gt", action="store_true", help="do not write analytic shadow masks")
    p.add_argument("--osm-scenes", default=None,
                   help="directory of real-layout scene files (*.json from osm_fetch.py / make_demo_scene.py). If given, scenes are "
                        "built from them (real footprints, heights, latitude/longitude) instead of the procedural boxes; "
                        "--lats/--lon are then ignored")
    p.add_argument("--ground", default="green", choices=["green", "urban"],
                   help="ground colour: uniform green (default, used by the procedural dataset) or asphalt-like grey")
    p.add_argument("--overwrite", action="store_true", help="re-render images that already exist")
    p.add_argument("--render-retries", type=int, default=2, help="extra attempts if a render produces no file")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# scene layout and exact ground truth (pure numpy, independent of Blender)
# ----------------------------------------------------------------------------
ROOF_COLORS = [  # linear RGB, deliberately non-green so roofs and ground are separable by hue
    (0.45, 0.45, 0.47), (0.55, 0.42, 0.30), (0.50, 0.20, 0.12),
    (0.30, 0.34, 0.42), (0.62, 0.60, 0.55), (0.40, 0.30, 0.28),
]


def make_layout(rng, size_px, gsd):
    """3x3 jittered grid of rotated rectangular flat-roof buildings (never overlapping)."""
    field = size_px * gsd
    n_cells = 3
    cell = field / n_cells
    while True:
        buildings = []
        for i in range(n_cells):
            for j in range(n_cells):
                if rng.random() > 0.85:
                    continue
                max_dim = min(40.0, 0.47 * cell)  # keep rotated footprints inside their grid cell
                w, d = rng.uniform(0.35 * max_dim, max_dim), rng.uniform(0.35 * max_dim, max_dim)
                theta = rng.uniform(0.0, math.pi)
                h = rng.uniform(6.0, 30.0)
                slack = max(0.0, cell / 2.0 - 0.5 * math.hypot(w, d) - 1.0)
                cx = -field / 2 + (i + 0.5) * cell + rng.uniform(-slack, slack)
                cy = -field / 2 + (j + 0.5) * cell + rng.uniform(-slack, slack)
                color = ROOF_COLORS[int(rng.integers(len(ROOF_COLORS)))]
                buildings.append(dict(cx=float(cx), cy=float(cy), w=float(w), d=float(d),
                                      theta=float(theta), h=float(h), color=list(color)))
        if len(buildings) >= 3:
            return buildings


def pixel_grid(size_px, gsd):
    """World coordinates (east X, north Y, metres) of pixel centres; row 0 is the north edge."""
    c = (np.arange(size_px) + 0.5 - size_px / 2.0) * gsd
    r = (size_px / 2.0 - np.arange(size_px) - 0.5) * gsd
    return np.meshgrid(c, r)  # X[r, c] = east of column c, Y[r, c] = north of row r


def rasterize_height(buildings, size_px, gsd):
    X, Y = pixel_grid(size_px, gsd)
    H = np.zeros((size_px, size_px), dtype=np.float32)
    for b in buildings:
        dx, dy = X - b["cx"], Y - b["cy"]
        ct, st = math.cos(b["theta"]), math.sin(b["theta"])
        u, v = dx * ct + dy * st, -dx * st + dy * ct
        inside = (np.abs(u) <= b["w"] / 2) & (np.abs(v) <= b["d"] / 2)
        H[inside] = np.maximum(H[inside], b["h"])
    return H



# ----------------------------------------------------------------------------
# Blender scene construction
# ----------------------------------------------------------------------------
def make_diffuse_material(name, rgb):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True  # deprecated (always on) in Blender 5.x, still needed in 3.x/4.x
    except Exception:
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_mesh_object(name, verts, faces, material, collection):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.validate()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    collection.objects.link(obj)
    return obj


def box_mesh(b):
    ct, st = math.cos(b["theta"]), math.sin(b["theta"])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ground, top = [], []
    for sx, sy in corners:
        lx, ly = sx * b["w"] / 2, sy * b["d"] / 2
        x = b["cx"] + lx * ct - ly * st
        y = b["cy"] + lx * st + ly * ct
        ground.append((x, y, 0.0))
        top.append((x, y, b["h"]))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return ground + top, faces


def prism_mesh(b):
    """Flat-roof prism of an arbitrary (concave, with courtyard holes) polygon. Returns (verts, faces) or None."""
    from mathutils import geometry as mgeom
    rings = []
    for k, r in enumerate(b["rings"]):
        r = [tuple(p) for p in r]
        area = 0.5 * sum(r[i][0] * r[(i + 1) % len(r)][1] - r[(i + 1) % len(r)][0] * r[i][1] for i in range(len(r)))
        if (k == 0 and area < 0) or (k > 0 and area > 0):  # outer boundary counter-clockwise, holes clockwise
            r = r[::-1]
        rings.append(r)
    h = float(b["height"])
    tris = mgeom.tessellate_polygon([[Vector((x, y, h)) for x, y in r] for r in rings])
    if not tris:
        return None
    top = [(x, y, h) for r in rings for x, y in r]
    n = len(top)
    verts = top + [(x, y, 0.0) for r in rings for x, y in r]
    faces = [tuple(t) for t in tris]
    off = 0
    for r in rings:
        m = len(r)
        for i in range(m):
            a, c = off + i, off + (i + 1) % m
            faces.append((a, n + a, n + c, c))
        off += m
    return verts, faces


def load_osm_scene(path, rng):
    """Read a real-layout scene file; returns (header dict, list of buildings with rings/height/color)."""
    d = json.load(open(path))
    buildings = []
    for b in d["buildings"]:
        rings = [[(float(x), float(y)) for x, y in r] for r in b["rings"]]
        buildings.append(dict(id=b.get("id"), rings=rings, height=float(b["height"]),
                              color=list(ROOF_COLORS[int(rng.integers(len(ROOF_COLORS)))])))
    return d, buildings


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
                print(f"[generate] Cycles GPU backend: {dev_type} ({', '.join(d.name for d in gpus)})")
                return
    except Exception as exc:  # noqa: BLE001
        print(f"[generate] GPU setup failed ({exc})")
    print("[generate] WARNING: no usable GPU found, falling back to CPU")
    bpy.context.scene.cycles.device = "CPU"


def pick_engine(scene, engine, args):
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = bool(args.denoise)
        if args.denoise:
            try:
                scene.cycles.denoiser = "OPTIX" if getattr(args, "denoiser", "oidn") == "optix" else "OPENIMAGEDENOISE"
            except TypeError:
                print("[generate] WARNING: requested denoiser not available, using OpenImageDenoise")
                scene.cycles.denoiser = "OPENIMAGEDENOISE"
        if getattr(args, "threads", 0):
            scene.render.threads_mode = "FIXED"
            scene.render.threads = int(args.threads)
        setup_cycles_device(args.device)
    else:
        names = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        eevee = [n for n in names if "EEVEE" in n]
        if not eevee:
            raise RuntimeError(f"no EEVEE engine in this Blender build: {names}")
        scene.render.engine = sorted(eevee)[-1]
        print(f"[generate] using engine {scene.render.engine}")


def build_scene(buildings, ground_rgb, args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    col = bpy.data.collections.new("shadowclock")
    scene.collection.children.link(col)

    # geometry
    half = 400.0
    ground_mat = make_diffuse_material("ground", ground_rgb)
    add_mesh_object("ground", [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)],
                    [(0, 1, 2, 3)], ground_mat, col)
    n_bad = 0
    for k, b in enumerate(buildings):
        if "rings" in b:
            mesh = prism_mesh(b)
            if mesh is None:  # degenerate polygon: skipped in the render AND must be skipped in the height map by the caller
                n_bad += 1
                continue
            verts, faces = mesh
        else:
            verts, faces = box_mesh(b)
        add_mesh_object(f"building_{k}", verts, faces, make_diffuse_material(f"roof_{k}", b["color"]), col)
    if n_bad:
        print(f"[generate] WARNING: {n_bad} degenerate polygon(s) could not be triangulated and were left out of the render", flush=True)

    # world (uniform sky = ambient light that keeps shadows from being black)
    world = bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.85, 0.92, 1.0, 1.0)
    bg.inputs["Strength"].default_value = args.ambient

    # sun (small angular size => crisp but physically plausible penumbra)
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.angle = math.radians(0.53)
    sun_obj = bpy.data.objects.new("sun", sun_data)
    col.objects.link(sun_obj)

    # nadir orthographic camera, north up / east right
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = args.size * args.gsd
    cam_data.clip_start, cam_data.clip_end = 1.0, 3000.0
    cam_obj = bpy.data.objects.new("cam", cam_data)
    cam_obj.location = (0.0, 0.0, 1000.0)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)  # looks down -Z, image up = +Y, image right = +X
    col.objects.link(cam_obj)
    scene.camera = cam_obj

    # render settings
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
    pick_engine(scene, args.engine, args)
    return scene, sun_obj


def aim_sun(sun_obj, az_deg, el_deg, args):
    to_sun = Vector(sun_vector_enu(az_deg, el_deg))
    sun_obj.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()  # light travels along local -Z
    if args.physical_brightness:
        energy = args.horiz_irradiance                                  # brightness varies with cos(zenith)
    else:
        energy = args.horiz_irradiance / math.sin(math.radians(el_deg))  # horizontal brightness constant
    sun_obj.data.energy = energy


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def image_ok(path):
    """A render counts only if the PNG exists and is not empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def render_checked(scene, sun_obj, az, el, img_abs, args):
    """Render one frame and VERIFY that the file exists; retry on failure. Returns True on success."""
    for attempt in range(1 + args.render_retries):
        aim_sun(sun_obj, az, el, args)
        scene.render.filepath = img_abs.replace("\\", "/")
        status = None
        try:
            status = bpy.ops.render.render(write_still=True)
        except Exception as exc:  # noqa: BLE001 - GPU out-of-memory, driver reset, ...
            print(f"[generate] WARNING render raised {type(exc).__name__}: {exc}", flush=True)
            if "out of memory" in str(exc).lower() and attempt == 0:
                print("[generate] HINT: out of memory. Close other programs, then try --threads 2, or --device gpu, "
                      "or drop --denoise and raise --samples (e.g. 96).", flush=True)
            time.sleep(2.0)  # give the OS a moment to release memory before retrying
        if image_ok(img_abs):
            return True
        print(f"[generate] WARNING no image written (status={status}) for {os.path.basename(img_abs)}, "
              f"attempt {attempt + 1}/{1 + args.render_retries}", flush=True)
    return False


def time_grid(lat, lon, date_str, step_min, min_el):
    y, m, d = (int(x) for x in date_str.split("-"))
    minutes = np.arange(0, 24 * 60, step_min)
    az, el = solar_position(lat, lon, y, m, d, minutes / 60.0)
    keep = el >= min_el
    return [(int(mm), float(a), float(e)) for mm, a, e, k in zip(minutes, az, el, keep) if k]


def main():
    args = parse_args()
    args.out = os.path.abspath(args.out)  # Blender resolves relative render paths unpredictably (seen on Windows)
    print(f"[generate] output directory: {args.out}", flush=True)
    lats = [float(x) for x in args.lats.split(",") if x.strip()]
    dates = [x.strip() for x in args.dates.split(",") if x.strip()]
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "scenes"), exist_ok=True)
    if not args.no_shadow_gt:
        os.makedirs(os.path.join(args.out, "shadow_gt"), exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.jsonl")
    all_rows = []
    if os.path.exists(meta_path):  # resume: keep rows of scenes outside the requested range
        lo, hi = args.scene_start, args.scene_start + args.n_scenes
        with open(meta_path) as f:
            for line in f:
                row = json.loads(line)
                if not (lo <= row["scene"] < hi):
                    all_rows.append(row)

    t_start = time.time()
    n_rendered = 0
    consecutive_failures = 0
    failed = []
    osm_files = sorted(glob.glob(os.path.join(args.osm_scenes, "*.json"))) if args.osm_scenes else None
    if osm_files is not None:
        if not osm_files:
            sys.exit(f"no *.json scene files in {args.osm_scenes}")
        args.n_scenes = min(args.n_scenes, len(osm_files) - args.scene_start)
        print(f"[generate] real-layout mode: {len(osm_files)} scene files, rendering {args.n_scenes}", flush=True)
    for s in range(args.scene_start, args.scene_start + args.n_scenes):
        rng = np.random.default_rng(args.seed + s)
        if osm_files is not None:
            header, buildings = load_osm_scene(osm_files[s], rng)
            n_before = len(buildings)
            buildings = [b for b in buildings if prism_mesh(b) is not None]  # same set for the render and the ground truth
            if len(buildings) < n_before:
                print(f"[generate] scene {s}: dropped {n_before - len(buildings)} building(s) that could not be triangulated", flush=True)
            H = rasterize_height_polys(buildings, args.size, args.gsd)
            u = float(rng.uniform(-0.02, 0.02))
            ground_rgb = (0.14 + u, 0.14 + u, 0.15 + u) if args.ground == "urban" else (0.10 + u, 0.30 + u, 0.06 + u)
            lat_lon = [(float(header["lat0"]), float(header["lon0"]))]
            scene_doc = dict(scene=s, mode="osm", name=header.get("name"), source_file=os.path.basename(osm_files[s]),
                             lat0=header["lat0"], lon0=header["lon0"], source=header.get("source"), gsd=args.gsd,
                             size=args.size, ground_rgb=ground_rgb,
                             buildings=[dict(id=b["id"], height=b["height"], rings=b["rings"], color=b["color"]) for b in buildings])
        else:
            buildings = make_layout(rng, args.size, args.gsd)
            ground_rgb = tuple(float(x) for x in (0.10 + rng.uniform(-0.02, 0.02),
                                                  0.30 + rng.uniform(-0.04, 0.04),
                                                  0.06 + rng.uniform(-0.02, 0.02)))
            H = rasterize_height(buildings, args.size, args.gsd)
            lat_lon = [(lat, args.lon) for lat in lats]
            scene_doc = dict(scene=s, gsd=args.gsd, size=args.size, ground_rgb=ground_rgb, buildings=buildings)
        np.save(os.path.join(args.out, "scenes", f"scene_{s:03d}_height.npy"), H)
        with open(os.path.join(args.out, "scenes", f"scene_{s:03d}.json"), "w") as f:
            json.dump(scene_doc, f)

        scene, sun_obj = build_scene(buildings, ground_rgb, args)
        scene_rows = []
        for lat, lon in lat_lon:
            for date_str in dates:
                for minute, az, el in time_grid(lat, lon, date_str, args.step_min, args.min_el):
                    name = f"s{s:03d}_lat{int(round(lat)):+03d}_{date_str}_{minute // 60:02d}{minute % 60:02d}"
                    img_rel = "images/" + name + ".png"  # always forward slashes: portable Windows <-> Linux
                    img_abs = os.path.join(args.out, "images", name + ".png")
                    sh_rel = ("shadow_gt/" + name + ".npz") if not args.no_shadow_gt else None
                    if args.overwrite or not image_ok(img_abs):
                        if render_checked(scene, sun_obj, az, el, img_abs, args):
                            n_rendered += 1
                            consecutive_failures = 0
                        else:
                            failed.append(img_rel)
                            consecutive_failures += 1
                            if consecutive_failures >= 3:
                                print("[generate] ABORT: 3 renders in a row produced no file. Expected location: "
                                      f"{img_abs}\n[generate] Look at the 'Saved:' line Blender printed just above - "
                                      "it shows where the file really went (or the error). Stopping early.", flush=True)
                                sys.exit(3)
                            continue  # never list an image that does not exist
                    if sh_rel:
                        sh_abs = os.path.join(args.out, "shadow_gt", name + ".npz")
                        if args.overwrite or not os.path.exists(sh_abs):
                            np.savez_compressed(sh_abs, shadow=analytic_shadow_mask(H, args.gsd, az, el))
                    scene_rows.append(dict(
                        id=name, scene=s, image=img_rel, shadow_gt=sh_rel, lat=lat, lon=lon,
                        date=date_str, utc_minute=minute, utc_hour=minute / 60.0,
                        sun_az_deg=az, sun_el_deg=el, gsd_m=args.gsd, size_px=args.size))
        all_rows.extend(scene_rows)
        all_rows.sort(key=lambda r: (r["scene"], r["lat"], r["date"], r["utc_minute"]))
        with open(meta_path, "w") as f:  # rewritten after every scene so an interrupted run loses nothing
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[generate] scene {s}: {len(scene_rows)} images listed "
              f"({n_rendered} rendered this run, {len(failed)} failed, {time.time() - t_start:.0f}s elapsed)", flush=True)
    print(f"[generate] done. metadata: {meta_path} ({len(all_rows)} rows)")
    if failed:
        print(f"[generate] ERROR: {len(failed)} image(s) could not be rendered and are NOT in metadata.jsonl:")
        for f_ in failed[:20]:
            print("   ", f_)
        print("[generate] Re-run the SAME command to retry only the missing images "
              "(try --device cpu if you used --device gpu), and send the WARNING lines above if it keeps failing.")
        sys.exit(2)


if __name__ == "__main__":
    main()
