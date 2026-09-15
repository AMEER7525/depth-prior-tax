"""Render exact depth for NeRF-Synthetic poses from the original .blend scenes.

Runs INSIDE Blender (tested with 5.2), not in the project environment:

    blender -b <blend_dir>/lego.blend -P scripts/render_blender_depth.py -- \
        --transforms $DATA_ROOT/nerf_synthetic/lego/transforms_test.json \
        --ids 0 8 16 --out exact_depth/lego/test

The NeRF release ships no metric depth, but the modified scenes it was
rendered from are available: the NeRF authors shared them in bmild/nerf issue
#59 (that link is dead) and a community copy with repaired texture paths,
`blend_files_fixed.zip`, is linked from issue #198. The study's reference depth
is a 400-view reconstruction (scripts/make_gt_depth.py); this script produces
the depth the scene geometry itself defines, so that reconstruction's error
can be measured directly (scripts/validate_reference_depth.py).

For every requested frame of a transforms_*.json it places a fresh camera at
the frame's `transform_matrix` (Blender's own camera matrix_world, so no
convention change) with the release's horizontal field of view, and renders
one Cycles sample through the pixel centre. The Z pass is planar depth along
the camera's forward axis -- the convention of every depth map in this repo --
and a one-sample render with a near-zero pixel filter reads it exactly at the
pixel centre instead of at a jittered sub-pixel position.

Output per frame: <out>/r_<i>.npz with
    depth  (H, W) float32, 0 where no surface was hit
    alpha  (H, W) float32, the one-sample coverage (0 or 1)
plus <out>/render_meta.json. Before any scene frame, a synthetic plane is
rendered and checked, so a Blender version whose Z pass is radial distance, or
whose image rows come back flipped, fails loudly instead of producing depth
that is subtly wrong.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix

NO_HIT = 1e9          # Cycles writes ~1e10 where a ray escapes


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=800)
    return ap.parse_args(argv)


def configure(scene, resolution):
    """One centred sample per pixel, depth pass on, nothing that blurs depth."""
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = True
    scene.render.use_motion_blur = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    cy = scene.cycles
    cy.samples = 1
    cy.use_adaptive_sampling = False
    cy.use_denoising = False
    cy.filter_width = 0.01
    cy.max_bounces = 0                 # depth and coverage need primary rays only
    for vl in scene.view_layers:
        vl.use_pass_z = True
    fmt = scene.render.image_settings
    if hasattr(fmt, "media_type"):         # Blender >= 5.0 gates formats by media
        fmt.media_type = "MULTI_LAYER_IMAGE"
    fmt.file_format = "OPEN_EXR_MULTILAYER"
    fmt.color_depth = "32"


def make_camera(scene, angle_x):
    data = bpy.data.cameras.new("depth_cam")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.angle_x = angle_x
    data.shift_x = data.shift_y = 0.0
    data.clip_start, data.clip_end = 1e-3, 1e4
    data.dof.use_dof = False
    cam = bpy.data.objects.new("depth_cam", data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def render_frame(scene, cam, c2w, exr_path):
    """Render one pose; return (depth, alpha) with row 0 at the image top."""
    cam.matrix_world = Matrix(c2w)
    bpy.context.view_layer.update()
    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(str(exr_path), scene=scene)

    inp = oiio.ImageInput.open(str(exr_path))
    spec = inp.spec()
    pixels = inp.read_image(0, 0, 0, spec.nchannels, "float")
    inp.close()
    names = list(spec.channelnames)

    def channel(suffix):
        hits = [i for i, n in enumerate(names) if n.endswith(suffix)]
        if not hits:
            raise RuntimeError(f"no channel ending in {suffix!r} in {names}")
        return np.asarray(pixels)[..., hits[0]].astype(np.float64)

    z = channel("Depth.Z")
    alpha = channel("Combined.A")
    depth = np.where((z > 0) & (z < NO_HIT), z, 0.0)
    exr_path.unlink()
    return depth.astype(np.float32), alpha.astype(np.float32)


def self_test(scene, resolution, angle_x, tmp_dir):
    """A plane 3 units ahead covering the top half: planar depth, upright rows.

    Radial distance would grow towards the corners; a flipped image would put
    the plane in the bottom half. It runs in the loaded scene with that
    scene's objects hidden: a scene created from Python renders without its
    depth pass in Blender 5.2, which would test nothing.
    """
    hidden = {o.name: o.hide_render for o in scene.objects}
    for o in scene.objects:
        o.hide_render = True
    mesh = bpy.data.meshes.new("depth_self_test")
    # Camera at the origin looks down -z with +y up: this covers the upper half.
    mesh.from_pydata([(-50, 0, -3), (50, 0, -3), (50, 50, -3), (-50, 50, -3)],
                     [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new("depth_self_test", mesh)
    scene.collection.objects.link(plane)
    cam = make_camera(scene, angle_x)
    try:
        depth, _ = render_frame(scene, cam, np.eye(4).tolist(),
                                tmp_dir / "self_test.exr")
    finally:
        bpy.data.objects.remove(plane)
        bpy.data.meshes.remove(mesh)
        bpy.data.objects.remove(cam)
        for o in scene.objects:
            o.hide_render = hidden[o.name]
    H = depth.shape[0]
    top, bottom = depth[: H // 4], depth[3 * H // 4:]
    ok = bool((top > 0).all() and (bottom == 0).all()
              and np.abs(top - 3.0).max() < 1e-4)
    result = {"planar": ok, "top_min": float(top.min()), "top_max": float(top.max()),
              "bottom_hits": int((bottom > 0).sum())}
    if not ok:
        raise SystemExit(f"depth self-test failed: {result}")
    return result


def main():
    args = parse_args()
    meta = json.loads(Path(args.transforms).read_text())
    angle_x = float(meta["camera_angle_x"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    alpha_threshold = scene.view_layers[0].pass_alpha_threshold
    configure(scene, args.resolution)
    check = self_test(scene, args.resolution, angle_x, out)
    cam = make_camera(scene, angle_x)

    for i in args.ids:
        frame = meta["frames"][i]
        depth, alpha = render_frame(scene, cam, frame["transform_matrix"],
                                    out / f"r_{i}.exr")
        np.savez_compressed(out / f"r_{i}.npz", depth=depth, alpha=alpha)
        print(f"  r_{i}: {(depth > 0).mean():.1%} of pixels hit, depth "
              f"{depth[depth > 0].min():.3f}..{depth[depth > 0].max():.3f}", flush=True)

    (out / "render_meta.json").write_text(json.dumps({
        "blend": bpy.path.basename(bpy.data.filepath),
        "blender": bpy.app.version_string,
        "transforms": Path(args.transforms).name,
        "ids": args.ids, "resolution": args.resolution, "camera_angle_x": angle_x,
        "frame": scene.frame_current, "pass_alpha_threshold": alpha_threshold,
        "self_test": check,
    }, indent=2))


if __name__ == "__main__":
    main()
