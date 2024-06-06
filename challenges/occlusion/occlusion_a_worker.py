# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# type: ignore
"""
Worker file for the Multi-Object Video (MOVi) C (and CC) datasets.
  * The number of objects is randomly chosen between
    --min_num_objects (3) and --max_num_objects (10)
  * The objects are randomly chosen from the Google Scanned Objects dataset

  * Background is an random HDRI from the HDRI Haven dataset,
    projected onto a Dome (half-sphere).
    The HDRI is also used for lighting the scene.
"""

import os
import logging

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import shutil
import tqdm


# --- Some configuration values
# the region in which to place objects [(min), (max)]
SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-4.0, -4.0, 0.0), (4.0, 4.0, 0.0)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"], default="train")
# Configuration for the objects of the scene
parser.add_argument(
    "--min_num_objects", type=int, default=3, help="minimum number of objects"
)
parser.add_argument(
    "--max_num_objects", type=int, default=3, help="maximum number of objects"
)
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train")

parser.add_argument(
    "--camera", choices=["fixed_random", "linear_movement"], default="fixed_random"
)
parser.add_argument("--max_camera_movement", type=float, default=4.0)

# Configuration for the source of the assets
parser.add_argument(
    "--kubasic_assets",
    type=str,
    default="gs://kubric-public/assets/KuBasic/KuBasic.json",
)
parser.add_argument(
    "--hdri_assets",
    type=str,
    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json",
)
parser.add_argument(
    "--gso_assets", type=str, default="gs://kubric-public/assets/GSO/GSO.json"
)
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12, resolution=256)

parser.add_argument("--num_scenes", type=int, default=1)
FLAGS = parser.parse_args()


def process(flags):
    # --- Common setups & resources
    scene, rng, output_dir, scratch_dir = kb.setup(flags)
    print("output_dir", output_dir)
    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(
        scene, scratch_dir, samples_per_pixel=64, background_transparency=True
    )
    kubasic = kb.AssetSource.from_manifest(flags.kubasic_assets)
    gso = kb.AssetSource.from_manifest(flags.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(flags.hdri_assets)

    # --- Populate the scene
    # background HDRI
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    if flags.backgrounds_split == "train":
        logging.info(
            "Choosing one of the %d training backgrounds...", len(train_backgrounds)
        )
        hdri_id = rng.choice(train_backgrounds)
    else:
        logging.info(
            "Choosing one of the %d held-out backgrounds...", len(test_backgrounds)
        )
        hdri_id = rng.choice(test_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    # assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    # scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(
        asset_id="dome",
        name="dome",
        friction=flags.floor_friction,
        restitution=flags.floor_restitution,
        static=True,
        background=True,
    )
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    def get_linear_camera_motion_start_end(
        movement_speed: float,
        inner_radius: float = 8.0,
        outer_radius: float = 12.0,
        z_offset: float = 0.1,
    ):
        """Sample a linear path which starts and ends within a half-sphere shell."""
        while True:
            camera_start = np.array(
                kb.sample_point_in_half_sphere_shell(
                    inner_radius, outer_radius, z_offset
                )
            )
            direction = rng.rand(3) - 0.5
            movement = direction / np.linalg.norm(direction) * movement_speed
            camera_end = camera_start + movement
            if (
                inner_radius <= np.linalg.norm(camera_end) <= outer_radius
                and camera_end[2] > z_offset
            ):
                return camera_start, camera_end

    # Camera
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=32)
    if flags.camera == "fixed_random":
        scene.camera.position = kb.sample_point_in_half_sphere_shell(
            inner_radius=7.0, outer_radius=9.0, offset=0.1
        )
        scene.camera.look_at((0, 0, 0))
    elif flags.camera == "linear_movement":
        camera_start, camera_end = get_linear_camera_motion_start_end(
            movement_speed=rng.uniform(low=0.0, high=flags.max_camera_movement)
        )
        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames
        for frame in range(flags.frame_start - 1, flags.frame_end + 2):
            interp = (frame - flags.frame_start + 1) / (
                flags.frame_end - flags.frame_start + 3
            )
            scene.camera.position = interp * np.array(camera_start) + (
                1 - interp
            ) * np.array(camera_end)
            scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)

    # Add random objects
    train_split, test_split = gso.get_test_split(fraction=0.1)
    if flags.objects_split == "train":
        logging.info("Choosing one of the %d training objects...", len(train_split))
        active_split = train_split
    else:
        logging.info("Choosing one of the %d held-out objects...", len(test_split))
        active_split = test_split

    num_objects = rng.randint(flags.min_num_objects, flags.max_num_objects + 1)
    logging.info("Randomly placing %d objects:", num_objects)
    for i in range(num_objects):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale"] = scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=SPAWN_REGION, rng=rng)
        # initialize velocity randomly but biased towards center
        obj.velocity = rng.uniform(*VELOCITY_RANGE) - [
            obj.position[0],
            obj.position[1],
            0,
        ]
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    if flags.save_state:
        logging.info(
            "Saving the simulator state to '%s' prior to the simulation.",
            output_dir / "scene.bullet",
        )
        simulator.save_state(output_dir / "scene.bullet")

    # Run dynamic objects simulation
    logging.info("Running the simulation ...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)

    # --- Rendering
    if flags.save_state:
        logging.info("Saving the renderer state to '%s' ", output_dir / "scene.blend")
        renderer.save_state(output_dir / "scene.blend")

    # Store the position of every asset
    frame_number = scene.frame_end // 2
    for i in range(len(scene.foreground_assets)):
        logging.info(
            f"{scene.foreground_assets[i].uid} -> {scene.foreground_assets[i].asset_id}"
        )
    pos_at_frame = [
        {
            scene.foreground_assets[i]
            .asset_id: scene.foreground_assets[i]
            .get_value_at("position", t)
            for i in range(num_objects)
        }
        for t in range(0, scene.frame_end + 1)
    ]
    scale_at_frame = [
        {
            scene.foreground_assets[i]
            .asset_id: scene.foreground_assets[i]
            .get_value_at("scale", t)
            for i in range(num_objects)
        }
        for t in range(0, scene.frame_end + 1)
    ]

    ds_offset = 2
    data_stacks = [None] * (num_objects + ds_offset)
    # Regular render pass
    data_stacks[0] = renderer.render(frames=[frame_number])
    kb.compute_visibility(data_stacks[0]["segmentation"], scene.assets)
    visible_foreground_assets = [
        asset
        for asset in scene.foreground_assets
        if np.max(asset.metadata["visibility"]) > 0
    ]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True,
    )
    data_stacks[0]["segmentation"] = kb.adjust_segmentation_idxs(
        data_stacks[0]["segmentation"], scene.assets, visible_foreground_assets
    )
    scene.metadata["num_instances"] = len(visible_foreground_assets)
    # Background-less render pass
    scene.remove(dome)
    data_stacks[1] = renderer.render(frames=[frame_number])
    # N.B. visible_foreground_assets are unchanged from the regular render pass
    data_stacks[1]["segmentation"] = kb.adjust_segmentation_idxs(
        data_stacks[1]["segmentation"], scene.assets, visible_foreground_assets
    )

    visibility_in_scene = {
        x.asset_id: x.metadata["visibility"] for x in scene.foreground_assets
    }
    visibility_in_isolation = {}

    def search(xs, key, _default=None):
        return next(filter(lambda x: x.asset_id == key, xs), _default)

    logging.info("Rendering the scene ...")
    for i in range(num_objects):
        data_stack = data_stacks[i + ds_offset]
        logging.info(f"{i} object rendering individually")
        for j in range(num_objects):
            # Retrieve asset_id
            k_id = list(pos_at_frame[frame_number].keys())[j]
            k = search(scene.foreground_assets, k_id)
            assert k is not None, "No matching asset id found"

            asset_pos = pos_at_frame[frame_number][k.asset_id]
            asset_scale = scale_at_frame[frame_number][k.asset_id]
            # Move items behind the camera
            scene.foreground_assets[j].position = (
                np.full([3], fill_value=scene.camera.position * 2)
                if j != i
                else asset_pos
            )
            k.keyframe_insert("position", frame_number)
            # and make them zero size
            scene.foreground_assets[j].scale = np.full(
                [3], 0.0 if j != i else asset_scale
            )
            k.keyframe_insert("scale", frame_number)

        # scene.metadata["background"] = None
        # renderer._set_background_color()
        # renderer._set_ambient_light_color()

        # renderer.background_transparency = True

        data_stack = renderer.render(frames=[frame_number])

        # --- Postprocessing
        kb.compute_visibility(data_stack["segmentation"], scene.assets)
        visibility_in_isolation.update(
            {
                x.asset_id: x.metadata["visibility"]
                for x in scene.foreground_assets
                if np.max(x.metadata["visibility"]) > 0
            }
        )
        visible_foreground_assets = [
            asset
            for asset in scene.foreground_assets
            if np.max(asset.metadata["visibility"]) > 0
        ]
        assert len(visible_foreground_assets) <= 1
        if len(visible_foreground_assets) == 0:
            logging.warning(
                f"Object (#{i} `{list(pos_at_frame[frame_number].keys())[i]}`) rendered in isolation is not visible!"
            )
        visible_foreground_assets = sorted(  # sort assets by their visibility
            visible_foreground_assets,
            key=lambda asset: np.sum(asset.metadata["visibility"]),
            reverse=True,
        )

        data_stack["segmentation"] = kb.adjust_segmentation_idxs(
            data_stack["segmentation"], scene.assets, visible_foreground_assets
        )
        data_stacks[i + ds_offset] = data_stack
        scene.metadata["num_instances"] = len(visible_foreground_assets)

        # Save to image files
        # kb.write_image_dict(data_stack, output_dir / f"_obj{i}")
        kb.post_processing.compute_bboxes(
            data_stack["segmentation"], visible_foreground_assets
        )

    # Combine all data stacks together into one stack
    ds = {k: None for k in data_stacks[0].keys()}
    for k in ds.keys():
        ds[k] = np.concatenate([x[k] for x in data_stacks], axis=0)
    # Write to file
    kb.write_image_dict(ds, output_dir)

    logging.info(visibility_in_scene)
    visibility = {
        x: (
            np.max(visibility_in_scene.get(x, 0)),
            np.max(visibility_in_isolation.get(x, 0)),
        )
        for x in (set(visibility_in_scene.keys()) | set(visibility_in_isolation.keys()))
    }
    ratio_vis_indiv_scene = dict(
        map(lambda x: (x[0], x[1][0] / x[1][1]), visibility.items())
    )
    logging.info(f"Visibility {ratio_vis_indiv_scene}")
    kb.write_json(
        filename=output_dir / "visibility.json",
        data=ratio_vis_indiv_scene,
    )

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(
        filename=output_dir / "metadata.json",
        data={
            "flags": vars(flags),
            "metadata": kb.get_scene_metadata(scene),
            "camera": kb.get_camera_info(scene.camera),
            "instances": kb.get_instance_info(scene, visible_foreground_assets),
        },
    )
    kb.write_json(
        filename=output_dir / "events.json",
        data={
            "collisions": kb.process_collisions(
                collisions, scene, assets_subset=visible_foreground_assets
            ),
        },
    )

    kb.done()
    return ratio_vis_indiv_scene


num_scenes = 0
attempt = 0
#pbar = tqdm.tqdm(total=FLAGS.num_scenes, disable=True)

#from tqdm.contrib.logging import logging_redirect_tqdm

while num_scenes < FLAGS.num_scenes:
    attempt += 1
    logging.info(f"Scene {num_scenes}. Total attempts {attempt}")

    path = f"output/scene{num_scenes}"
    if not os.path.exists(path):
        os.makedirs(path)
    flags = FLAGS
    flags.job_dir = path

    ratio_vis_indiv_scene = process(flags)
    if any(0.25 <= x <= 0.75 for x in ratio_vis_indiv_scene.values()):
        num_scenes += 1
        #pbar.update(1)
    else:
        logging.warning(
            "Discarding scene due to lack of partially occluded visible objects."
        )
        # Delete the folder path

        shutil.rmtree(path)
        continue
