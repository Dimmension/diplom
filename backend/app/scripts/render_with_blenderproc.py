import blenderproc as bproc

import json
import math
import os
import sys

import bpy
from mathutils import Vector


def _extract_script_args(argv: list[str]) -> list[str]:
    if '--' in argv:
        return argv[argv.index('--') + 1 :]

    script_path = os.path.abspath(__file__)
    script_basename = os.path.basename(script_path)

    for index, token in enumerate(argv):
        if token == __file__:
            return argv[index + 1 :]
        if token.endswith('.py') and os.path.basename(token) == script_basename:
            return argv[index + 1 :]
        if os.path.basename(token) == script_basename:
            return argv[index + 1 :]

    if argv and os.path.basename(argv[0]) == script_basename:
        return argv[1:]

    # Last-resort fallback for wrappers that rewrite argv:
    # keep only positional args so config path can still be detected.
    return [token for token in argv[1:] if token and not token.startswith('-')]


def import_asset(path: str):
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()

    if ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in {'.glb', '.gltf'}:
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise RuntimeError(f'Unsupported import format: {ext}')

    added = [obj for obj in bpy.data.objects if obj not in before]
    if not added:
        raise RuntimeError(f'No objects imported from {path}')

    root = added[0]
    if len(added) > 1:
        empty = bpy.data.objects.new(name=f'group_{os.path.basename(path)}', object_data=None)
        bpy.context.scene.collection.objects.link(empty)
        for obj in added:
            obj.parent = empty
        root = empty
    return root


def apply_transform(obj, cfg):
    obj.location = (cfg['position']['x'], cfg['position']['y'], cfg['position']['z'])
    obj.rotation_euler = (
        math.radians(cfg['rotation']['x']),
        math.radians(cfg['rotation']['y']),
        math.radians(cfg['rotation']['z']),
    )
    obj.scale = (cfg['scale']['x'], cfg['scale']['y'], cfg['scale']['z'])


def configure_render_device(require_gpu: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    cycles_addon = bpy.context.preferences.addons.get('cycles')
    if cycles_addon is None:
        raise RuntimeError('Cycles addon not found')

    prefs = cycles_addon.preferences
    prefs.get_devices()

    gpu_devices = [dev for dev in prefs.devices if dev.type != 'CPU']
    if not gpu_devices:
        if require_gpu:
            raise RuntimeError('GPU is required but no GPU device found')
        print('GPU device not found, falling back to CPU rendering')
        scene.cycles.device = 'CPU'
        for dev in prefs.devices:
            dev.use = dev.type == 'CPU'
        return

    scene.cycles.device = 'GPU'
    for dev in prefs.devices:
        dev.use = dev.type != 'CPU'


def setup_camera(camera_cfg):
    cam_data = bpy.data.cameras.new(name='RenderCamera')
    cam_obj = bpy.data.objects.new('RenderCamera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location = (
        camera_cfg['position']['x'],
        camera_cfg['position']['y'],
        camera_cfg['position']['z'],
    )
    target = Vector((camera_cfg['target']['x'], camera_cfg['target']['y'], camera_cfg['target']['z']))
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam_data.angle = math.radians(camera_cfg['fov_degrees'])


def setup_skybox(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in {'.hdr', '.exr'}:
        raise RuntimeError(f'Unsupported skybox format: {ext}')

    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new('World')
        scene.world = world

    world.use_nodes = True
    node_tree = world.node_tree
    if node_tree is None:
        raise RuntimeError('World nodes are unavailable')

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    env_tex = nodes.new(type='ShaderNodeTexEnvironment')
    env_tex.image = bpy.data.images.load(path, check_existing=True)
    background = nodes.new(type='ShaderNodeBackground')
    world_output = nodes.new(type='ShaderNodeOutputWorld')

    links.new(env_tex.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], world_output.inputs['Surface'])


def main() -> None:
    args = _extract_script_args(sys.argv)
    if len(args) != 1:
        raise RuntimeError(f'Expected exactly 1 argument: config_path, got args={args}')

    config_path = args[0]
    with open(config_path, 'r', encoding='utf-8') as fh:
        payload = json.load(fh)

    bproc.init()

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    configure_render_device(bool(payload.get('require_gpu', False)))

    object_root = import_asset(payload['object_path'])
    env_root = import_asset(payload['environment_path'])

    scene_config = payload['scene_config']
    apply_transform(object_root, scene_config['object_transform'])
    apply_transform(env_root, scene_config['environment_transform'])
    setup_camera(scene_config['camera'])
    skybox_path = payload.get('skybox_path')
    if skybox_path:
        setup_skybox(skybox_path)

    scene = bpy.context.scene
    scene.render.resolution_x = int(payload['width'])
    scene.render.resolution_y = int(payload['height'])
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = payload['output_path']

    os.makedirs(os.path.dirname(payload['output_path']), exist_ok=True)
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
