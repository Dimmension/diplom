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


def collect_meshes(root):
    meshes = []
    if root.type == 'MESH':
        meshes.append(root)
    for child in root.children_recursive:
        if child.type == 'MESH':
            meshes.append(child)
    return meshes


def setup_cycles_gpu(require_gpu: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    cycles_addon = bpy.context.preferences.addons.get('cycles')
    if cycles_addon is None:
        raise RuntimeError('Cycles addon not found')

    prefs = cycles_addon.preferences
    backend_order = ('OPTIX', 'CUDA')
    selected_backend = None

    for backend in backend_order:
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:
            continue

        has_gpu = any(dev.type in {'CUDA', 'OPTIX'} and dev.name for dev in prefs.devices)
        if has_gpu:
            selected_backend = backend
            break

    if selected_backend is None:
        devices_dump = [(dev.name, dev.type, dev.use) for dev in prefs.devices]
        if require_gpu:
            raise RuntimeError(f'GPU is required but no GPU device found. Devices: {devices_dump!r}')

        print(f'GPU device not found, falling back to CPU rendering. Devices: {devices_dump!r}')
        scene.cycles.device = 'CPU'
        for dev in prefs.devices:
            dev.use = dev.type == 'CPU'
        return

    for dev in prefs.devices:
        dev.use = dev.type != 'CPU'

    scene.cycles.device = 'GPU'
    print(f'Selected backend: {selected_backend}')
    print('Devices:', [(dev.name, dev.type, dev.use) for dev in prefs.devices])


def set_camera_transform(cam_obj, camera_cfg):
    cam_obj.location = (
        camera_cfg['position']['x'],
        camera_cfg['position']['y'],
        camera_cfg['position']['z'],
    )
    target = Vector((camera_cfg['target']['x'], camera_cfg['target']['y'], camera_cfg['target']['z']))
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam_obj.data.angle = math.radians(camera_cfg['fov_degrees'])


def setup_camera(camera_cfg):
    cam_data = bpy.data.cameras.new(name='RenderCamera')
    cam_obj = bpy.data.objects.new('RenderCamera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    set_camera_transform(cam_obj, camera_cfg)
    return cam_obj


def setup_cycles_quality(scene, payload: dict) -> None:
    cycles = scene.cycles
    cycles.samples = max(1, int(payload.get('samples', 64)))
    cycles.max_bounces = max(0, int(payload.get('max_bounces', 4)))

    use_adaptive_sampling = bool(payload.get('use_adaptive_sampling', True))
    cycles.use_adaptive_sampling = use_adaptive_sampling
    if use_adaptive_sampling:
        cycles.adaptive_threshold = max(0.0, float(payload.get('adaptive_threshold', 0.03)))

    use_denoising = bool(payload.get('use_denoising', True))
    denoiser = str(payload.get('denoiser', 'OPTIX')).upper()
    for view_layer in scene.view_layers:
        view_layer.cycles.use_denoising = use_denoising
        if use_denoising:
            try:
                view_layer.cycles.denoiser = denoiser
            except Exception:
                pass


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


def _assign_single_material(obj, material):
    slots_count = len(obj.material_slots)
    if slots_count == 0:
        obj.data.materials.append(material)
        return
    for idx in range(slots_count):
        obj.material_slots[idx].material = material


def render_object_mask(
    *,
    scene,
    object_meshes,
    environment_meshes,
    output_path: str,
) -> None:
    white_mat = bpy.data.materials.new(name='MaskWhite')
    white_mat.use_nodes = True
    white_nodes = white_mat.node_tree.nodes
    white_links = white_mat.node_tree.links
    white_nodes.clear()
    white_emission = white_nodes.new(type='ShaderNodeEmission')
    white_emission.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    white_emission.inputs['Strength'].default_value = 1.0
    white_output = white_nodes.new(type='ShaderNodeOutputMaterial')
    white_links.new(white_emission.outputs['Emission'], white_output.inputs['Surface'])

    black_mat = bpy.data.materials.new(name='MaskBlack')
    black_mat.use_nodes = True
    black_nodes = black_mat.node_tree.nodes
    black_links = black_mat.node_tree.links
    black_nodes.clear()
    black_emission = black_nodes.new(type='ShaderNodeEmission')
    black_emission.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    black_emission.inputs['Strength'].default_value = 1.0
    black_output = black_nodes.new(type='ShaderNodeOutputMaterial')
    black_links.new(black_emission.outputs['Emission'], black_output.inputs['Surface'])

    all_meshes = list(dict.fromkeys([*object_meshes, *environment_meshes]))
    original_materials = {obj.name: [slot.material for slot in obj.material_slots] for obj in all_meshes}

    render_file_format = scene.render.image_settings.file_format
    render_color_mode = scene.render.image_settings.color_mode
    film_transparent = scene.render.film_transparent
    view_transform = scene.view_settings.view_transform
    exposure = scene.view_settings.exposure
    gamma = scene.view_settings.gamma
    cycles_samples = scene.cycles.samples
    cycles_use_adaptive = scene.cycles.use_adaptive_sampling
    render_filepath = scene.render.filepath

    try:
        for obj in object_meshes:
            _assign_single_material(obj, white_mat)
        for obj in environment_meshes:
            _assign_single_material(obj, black_mat)

        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.film_transparent = True
        scene.view_settings.view_transform = 'Raw'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.cycles.samples = 1
        scene.cycles.use_adaptive_sampling = False
        scene.render.filepath = output_path

        bpy.ops.render.render(write_still=True)
    finally:
        for obj in all_meshes:
            slots_count = len(obj.material_slots)
            saved = original_materials.get(obj.name, [])
            if slots_count == 0 and saved:
                for mat in saved:
                    obj.data.materials.append(mat)
                continue
            for idx in range(slots_count):
                obj.material_slots[idx].material = saved[idx] if idx < len(saved) else None

        scene.render.image_settings.file_format = render_file_format
        scene.render.image_settings.color_mode = render_color_mode
        scene.render.film_transparent = film_transparent
        scene.view_settings.view_transform = view_transform
        scene.view_settings.exposure = exposure
        scene.view_settings.gamma = gamma
        scene.cycles.samples = cycles_samples
        scene.cycles.use_adaptive_sampling = cycles_use_adaptive
        scene.render.filepath = render_filepath

        bpy.data.materials.remove(white_mat, do_unlink=True)
        bpy.data.materials.remove(black_mat, do_unlink=True)


def build_bbox_from_mask(
    *,
    source_path: str,
    mask_path: str,
    output_path: str,
) -> dict[str, int | bool]:
    mask_image = bpy.data.images.load(mask_path, check_existing=False)
    source_image = bpy.data.images.load(source_path, check_existing=False)

    width = int(mask_image.size[0])
    height = int(mask_image.size[1])
    if width <= 0 or height <= 0:
        raise RuntimeError('Mask image has invalid size')

    mask_pixels = list(mask_image.pixels[:])
    bbox_pixels = list(source_image.pixels[:])

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for pixel_index in range(width * height):
        offset = pixel_index * 4
        if mask_pixels[offset] > 0.2:
            x = pixel_index % width
            y = pixel_index // width
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    visible = max_x >= min_x and max_y >= min_y
    if visible:
        thickness = max(2, width // 512)

        def paint_pixel(px: int, py: int) -> None:
            if px < 0 or px >= width or py < 0 or py >= height:
                return
            idx = (py * width + px) * 4
            bbox_pixels[idx] = 1.0
            bbox_pixels[idx + 1] = 0.0
            bbox_pixels[idx + 2] = 0.0
            bbox_pixels[idx + 3] = 1.0

        for line_offset in range(thickness):
            top_y = min(max_y + line_offset, height - 1)
            bottom_y = max(min_y - line_offset, 0)
            left_x = max(min_x - line_offset, 0)
            right_x = min(max_x + line_offset, width - 1)

            for x in range(left_x, right_x + 1):
                paint_pixel(x, top_y)
                paint_pixel(x, bottom_y)

            for y in range(min_y, max_y + 1):
                paint_pixel(left_x, y)
                paint_pixel(right_x, y)

    bbox_image = bpy.data.images.new(
        name=f'bbox_{os.path.basename(output_path)}',
        width=width,
        height=height,
        alpha=True,
        float_buffer=False,
    )
    bbox_image.pixels = bbox_pixels
    bbox_image.file_format = 'PNG'
    bbox_image.filepath_raw = output_path
    bbox_image.save()

    bpy.data.images.remove(mask_image, do_unlink=True)
    bpy.data.images.remove(source_image, do_unlink=True)
    bpy.data.images.remove(bbox_image, do_unlink=True)
    return {
        'visible': visible,
        'xmin': int(min_x if visible else 0),
        'ymin': int(min_y if visible else 0),
        'xmax': int(max_x if visible else 0),
        'ymax': int(max_y if visible else 0),
        'width': width,
        'height': height,
    }


def render_single_sample(
    *,
    scene,
    object_root,
    env_root,
    object_meshes,
    environment_meshes,
    camera_obj,
    scene_config: dict,
    output_path: str,
    mask_output_path: str | None,
    bbox_output_path: str | None,
    bbox_meta_output_path: str | None,
) -> None:
    apply_transform(object_root, scene_config['object_transform'])
    apply_transform(env_root, scene_config['environment_transform'])
    set_camera_transform(camera_obj, scene_config['camera'])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    bbox_meta: dict[str, int | bool] | None = None
    if mask_output_path:
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        render_object_mask(
            scene=scene,
            object_meshes=object_meshes,
            environment_meshes=environment_meshes,
            output_path=mask_output_path,
        )
    if mask_output_path and bbox_output_path:
        os.makedirs(os.path.dirname(bbox_output_path), exist_ok=True)
        bbox_meta = build_bbox_from_mask(
            source_path=output_path,
            mask_path=mask_output_path,
            output_path=bbox_output_path,
        )
    if bbox_meta_output_path and bbox_meta is not None:
        os.makedirs(os.path.dirname(bbox_meta_output_path), exist_ok=True)
        with open(bbox_meta_output_path, 'w', encoding='utf-8') as fh:
            json.dump(bbox_meta, fh)


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

    setup_cycles_gpu(bool(payload.get('require_gpu', False)))

    object_root = import_asset(payload['object_path'])
    env_root = import_asset(payload['environment_path'])

    dataset_samples = payload.get('dataset_samples')
    if dataset_samples:
        initial_scene_config = dataset_samples[0]['scene_config']
    else:
        initial_scene_config = payload['scene_config']

    apply_transform(object_root, initial_scene_config['object_transform'])
    apply_transform(env_root, initial_scene_config['environment_transform'])
    camera_obj = setup_camera(initial_scene_config['camera'])
    skybox_path = payload.get('skybox_path')
    if skybox_path:
        setup_skybox(skybox_path)

    object_meshes = collect_meshes(object_root)
    environment_meshes = collect_meshes(env_root)

    scene = bpy.context.scene
    scene.render.resolution_x = int(payload['width'])
    scene.render.resolution_y = int(payload['height'])
    scene.render.image_settings.file_format = 'PNG'
    setup_cycles_quality(scene, payload)

    if dataset_samples:
        for sample in dataset_samples:
            render_single_sample(
                scene=scene,
                object_root=object_root,
                env_root=env_root,
                object_meshes=object_meshes,
                environment_meshes=environment_meshes,
                camera_obj=camera_obj,
                scene_config=sample['scene_config'],
                output_path=sample['output_path'],
                mask_output_path=sample.get('mask_output_path'),
                bbox_output_path=sample.get('bbox_output_path'),
                bbox_meta_output_path=sample.get('bbox_meta_output_path'),
            )
        return

    render_single_sample(
        scene=scene,
        object_root=object_root,
        env_root=env_root,
        object_meshes=object_meshes,
        environment_meshes=environment_meshes,
        camera_obj=camera_obj,
        scene_config=payload['scene_config'],
        output_path=payload['output_path'],
        mask_output_path=payload.get('mask_output_path'),
        bbox_output_path=payload.get('bbox_output_path'),
        bbox_meta_output_path=payload.get('bbox_meta_output_path'),
    )


if __name__ == '__main__':
    main()
