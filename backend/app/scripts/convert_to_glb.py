import os
import sys

import bpy


def import_asset(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.obj':
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in {'.glb', '.gltf'}:
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise RuntimeError(f'Unsupported import format: {ext}')


def main() -> None:
    argv = sys.argv
    if '--' not in argv:
        raise RuntimeError('Expected -- <input_path> <output_glb_path>')

    args = argv[argv.index('--') + 1 :]
    if len(args) != 2:
        raise RuntimeError('Expected exactly 2 arguments: input_path output_glb_path')

    input_path, output_path = args

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    import_asset(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')


if __name__ == '__main__':
    main()
