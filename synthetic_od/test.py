import omni.replicator.core as rep

OUTPUT_DIR = "/_output/test_run"

with rep.new_layer():
    cube = rep.create.cube(
        position=(0, 0, 0),
        semantics=[("class", "cube")]
    )

    camera = rep.create.camera(
        position=(0, 0, 500),
        look_at=(0, 0, 0)
    )

    render_product = rep.create.render_product(camera, (512, 512))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True
    )
    writer.attach([render_product])

    with rep.trigger.on_frame(num_frames=5):
        rep.modify.pose(
            cube,
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
        )

    rep.orchestrator.run()

print(f"Done. Output written to: {OUTPUT_DIR}")
