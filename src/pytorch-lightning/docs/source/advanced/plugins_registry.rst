Training Type Plugins Registry
==============================

.. warning:: The Plugins Registry is experimental and subject to change.

Lightning includes a registry that holds information about Training Type plugins and allows for the registration of new custom plugins.

The Plugins are assigned strings that identify them, such as "ddp", "deepspeed_stage_2_offload", and so on.
It also returns the optional description and parameters for initialising the Plugin that were defined during registration.


.. code-block:: python

    # Training with the DDP Plugin with `find_unused_parameters` as False
    trainer = Trainer(strategy="ddp_find_unused_parameters_false", accelerator="gpu", devices=4)

    # Training with DeepSpeed ZeRO Stage 3 and CPU Offload
    trainer = Trainer(strategy="deepspeed_stage_3_offload", accelerator="gpu", devices=3)

    # Training with the TPU Spawn Plugin with `debug` as True
    trainer = Trainer(strategy="tpu_spawn_debug", accelerator="tpu", devices=8)


Additionally, you can pass your custom registered training type plugins to the ``strategy`` argument.

.. code-block:: python

    from pytorch_lightning.plugins import DDPPlugin, TrainingTypePluginsRegistry, CheckpointIO


    class CustomCheckpointIO(CheckpointIO):
        def save_checkpoint(self, checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
            ...

        def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
            ...


    custom_checkpoint_io = CustomCheckpointIO()

    # Register the DDP Plugin with your custom CheckpointIO plugin
    TrainingTypePluginsRegistry.register(
        "ddp_custom_checkpoint_io",
        DDPPlugin,
        description="DDP Plugin with custom checkpoint io plugin",
        checkpoint_io=custom_checkpoint_io,
    )

    trainer = Trainer(strategy="ddp_custom_checkpoint_io", accelerator="gpu", devices=2)
