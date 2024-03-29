import dacapo
import logging
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.tasks import AffinitiesTaskConfig
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    SimpleAugmentConfig,
    ElasticAugmentConfig,
    IntensityAugmentConfig,
)
from funlib.geometry import Coordinate
from dacapo.experiments.run_config import RunConfig
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_config_store
from dacapo.train import train

logging.basicConfig(level=logging.INFO)

# TODO: create datasplit config
train_array
datasplit_config = ...


# Create Architecture Config
architecture_config = CNNectomeUNetConfig(
    name="small_unet",
    input_shape=Coordinate(212, 212, 212),
    eval_shape_increase=Coordinate(72, 72, 72),
    fmaps_in=1,
    num_fmaps=8,
    fmaps_out=32,
    fmap_inc_factor=4,
    downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
    constant_upsample=True,
)

# Create Task Config

task_config = AffinitiesTaskConfig(
    name="AffinitiesPrediction",
    neighborhood=[(0,0,1),(0,1,0),(1,0,0)]
)


# Create Trainer Config

trainer_config = GunpowderTrainerConfig(
    name="gunpowder",
    batch_size=2,
    learning_rate=0.0001,
    augments=[
        SimpleAugmentConfig(),
        ElasticAugmentConfig(
            control_point_spacing=(100, 100, 100),
            control_point_displacement_sigma=(10.0, 10.0, 10.0),
            rotation_interval=(0, math.pi / 2.0),
            subsample=8,
            uniform_3d_rotation=True,
        ),
        IntensityAugmentConfig(
            scale=(0.25, 1.75),
            shift=(-0.5, 0.35),
            clip=False,
        )
    ],
    num_data_fetchers=20,
    snapshot_interval=10000,
    min_masked=0.15,
    min_labelled=0.1,
)

# Create Run Config

run_config = RunConfig(
    name="tutorial_run",
    task_config=task_config,
    architecture_config=architecture_config,
    trainer_config=trainer_config,
    datasplit_config=datasplit_config,
    repetition=0,
    num_iterations=100000,
    validation_interval=1000,
)

run = Run(run_config)

# Store configs

config_store = create_config_store()

config_store.store_datasplit_config(datasplit_config)
config_store.store_architecture_config(architecture_config)
config_store.store_task_config(task_config)
config_store.store_trainer_config(trainer_config)
config_store.store_run_config(run_config)

# Optional start training by config name:
train(run_config.name)